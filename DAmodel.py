# Standard library imports
import argparse
import glob
import os
import re
import sys

# Third-party library imports
import astropy.constants as const
import astropy.table as at
import h5py
import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import pandas as pd
import spline_utils
from astropy.io import ascii
from get_phot import load_phot
from interpax import interp2d
from jax import device_put, random
from jax.lax import fori_loop
from numpy.lib.recfunctions import append_fields
from numpyro.diagnostics import hpdi
from numpyro.infer import MCMC, NUTS, Predictive, init_to_value
import numpyro.distributions as dist
from ruamel.yaml import YAML
from scipy.integrate import simpson
from scipy.interpolate import interp1d
from scipy.stats import linregress


yaml = YAML(typ="safe")


parser = argparse.ArgumentParser()

parser.add_argument("--name", type=str,default='test')
parser.add_argument("--no_warm", type=int, default=500)
parser.add_argument("--no_samps", type=int,default=500)
parser.add_argument("--no_chains", type=int,default=4)
parser.add_argument("--rv", type=str,default='pop')
parser.add_argument("--stis", type=str,default='True')
parser.add_argument("--redlaw", type=str,default='G23')


args = parser.parse_args()

run_name = args.name
Rv_prior = args.rv
redlaw = args.redlaw

stis_flag= True if args.stis=='True' else False

no_warm = args.no_warm
no_samps =args.no_samps
no_chains = args.no_chains


numpyro.set_host_device_count(int(no_chains))
jax.config.update('jax_enable_x64',True)


def f_lam(l):
    f = (const.c.to('AA/s').value / 1e23) * ((l) ** -2) * 10 ** (-48.6 / 2.5) * 1e23
    return f


grid_file = np.load('hubeny_grid.npz')
tgrid =grid_file['tgrid']
ggrid =grid_file['ggrid']
wd_flux = grid_file['flux']
wav=  grid_file['wav']
wav_=  jnp.array(grid_file['wav'])

ggrid = jnp.array(ggrid)

tgrid = jnp.array(tgrid)

wd_flux = jnp.array(wd_flux).T

def invKD_irr(x):
	"""
	Compute K^{-1}D for a set of spline knots.

	For knots y at locations x, the vector, y'' of non-zero second
	derivatives is constructed from y'' = K^{-1}Dy, where K^{-1}D
	is independent of y, meaning it can be precomputed and reused for
	arbitrary y to compute the second derivatives of y.

	Parameters
	----------
	x : :py:class:`numpy.array`
		Numpy array containing the locations of the cubic spline knots.
	
	Returns
	-------
	KD : :py:class:`numpy.array`
		y independednt matrix whose product can be taken with y to
		obtain a vector of second derivatives of y.
	"""
	n = len(x)

	K = np.zeros((n-2,n-2))
	D = np.zeros((n-2,n))

	K[0,0:2] = [(x[2] - x[0])/3, (x[2] - x[1])/6]
	K[-1, -2:n-2] = [(x[n-2] - x[n-3])/6, (x[n-1] - x[n-3])/3]

	for j in np.arange(2,n-2):
		row = j - 1
		K[row, row-1:row+2] = [(x[j] - x[j-1])/6, (x[j+1] - x[j-1])/3, (x[j+1] - x[j])/6]
	for j in np.arange(1,n-1):
		row = j - 1
		D[row, row:row+3] = [1./(x[j] - x[j-1]), -(1./(x[j+1] - x[j]) + 1./(x[j] - x[j-1])), 1./(x[j+1] - x[j])]
	
	M = np.zeros((n,n))
	M[1:-1, :] = np.linalg.solve(K,D)
	return M


uv_ind1 = (
wav_ < 2700
)  # Need to use separate UV term for F99 law below 2700AA
uv_ind2 = (wav_ < 2700) & ((1e4 / wav_) >= 5.9)
uv_ind3 = (1e4 / wav_[uv_ind1]) >= 5.9
uv_x = 1e4 / wav_[uv_ind1]


def spline_coeffs_irr_step(x_now, x, invkd):
    """
    Vectorized version of cubic spline coefficient calculator found in spline_utils

    Parameters
    ----------
    x_now: array-like
        Current x location to calculate spline knots for
    x: array-like
        Numpy array containing the locations of the spline knots.
    invkd: array-like
        Precomputed matrix for generating second derivatives. Can be obtained
        from the output of ``spline_utils.invKD_irr``.

    Returns
    -------

    X: Set of spline coefficients for each x knot

    """
    X = jnp.zeros_like(x)
    up_extrap = x_now > x[-1]
    down_extrap = x_now < x[0]
    interp = 1 - up_extrap - down_extrap

    h = x[-1] - x[-2]
    a = (x[-1] - x_now) / h
    b = 1 - a
    f = (x_now - x[-1]) * h / 6.0

    X = X.at[-2].set(X[-2] + a * up_extrap)
    X = X.at[-1].set(X[-1] + b * up_extrap)
    X = X.at[:].set(X[:] + f * invkd[-2, :] * up_extrap)

    h = x[1] - x[0]
    b = (x_now - x[0]) / h
    a = 1 - b
    f = (x_now - x[0]) * h / 6.0

    X = X.at[0].set(X[0] + a * down_extrap)
    X = X.at[1].set(X[1] + b * down_extrap)
    X = X.at[:].set(X[:] - f * invkd[1, :] * down_extrap)

    q = jnp.argmax(x_now < x) - 1
    h = x[q + 1] - x[q]
    a = (x[q + 1] - x_now) / h
    b = 1 - a
    c = ((a**3 - a) / 6) * h**2
    d = ((b**3 - b) / 6) * h**2

    X = X.at[q].set(X[q] + a * interp)
    X = X.at[q + 1].set(X[q + 1] + b * interp)
    X = X.at[:].set(X[:] + c * invkd[q, :] * interp + d * invkd[q + 1, :] * interp)

    return X


def spline_coeffs_irr(x_int, x, invkd, allow_extrap=True):
	"""
	Compute a matrix of spline coefficients.

	Given a set of knots at x, with values y, compute a matrix, J, which
	can be multiplied into y to evaluate the cubic spline at points
	x_int.

	Parameters
	----------
	x_int : :py:class:`numpy.array`
		Numpy array containing the locations which the output matrix will
		interpolate the spline to.
	x : :py:class:`numpy.array`
		Numpy array containing the locations of the spline knots.
	invkd : :py:class:`numpy.array`
		Precomputed matrix for generating second derivatives. Can be obtained
		from the output of ``invKD_irr``.
	allow_extrap : bool
		Flag permitting extrapolation. If True, the returned matrix will be
		configured to extrapolate linearly beyond the outer knots. If False,
		values which fall out of bounds will raise ValueError.
	
	Returns
	-------
	J : :py:class:`numpy.array`
		y independednt matrix whose product can be taken with y to evaluate
		the spline at x_int.
	"""
	n_x_int = len(x_int)
	n_x = len(x)
	X = np.zeros((n_x_int,n_x))

	if not allow_extrap and ((max(x_int) > max(x)) or (min(x_int) < min(x))):
		raise ValueError("Interpolation point out of bounds! " + 
			"Ensure all points are within bounds, or set allow_extrap=True.")
	
	for i in range(n_x_int):
		x_now = x_int[i]
		if x_now > max(x):
			h = x[-1] - x[-2]
			a = (x[-1] - x_now)/h
			b = 1 - a
			f = (x_now - x[-1])*h/6.0

			X[i,-2] = a
			X[i,-1] = b
			X[i,:] = X[i,:] + f*invkd[-2,:]
		elif x_now < min(x):
			h = x[1] - x[0]
			b = (x_now - x[0])/h
			a = 1 - b
			f = (x_now - x[0])*h/6.0

			X[i,0] = a
			X[i,1] = b
			X[i,:] = X[i,:] - f*invkd[1,:]
		else:
			q = np.where(x[0:-1] <= x_now)[0][-1]
			h = x[q+1] - x[q]
			a = (x[q+1] - x_now)/h
			b = 1 - a
			c = ((a**3 - a)/6)*h**2
			d = ((b**3 - b)/6)*h**2

			X[i,q] = a
			X[i,q+1] = b
			X[i,:] = X[i,:] + c*invkd[q,:] + d*invkd[q+1,:]

	return X

@jax.jit
def matrix_mult(a,b):
    return a @ b


redlaw_name = redlaw
x = "default"

with open("dust_files/BAYESN_"+redlaw_name+".YAML") as file:
    redlaw_params = yaml.load(file)


redlaw_rv_coeffs = device_put(
    jnp.array(redlaw_params.get("RV_COEFFS", [[1]]))
)
redlaw_num_knots = len(redlaw_params.get("L_KNOTS", [1]))
redlaw_min_order = redlaw_params.get("MIN_ORDER", 0)
n_regimes = len(redlaw_params.get("REGIME_WLS", [1]))
ones = jnp.ones((n_regimes, 1))
zeros = jnp.zeros((n_regimes, 1))
redlaw = {}
for var in "AB":
    redlaw[var] = jnp.zeros((len(wav_), 1))
units = redlaw_params.get("UNITS", "inverse microns")
if x == "default":
    x = wav_
if "micron" in units:
    x = x / 1e4
if "inverse" in units:
    x = 1 / x
if "L_KNOTS" in redlaw_params:
    redlaw_xk = jnp.array(redlaw_params["L_KNOTS"])
    redlaw["B"] = spline_coeffs_irr(
        x, redlaw_xk, invKD_irr(redlaw_xk)
    )
for i in range(n_regimes):
    wl_range = redlaw_params.get("REGIME_WLS", [[0, 10]])[i]
    idx = jnp.where((wl_range[0] <= x) & (x < wl_range[1]))
    if not idx[0].shape[0]:
        continue
    mod_x = x[idx]
    for var in "AB":
        if var == "B" and "L_KNOTS" in redlaw_params:
            continue
        poly = jnp.array(redlaw_params.get(f"{var}_POLY_COEFFS", ones)[i])
        rem = jnp.array(redlaw_params.get(f"{var}_REMAINDER_COEFFS", zeros)[i])
        div = jnp.array(redlaw_params.get(f"{var}_DIVISOR_COEFFS", zeros)[i])
        # Asymmetric Drude for G23
        # Symmetric Drude profiles converted to polynomials
        amp, center, fwhm, asym = jnp.array(
            redlaw_params.get(f"{var}_DRUDE_PARAMS", jnp.zeros((n_regimes, 4)))[
                i
            ]
        )
        gamma = 2 * fwhm / (1 + jnp.exp(asym * (1 / mod_x - center)))
        redlaw[var] = (
            redlaw[var]
            .at[idx, 0]
            .add(
                mod_x ** redlaw_params.get("REGIME_EXP", zeros)[i]
                * (
                    jnp.polyval(poly, mod_x)
                    + jnp.nan_to_num(
                        jnp.polyval(rem, mod_x) / jnp.polyval(div, mod_x),
                        posinf=0,
                        neginf=0,
                    )
                )
                * jnp.nan_to_num(
                    amp
                    * (gamma / center) ** 2
                    / (
                        (1 / (mod_x * center) - mod_x * center) ** 2
                        + (gamma / center) ** 2
                    ),
                    nan=1,
                    posinf=0,
                    neginf=0,
                )
            )
        )
redlaw_ax = device_put(redlaw["A"])
redlaw_bx = device_put(redlaw["B"])

def get_axav( RV, num_batch=1):
    yk = jnp.zeros((num_batch, redlaw_num_knots))
    ones = jnp.ones((1, num_batch))

    def get_knot_value(i, yk):
        return yk.at[:, i].set(
            jnp.polyval(redlaw_rv_coeffs[i], RV, unroll=16)
            * RV**redlaw_min_order
        )

    yk = fori_loop(0, redlaw_num_knots, get_knot_value, yk)
    ax = (redlaw_ax @ ones).T
    bx = (redlaw_bx @ yk.T).T
    return ax + bx / jnp.array(RV)[..., None]





def extinction_jax(RV,AV):
    num_batch = RV.shape[-1]
    J_t_map = jax.jit(
            jax.vmap(spline_coeffs_irr_step, in_axes=(0, None, None))
        )
    
    return AV[..., None] * get_axav(RV, num_batch)




detail = 'cubic'

@jax.jit
def grid(tq,gq,t,g,f):
    
    
    
    return interp2d(tq,gq,t,g,f,method=detail)


wav = jnp.array(wav)

"""
BayeSN Spline Utilities. Defines a set of functions which carry out the
2D spline operations essential to BayeSN.
"""




@jax.jit
def sed_maker(t_eff,log_g,av,rv):
    interp_sed=grid(t_eff,log_g,tgrid,ggrid,wd_flux)
    a_s=extinction_jax(rv,av)
    interp_sed = interp_sed * 10**(-0.4*a_s)


    return interp_sed

filters = ['filters/UVIS/wfc3_uvis2_f275w.txt',
           'filters/UVIS/wfc3_uvis2_f336w.txt',
           'filters/UVIS/wfc3_uvis2_f475w.txt',
           'filters/UVIS/wfc3_uvis2_f625w.txt',
           'filters/UVIS/wfc3_uvis2_f775w.txt',
            'filters/IR/wfc3_ir_f160w.txt']


band_weights = []
zps = []
mins = []
maxs = []

for i,filt in enumerate(filters):
    R = np.loadtxt(filt)
    

    
    
    T = np.zeros(len(wav))
    
    
    min_id=np.argmin((wav-np.min(R[:,0]))**2)
    max_id=np.argmin((wav-np.max(R[:,0]))**2)
    
    T[min_id:max_id]= jnp.interp(wav[min_id:max_id], R[:, 0], R[:, 1])
    
    

    
    
    dlambda = jnp.diff(wav)
    dlambda = jnp.r_[dlambda, dlambda[-1]]

    num = wav * T * dlambda
    denom = jnp.sum(num)
    band_weight = num / denom
    band_weights.append(band_weight)

    
    lam = R[:, 0]
    zp_sed = f_lam(lam)

    int1 = simpson(lam * zp_sed * R[:, 1], lam)
    int2 = simpson(lam * R[:, 1], lam)
    zp = 2.5 * np.log10(int1 / int2)
    zps.append(zp)

    
band_weights = jnp.array(band_weights)
zps = jnp.array(zps)



@jax.jit
def get_model_mag(flux_grid, band_weights):
    model_flux = band_weights @ flux_grid

    model_mag = -2.5 * jnp.log10(model_flux) + zps[:, None]
    return model_mag





def gauss(x,sigma):
    
    return 1/(sigma*jnp.sqrt(2*jnp.pi)) * jnp.exp(-0.5*(x/sigma)**2)
    
    
              
def convolve(signal,window):
    
    return jnp.convolve(signal,window ,mode='same')


@jax.jit
def v_convolve(signal,window):
    
    
    return jax.vmap(convolve ,in_axes=(0,0))(signal,window)
        
        


def smooth(signal,fwhm):
    sig=fwhm/jnp.sqrt(8*jnp.log(2))


    grid = np.arange(-50,50,1)


    sig = jnp.repeat(sig.reshape(-1,1),grid.shape[0],axis=1)
    

    grid = jnp.repeat(grid.reshape(1,-1),sig.shape[0],axis=0)



    window = gauss(grid,sig)



    window = jnp.where(grid<-4*sig,0,window)

    window = jnp.where(grid>4*sig,0,window)

    window = window/jnp.sum(window,axis=1)[...,None]




    return v_convolve(signal, window)


@jax.jit
def eval_gauss(mu,cov_matrix,x):
    
    return dist.MultivariateNormal(mu,cov_matrix).log_prob(x)





@jax.jit
def syn_phot_sed(t_eff,log_g,av,rv):
    
    seds = sed_maker(t_eff,log_g,av,rv)
    
    phot_vals= get_model_mag(seds.T, band_weights)
    
    phot_vals = phot_vals.T
    

    return phot_vals,seds

@jax.jit
def ext(arr):


    return arr




def phot_cal(mag_app_i,model_samp_mags,standard_mags,sample_mags,sig_i,sig_j,standard_idx,sample_idx,stand_cycles=0,samp_cycles=0,zpt_est=24):

    n_bands= mag_app_i.shape[0]

    alpha = numpyro.sample("alpha",dist.Normal(loc=0,scale=0.01))
    beta = 15

    alpha_arr = jnp.append(jnp.zeros(n_bands-1),jnp.array([alpha]))


    with numpyro.plate("plate_b", n_bands):

        sig_int_b =  numpyro.sample("sig_intrinsic", dist.HalfCauchy(1))
        zpt_b = numpyro.sample("zeropoint",dist.Normal(loc=zpt_est,scale=1))
        c20_offset = numpyro.sample("c20_offset",dist.Normal(loc=0,scale=1))
        c20_offset2 = numpyro.sample("c20_offset2",dist.Normal(loc=0,scale=1))
        
        nu_b = numpyro.sample("nu", dist.HalfCauchy(5))


    
    n_sample_obj= len(np.unique(sample_idx))-1


    with numpyro.plate("plate_j", n_sample_obj*n_bands):
        mag_inst_j = model_samp_mags - zpt_b.reshape(n_bands,1)+  alpha_arr.reshape(n_bands,1)*(model_samp_mags-beta)


    n_standard_obj= len(np.unique(standard_idx))-1

    with numpyro.plate("plate_i", n_standard_obj*n_bands):

        mag_inst_i = mag_app_i -zpt_b.reshape(n_bands,1)+  alpha_arr.reshape(n_bands,1)*(mag_app_i-beta)

    n_sample_obs= sample_mags.shape[1]

    m_j = mag_inst_j[jnp.arange(n_bands).astype(int)[:,None], sample_idx] -(samp_cycles==1).astype(float)*c20_offset.reshape(n_bands,1)-(samp_cycles==2).astype(float)*c20_offset2.reshape(n_bands,1)
    mask1 = sample_idx!=1000

    with numpyro.plate("data_k",n_sample_obs):
        with numpyro.plate("data_j", n_bands):
    
            full_var_j = (sig_int_b.reshape(n_bands,1)**2.+sig_j**2.)**0.5

            with numpyro.handlers.mask(mask=ext(mask1)):
                numpyro.sample("m_j", dist.StudentT(loc=ext(m_j),df=ext(nu_b.reshape(n_bands,1)),scale=ext(full_var_j)), obs=ext(sample_mags))

    n_standard_obs= standard_mags.shape[1]

    m_i = mag_inst_i[jnp.arange(n_bands).astype(int)[:,None], standard_idx] -(stand_cycles==1).astype(float)*c20_offset.reshape(n_bands,1)  -(stand_cycles==2).astype(float)*c20_offset2.reshape(n_bands,1)


    mask2= standard_idx!=1000
    with numpyro.plate("data_l",n_standard_obs):
        with numpyro.plate("data_i", n_bands):
            full_var_i = (sig_int_b.reshape(n_bands,1)**2. + sig_i**2.)**0.5
            with numpyro.handlers.mask(mask=ext(mask2)):

                numpyro.sample("m_i", dist.StudentT(loc=ext(m_i),df=ext(nu_b.reshape(n_bands,1)),scale=ext(full_var_i)), obs=ext(standard_mags))


def wd_model(wav,true_stand_mags=None,stand_mags=None,samp_mags=None,stand_err=None,samp_err=None,stand_ids=None,samp_ids=None,
            cycle_stand_ids=None,cycle_samp_ids=None,zpt_est=None,
             spec=None,spec_err=None,wave_mask=None,spec_res=1,n_bands=6,spline_coeffs=None,
             spec_stis=None,spec_err_stis=None,stis_id=None,spline_coeffs_stis=None):
    n_wd = spec.shape[1]
    wav_stis = wav[481:4496][::2]
    wav = wav[4999:8600]
    wav = wav

    if Rv_prior=='pop':
        mu_Rv = numpyro.sample('mu_Rv',dist.ImproperUniform(dist.constraints.greater_than(1.2),(),event_shape=()))
        sigma_Rv =  numpyro.sample("sigma_Rv", dist.HalfCauchy(1))
        tau =numpyro.sample('tau', dist.ImproperUniform(dist.constraints.greater_than(0),(),event_shape=()))
    
    else:
        tau =0.1

    with numpyro.plate("plate_s",n_wd):

        
        t_eff = numpyro.sample('t_eff',dist.Uniform(15000,70000))
        log_g = numpyro.sample('log_g',dist.Uniform(7,9.5))
        Av = numpyro.sample('Av',dist.Exponential(1/tau))



        if Rv_prior == 'const':
            Rv = jnp.array([3.1]*n_wd)
        elif Rv_prior=='normal':
            Rv =  numpyro.sample('Rv',dist.TruncatedNormal(3.1,0.18,low=1.2,high=100))
        elif Rv_prior=='uniform':
            Rv =  numpyro.sample('Rv',dist.Uniform(1.2,7))
        elif Rv_prior=='pop':
            Rv =  numpyro.sample('Rv',dist.TruncatedNormal(mu_Rv,sigma_Rv,low=1.2,high=100))


        dl = numpyro.sample('dl',dist.Uniform(0,2e4)) 
        fwhm = numpyro.sample('fwhm',dist.TruncatedNormal(0,8,low=0,high=25))
        


        mu = numpyro.sample('mu',dist.Uniform(50,65))
        
        
        syn_phot,syn_sed = syn_phot_sed(t_eff,log_g,Av,Rv)





        syn_sed_= syn_sed[:,4999:8600]/(4*jnp.pi*dl[...,None]**2)
        
        sed = syn_sed_.at[:,6:syn_sed_.shape[1]-9].set(smooth(syn_sed_,fwhm/spec_res)[:,7:syn_sed_.shape[1]-8])

   
        with numpyro.plate("plate_knots",spline_coeffs.shape[1]):
            eps= numpyro.sample('eps', dist.Normal(0,1))



        with numpyro.plate("pix",wav.shape[0]):
            with numpyro.handlers.mask(mask=wave_mask==1.):
                

                numpyro.sample("spec_obs", dist.Normal((sed.T+matrix_mult(spline_coeffs,eps)),spec_err), obs=spec)

    if stis_flag:

        syn_sed_stis_= syn_sed[stis_id.astype(int),481:4496][:,::2]

        with numpyro.plate("plate_stis",len(stis_id)):

            fwhm_stis = numpyro.sample('fwhm_stis',dist.TruncatedNormal(0,8,low=0,high=25))
            dl_stis = numpyro.sample('dl_stis',dist.Uniform(0,1e5))
            syn_sed_stis=syn_sed_stis_/(4*jnp.pi*dl_stis[...,None]**2)  
            sed_stis = syn_sed_stis.at[:,6:syn_sed_stis.shape[1]-9].set(smooth(syn_sed_stis,fwhm_stis)[:,7:syn_sed_stis.shape[1]-8])

            with numpyro.plate("plate_knot_stis",spline_coeffs_stis.shape[1]):
                eps_stis= numpyro.sample('eps_stis', dist.Normal(0,1))



            with numpyro.plate("pix_stis",wav_stis.shape[0]):
                with numpyro.handlers.mask(mask=np.ones_like(spec_stis)==1.):
            
                    morphed_stis_sed = sed_stis.T+matrix_mult(spline_coeffs_stis,eps_stis)
                    numpyro.sample("spec_obs_stis", dist.Normal(morphed_stis_sed,spec_err_stis), obs=spec_stis)

    syn_phot = syn_phot + mu.reshape(-1,1)

    phot_cal(true_stand_mags,syn_phot.T,stand_mags,samp_mags,stand_err,samp_err,stand_ids,samp_ids,cycle_stand_ids,cycle_samp_ids,zpt_est)
    

        
            

spec_dir = [
    'mmt/g191b2b-20150124-total.flm',
    'mmt/gd153-20150518-total.flm',
    'soar/gd71-20170223-total.flm',
    'gemini/sdssj010322-20131129-total.flm',
    'gemini/sdssj022817-20131013-total.flm',
    'mmt/sdssj024854-20151011.5-total.flm',
    'mmt/sdssj072752-20150124-total.flm',
    'gemini/sdssj081508-20130214-total.flm',
    'soar/sdssj102430-20170223-total.flm',
    'mmt/sdssj111059-20150124-total.flm',
    'mmt/sdssj111127-20150124-total.flm',
    'gemini/sdssj120650-20130310-total.flm',
    'gemini/sdssj121405-20150218-total.flm',
    'gemini/sdssj130234-20130215-total.flm',
    'gemini/sdssj131445-20130309-total.flm',
    'gemini/sdssj151421-20130317-total.flm',
    'mmt/sdssj155745-20150124-total.flm',
    'mmt/sdssj163800-20150518-total.flm',
    'gemini/sdssj181424-20150427-total.flm',
    'mmt/sdssj210150-20150518-total.flm',
    'gemini/sdssj232941-20150917-20151106-total.flm',
    'gemini/sdssj235144-20150915-total.flm',
    'soar/atlas020.503022-20161007-total.flm',
    'soar/sssj023824-20161007-total.flm',
    'soar/sssj045822-20170222-total.flm',
    'soar/sssj054114-20161007-total.flm',
    'soar/sssj063941-20170223-total.flm',
    'soar/sssj095657-20170222-total.flm',
    'soar/sssj105525-20170223-total.flm',
    'soar/sssj143459-20170222-total.flm',
    'soar/sssj183717-20161006-total.flm',
    'soar/sssj193018-20161006-total.flm',
    'soar/wd1203-20170222-total.flm',
    'soar/wd1529-772-20160212-total.flm',
    'soar/wd2314-293-20161005-total.flm'

]
    

spec_wave=np.arange(3400,7001,1)
spec_flux = np.empty((0,len(spec_wave)))
spec_err = np.empty((0,len(spec_wave)))
spec_res = np.array([])

dl_guess = np.array([])
names = []


wave_mask = np.empty((0,len(spec_wave)))
cov_mask = np.empty((0,len(spec_wave),len(spec_wave)))


for i in range(len(spec_dir)):
    file_path = 'data/spectroscopy/'+spec_dir[i]

    data_table = ascii.read(file_path)



    single_spec_wav=jnp.array(data_table['wave'])
    
    
    upper =len(single_spec_wav)
    if len(np.where(single_spec_wav==7000)[0])>0:
        upper = np.where(single_spec_wav==7000)[0][0]+1
        
        
    single_spec_wav=jnp.array(data_table['wave'][:upper])
           
    res=np.unique(np.diff(single_spec_wav))[0]
    spec_res = jnp.append(spec_res,np.max([float(1.),res]))

    start_id= np.where(spec_wave==single_spec_wav[0])[0][0]
    
    
    end_id= np.where(spec_wave==single_spec_wav[-1])[0][0]
    
    these_ids = start_id+np.arange(int(len(single_spec_wav)*res))[::int(res)]
    
    single_mask = np.zeros(len(spec_wave))
    

    single_mask[these_ids] = 1
    
    
    single_spec_flux = np.zeros(len(spec_wave))
    
    
    single_spec_flux[these_ids] = np.array(data_table['flux'][:upper])
    
    single_spec_err = np.zeros(len(spec_wave)) + np.median(data_table['flux_err'][:upper])
    
    single_spec_err[these_ids] = np.array(data_table['flux_err'][:upper])

    if len(single_spec_wav)>0:   

    

        wave_mask = jnp.append(wave_mask,single_mask.reshape(1,-1),axis=0)
    
        spec_flux= jnp.append(spec_flux,single_spec_flux.reshape(1,-1),axis=0)
        
        spec_err= jnp.append(spec_err,single_spec_err.reshape(1,-1),axis=0)
        
        names += [spec_dir[i][spec_dir[i].find('/')+1:spec_dir[i].find('-')]]

        dl_guess=np.append(dl_guess,2000*np.sqrt(1/np.median(np.array(data_table['flux'][:upper]))))
    




directory = 'data/spectroscopy/stis'


files = os.listdir(directory)

files = np.unique([file for file in files if file.endswith('.mrg')])


names_swap = {'wdfs1837':'sssj183717','wdfs1814':'sdssj181424','wdfs1930':'sssj193018','wdfs1214':'sdssj121405',
             'wdfs0956':'sssj095657','wdfs1557':'sdssj155745','wdfs0639':'sssj063941','wdfs1206':'wd1203',
              'wdfs1535':'wd1529','wdfs1110':'sdssj111059','wdfs1055':'sssj105525','wdfs1434':'sssj143459',
              'wdfs0122':'atlas020.503022','wdfs0458':'sssj045822','wdfs2317':'wd2314','wdfs1302':'sdssj130234',
               'wdfs0248':'sdssj024854','wdfs1514':'sdssj151421','wdfs2351':'sdssj235144'}
names_stis = np.array([])

wav_stis = np.arange(1141,3149,1)

dl_guess_stis = np.array([])

spec_flux_stis = np.empty((0,len(wav_stis)))
spec_err_stis = np.empty((0,len(wav_stis)))
stis_id = np.array([])
for k,file_ in enumerate(files):
    
    first_underscore_index = file_.find('_')
    
    if first_underscore_index == -1:
        first_underscore_index = file_.find('.')
        
    
    if file_[:first_underscore_index] in list(names_swap.keys()):
        
        name_stis = names_swap[file_[:first_underscore_index]]
        names_stis=np.append(names_stis,name_stis)
    
        file_path = directory+'/'+file_

        with open(file_path, 'r') as file:
            content = file.read()

        # Split the content into lines
        lines = content.splitlines()

        # Find the line where the data starts
        data_start_index = None
        for i, line in enumerate(lines):
            if re.match(r'\s*WAVELENGTH\s+COUNT-RATE\s+FLUX\s+STAT-ERROR\s+SYS-ERROR\s+NPTS\s+TIME\s+QUAL', line):
                data_start_index = i + 1
                break

        # Extract the data lines
        data_lines = lines[data_start_index:]

        # Parse the data into a list of dictionaries
        data = []
        for line in data_lines:
            parts = re.split(r'\s+', line.strip())
            if len(parts) == 8:
                try:
                    data.append({
                        'WAVELENGTH': float(parts[0]),
                        'COUNT-RATE': float(parts[1]),
                        'FLUX': float(parts[2]),
                        'STAT-ERROR': float(parts[3]),
                        'SYS-ERROR': float(parts[4]),
                        'NPTS': int(parts[5]),
                        'TIME': float(parts[6]),
                        'QUAL': int(parts[7]),
                    })
                except ValueError:
                    # Skip lines that do not contain valid numeric data
                    continue

        # Create a DataFrame from the parsed data
        df = pd.DataFrame(data)
        x = np.array(df['WAVELENGTH'])[:-1]
        y = np.array(df['FLUX'])[:-1]
        

        y_err = np.array(np.sqrt(df['STAT-ERROR']**2+df['SYS-ERROR']**2))[:-1]    
        x=x[y>0]
        y_err=y_err[y>0]
        y=y[y>0]
        
        
        y_err = y_err/(10*np.median(y))
        y= y/(10*np.median(y))
        
  
        f = interp1d(x, y, kind='linear')


        spec_flux_stis =np.append(spec_flux_stis,f(wav_stis)[None,...],axis=0)
        
        f2 = interp1d(x, y_err, kind='linear')

        spec_err_stis =np.append(spec_err_stis,f2(wav_stis)[None,...],axis=0)
        
        dl_guess_stis=np.append(dl_guess_stis,10000*np.sqrt(1/np.median(np.array(y))))

        stis_id = np.append(stis_id,np.where(np.array(names)==name_stis)[0])


stand_ids,samp_ids,stand_mags,samp_mags,stand_err,samp_err,cycle_stand_ids,cycle_samp_ids = load_phot('data/photometry',names)



mu_guess = np.array([])
for i in range(len(names)):
    mu_guess=jnp.append(mu_guess,jnp.mean(samp_mags[samp_ids==i])+65)


knots = jnp.linspace(np.min(spec_wave),np.max(spec_wave),10)
invkd= invKD_irr(knots)

spline_coeffs=spline_coeffs_irr(wav[4999:8600], knots,invkd, allow_extrap=True)

knots_stis = jnp.linspace(jnp.min(wav_stis),jnp.max(wav_stis),10)
invkd_stis= invKD_irr(knots_stis)

spline_coeffs_stis=spline_coeffs_irr(wav_stis, knots_stis,invkd_stis, allow_extrap=True)





#CALPSEC 2020 mags G191B2B, GD153 and GD71 from F275W to F160W
true_mags = jnp.array([[10.47152  , 12.181925 , 11.965953 ],
       [10.872241 , 12.549763 , 12.314495 ],
       [11.488517 , 13.0852995, 12.783156 ],
       [12.021632 , 13.58967  , 13.267444 ],
       [12.438829 , 13.994901 , 13.664389 ],
       [13.879408 , 15.410585 , 15.061585 ]])


zpt_est = np.zeros(6)
for i in range(zpt_est.shape[-1]):
    diff = np.array([])
    for j in range(stand_ids[i,:].shape[-1]):

        if stand_ids[i,j]<len(true_mags[i,:]):
            if cycle_stand_ids[i,j]==0:
                diff = np.append(diff,[true_mags[i,stand_ids[i,j]]-stand_mags[i,j]])
    zpt_est[i] = np.mean(diff)

zpt_est = jnp.array(zpt_est)

stand_ids=stand_ids.at[:2,:].set(np.ones_like(stand_ids.at[:2,:])*1000)
stand_ids=stand_ids.at[3:,:].set(np.ones_like(stand_ids.at[3:,:])*1000)

nuts_kernel = NUTS(wd_model,adapt_step_size=True,
init_strategy=init_to_value(values={'t_eff':jnp.array([40000.]*len(names)),'log_g':jnp.array([7.8]*len(names)),'Av':jnp.array([0.1]*len(names)),'Rv':jnp.array([3.1]*len(names)),'mu_Rv':jnp.array([3.1]),'sigma_Rv':jnp.array([1.]),'tau':jnp.array([0.1]),'mu':mu_guess,
'dl':jnp.asarray(dl_guess),'dl_stis':jnp.asarray(dl_guess_stis),'fwhm':jnp.array([np.max([float(1),res])]*len(names)),'fwhm_stis':jnp.array([0.5]*len(names_stis)),'eps':jnp.zeros((len(knots),len(names))),'eps_stis':jnp.zeros((len(knots_stis),len(names_stis))),'zeropoint':zpt_est,'c20_offset':jnp.array([0.]*samp_mags.shape[0]),'c20_offset2':jnp.array([0.]*samp_mags.shape[0]),
'sig_intrinsic':jnp.array([0.001]*samp_mags.shape[0]),'nu':jnp.array([7.]*samp_mags.shape[0]),'alpha':jnp.array([0.])}))

mcmc = MCMC(nuts_kernel, num_samples=no_samps, num_warmup=no_warm,num_chains=no_chains)

rng_key = random.PRNGKey(0)

print('mcmc')


mcmc.run(rng_key, wav,true_mags,stand_mags,samp_mags,stand_err,samp_err,stand_ids,samp_ids,cycle_stand_ids,cycle_samp_ids,zpt_est,spec=spec_flux.T,spec_err=spec_err.T,wave_mask=wave_mask.T,spec_res=spec_res,spline_coeffs=spline_coeffs,spec_stis=spec_flux_stis.T,spec_err_stis=spec_err_stis.T,stis_id=stis_id,spline_coeffs_stis=spline_coeffs_stis)



mcmc.print_summary()


posterior_samples = mcmc.get_samples()

np.savez('chains/'+run_name+'.npz',**posterior_samples)
