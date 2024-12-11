# DAmodel

Implementation of DAmodel from Boyd et al. (in prep). All data used in the paper can be found at: [https://zenodo.org/records/14339961](https://zenodo.org/records/14339961)

## Instructions

Clone repository: `git clone https://github.com/benboyd97/DAmodel.git`

Open folder: `cd DAmodel`

Download data: `wget https://zenodo.org/records/14339961/files/DA_WD_Boyd_et_al.zip -O DA_WD_Boyd_et_al.zip`

Unzip the data enter: `gunzip DA_WD_Boyd_et_al.zip`

Move data folder: `cp -r DA_WD_Boyd_et_al/data . `

To unzip the grid enter: `gunzip hubeny_grid.npz.gz`

Make conda environment: `conda create --name wd_env --file requirements.txt`

Activate conda environment: `conda activate wd_env`

To run the code type the command: `python DAmodel.py --name='test_name' --no_warm=500 --no_samps=500`
