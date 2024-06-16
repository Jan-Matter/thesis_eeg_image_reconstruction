
# Setup of Repository:

- tested with python 3.7.6
- create env `python -m venv pyenv`
- activate env `pyenv\Scripts\activate`
- download dependencies `python -m pip install -r requirements.txt`

- create folder data/science_direct_eeg
- download from https://osf.io/xrfzq and place category_mat_manual.tsv into folder data/science_direct_eeg
- download from https://osf.io/qkgtf and place image_metadata.npy into folder data/science_direct_eeg
- download from https://osf.io/y63gw unzip and place test_images and train_images folder into folder data/science_direct_eeg
- download from https://osf.io/crxs4 unzip and place sub-<SUBJECT_NR> into folder data/science_direct_eeg

-download and place 'sd-v1-4.ckpt' https://huggingface.co/CompVis/stable-diffusion-v-1-4-original/tree/main from hugging face and place it into src/trained_models

- run preprocessing: `python src/preprocessors/eeg_preprocessor.py`
- create img ref db: `mkdir db` + `touch science_direct_eeg_db` + `alembic upgreat head`
- initialize sql db: `python src/eeg_science_direct_sql_db/db_loaders/image_db_loader.py`


# Reproduce Results
- follow steps from Setup of Repository
- experiments were tested on Quadro RTX 6000. In case an out of memory issue occur a reduction of batch size in configs might help!
- run all experiments in scripts/experiments e12 depends on e10 and e11 to be ran first
- display results using src/analysis_notebooks/results.ipynb

