# VMSI Lab — standalone (Google Colab)

Self-contained version of the VMSI / CAP / FFT / mechano-transcriptomics lab.
Everything needed is bundled here: the notebook, the minimal `src/` pipeline,
the `cellcap` package, and the `dataset1` data. Participants run it on **Google
Colab with nothing installed on their own machine**.

## Contents

```
vmsi_lab_colab.ipynb        the lab notebook
src/                        segmentation + VMSI inference pipeline
cellcap/                    synthetic CAP tissue generator (private package)
reproduce_data/dataset1/    tensionmap_res.csv, imputed_counts.csv, adj_mat.csv
dataset1.tif                real segmentation mask
fft_complete.parquet        sampled FFT spectra for the noise injection
```

## Run on Google Colab

The notebook's **first cell** installs the only two missing packages
(`nlopt`, `pyfftw`) — everything else (numpy, pandas, scipy, scikit-image,
scikit-learn, statsmodels, numba, OpenCV, seaborn, tqdm) is already on Colab.

Two ways to open it:

1. **One click** — the "Open in Colab" badge below. The first cell clones this
   repo into the Colab session and steps into it automatically.
2. **Manual** — in a fresh Colab notebook run:
   ```python
   !git clone --depth 1 https://github.com/USER/REPO.git
   %cd REPO
   ```
   then open `vmsi_lab_colab.ipynb` and run all cells.

> Before sharing, edit `REPO_URL` in the notebook's first cell and the badge URL
> below to point to your repository.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/USER/REPO/blob/main/vmsi_lab_colab.ipynb)

## Run locally

```bash
pip install nlopt pyfftw numpy pandas scipy scikit-image scikit-learn statsmodels numba opencv-python seaborn tqdm
jupyter lab vmsi_lab_colab.ipynb   # open from inside this folder
```

The first cell detects that `src/` and `cellcap/` are already present and skips
the clone.

## Publish this folder to GitHub (one-off, by the organiser)

All files are < 100 MB (largest is `dataset1.tif`, 32 MB), so **plain git** works
— no Git LFS needed. From inside this folder:

```bash
git init
git add .
git commit -m "VMSI lab standalone"
git branch -M main
# create a PUBLIC repo on github.com (a public repo can be cloned without auth),
# then:
git remote add origin https://github.com/USER/REPO.git
git push -u origin main
```

Then:
- replace `USER/REPO` in `REPO_URL` (first notebook cell) and in the badge URL above;
- the repo is ~62 MB and **public**, so the bundled data becomes public — only
  ship data you are allowed to publish;
- this is meant as a **temporary** repo: delete it after the lab.
