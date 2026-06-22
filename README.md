# VMSI Lab (for Colab)

Self-contained version of the VMSI / CAP / FFT / mechano-transcriptomics lab.
Everything needed is bundled here: the notebook, the minimal `src/` pipeline, the `cellcap` package, and the `dataset1`/`dataset2` data. Participants run it on **Google Colab with nothing installed on their own machine**.

## Contents

```
src/                         segmentation + VMSI inference pipeline
cellcap/                     synthetic CAP tissue generator
reproduce_data/dataset1/     tensionmap_res.csv, imputed_counts.csv, adj_mat.csv
dataset1.tif, dataset2.tif   real segmentation mask
fft_complete.parquet         sampled FFT spectra for the noise injection
```

## Run on Google Colab

Download and open the notebook from the provided link. The notebook's **first cells** setup the enviroment and installs the only two missing packages (`nlopt`, `pyfftw`) everything else (numpy, pandas, scipy, scikit-image, scikit-learn, statsmodels, numba, OpenCV, seaborn, tqdm) is already on Colab.
   
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1wTTd4Z0T7CO0IUvKcuuHegfc69oC219j)

