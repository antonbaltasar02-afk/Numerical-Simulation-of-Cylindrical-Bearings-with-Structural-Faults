# Experimental measurements for external prediction

Place experimental `.mat` files directly in this folder. EXP measurements are distributed separately and are not included in this GitHub repository. A Zenodo deposit is planned; the DOI and download link will be recorded in [`../README.md`](../README.md) once published.

These measurements are used for external prediction with models trained on ADAMS. Prepare the model checkpoint, label encoder, and configuration as described in the model README, then run from the repository root:

```bash
python ffnn/predict_rev.py data/exp
python tcn/predict_tcn_fin.py data/exp
```

Only files directly in this folder are loaded. Excel prediction reports are saved in the repository's `predict/` folder. See [`../README.md`](../README.md) for MATLAB channel selection, sampling metadata, and file-format details.

Raw EXP files and dataset archives in this folder are excluded from Git. Keep this README in the repository so the input folder and its instructions are available after cloning.
