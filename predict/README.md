# External prediction inputs and reports

Both prediction scripts use this folder as their default input directory. Place external `.tab` or `.mat` files directly here, then run either prediction command from the repository root:

```bash
python ffnn/predict_rev.py
python tcn/predict_tcn_fin.py
```

Each script loads its model, label encoder, and configuration from its own `outputs/` folder. These files must exist before prediction.

A command-line argument can select another input file or folder. For example, after placing separately obtained EXP `.mat` files in `data/exp/`:

```bash
python ffnn/predict_rev.py data/exp
python tcn/predict_tcn_fin.py data/exp
```

The Excel exports are always saved in this `predict/` folder, including when another input path is supplied: `predictions.xlsx` for FFNN and `predictions_tcn.xlsx` for TCN. Each later run overwrites that model's report. Input directories are searched without descending into subfolders.

Local prediction inputs and generated reports are excluded from Git. The included ADAMS example lives in `data/adams/`. EXP measurements are distributed separately.
