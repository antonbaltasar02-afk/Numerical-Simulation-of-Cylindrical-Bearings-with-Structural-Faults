# ADAMS and experimental datasets

TCN and FFNN share the input directories in this folder. ADAMS is used for training, validation, and internal test evaluation. EXP is used for external prediction with the separate prediction scripts.

Keep raw files directly inside `adams/` and `exp/`; the loaders do not descend into subfolders. Preserve ADAMS filenames because training labels are inferred from their stems.

## Included data and external measurements

`adams/IR_03_OR_03.tab` is included as an ADAMS training-format example with the `IR_03_OR_03` compound-fault label.

EXP measurements are distributed separately. The `exp/` folder contains a README with setup instructions. Place externally obtained `.mat` files directly in `data/exp/` before running prediction. The full ADAMS and EXP release is planned for Zenodo; the DOI and download link are pending.

## ADAMS format

The `.tab` loaders expect a tab-separated header on line 7 and numerical rows from line 8 onward. Header names are reduced to their final dot-separated components, including `TIME` and `Q`. Both models use `Q` and remove duplicate `TIME` rows, keeping the last occurrence.

The configured ADAMS sampling rate is 10,000 Hz, consistent with the example's 0.0001 s sampling interval. FFNN uses the configured ADAMS shaft frequency of 9.8 Hz for its order spectrum.

The example supplies these training, validation, and test window counts with default settings:

| Model | Training | Validation | Test |
| --- | --- | --- | --- |
| TCN | 107 | 26 | 121 |
| FFNN | 86 | 21 | 97 |

The single included ADAMS file represents one class. Use the full ADAMS release for multi-class training.

## EXP format and prediction

The reference measurement `BPFI_05_EXP.mat` contains `Sample_rate = 10000` Hz and multiple signal channels with their time arrays. Each prediction script imports its MATLAB loader from the corresponding training module:

| Model | Channel selected from `BPFI_05_EXP.mat` | Prediction windows |
| --- | --- | --- |
| FFNN | `Data1_AI_4_AI_4_minus_Housing_Y` | Non-overlapping 1 s windows |
| TCN | `Data1_AI_1_AI_1_minus_Probe_Y` | 1 s windows with a 0.2 s stride |

Both loaders multiply the selected MATLAB channel by `9.81 * 1000`. Sample rate is read from metadata when present; otherwise `CWRU_FS = 12000` is used. Shaft frequency is read from an `RPM` field or an `RS<number>Hz` filename token, with a 10 Hz fallback. The reference EXP file uses that fallback. FFNN uses shaft frequency in the order spectrum; TCN uses time-domain windows.

The separate inference preprocessing does not apply the training amplitude-rejection rule. Prediction class names come from the trained label encoder; prediction input filenames do not add classes to the model.

After placing separately obtained EXP `.mat` files in `data/exp/` and preparing the saved models, run from the repository root:

```bash
python ffnn/predict_rev.py data/exp
python tcn/predict_tcn_fin.py data/exp
```

Reports are saved in `predict/`, independently of the selected input folder.

## Full datasets on Zenodo

**Publication status:** deposit pending. A DOI and download link will be added here when the full datasets are published.

The intended downloadable layout is:

| Zenodo archive | Archive contents | Extract into |
| --- | --- | --- |
| `adams_dataset.zip` | `adams/<original filename>.tab` | Repository `data/` directory |
| `exp_dataset.zip` | `exp/<original filename>.mat` | Repository `data/` directory |

This layout places the raw files directly in `data/adams/` and `data/exp/`. If an example also occurs in the full release, keep one copy under its original filename. Remove archive-level wrapper folders before running the scripts.

The Zenodo deposit will document sources, units and channels, simulation or acquisition conditions, fault naming and dimensions, and the associated repository revision. Dataset use and citation terms will be provided with the deposit.

The root `.gitignore` excludes additional raw ADAMS files, extracted EXP files, local prediction inputs and Excel reports, and generated model outputs. The included ADAMS example, EXP setup README, and full-dataset reference figure remain tracked.
