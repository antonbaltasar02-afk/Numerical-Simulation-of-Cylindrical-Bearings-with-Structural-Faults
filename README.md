# Numerical Simulation of Cylindrical Bearings with Structural Faults

This repository accompanies our project on bearing fault classification using numerical simulation and experimental measurements. It contains three independent workflows: a one-dimensional convolutional neural network (CNN), a temporal convolutional network (TCN), and a feedforward neural network (FFNN).

The CNN notebook uses CWRU data. TCN and FFNN train and perform internal validation and test evaluation on ADAMS simulation data. Separate prediction scripts apply the trained models to external files, including experimental (EXP) measurements. Each model uses its own preprocessing, training, and evaluation methods.

## Repository contents

| Folder | Contents |
| --- | --- |
| [`cwru_cnn/`](cwru_cnn/) | CWRU CNN notebook, included CWRU data, and documentation |
| [`tcn/`](tcn/) | TCN training and prediction scripts, dependencies, and local model outputs |
| [`ffnn/`](ffnn/) | FFNN training and prediction scripts, ADAMS feature t-SNE visualization, and local model outputs |
| [`data/adams/`](data/adams/) | Shared ADAMS training input directory, including one example `.tab` file |
| [`data/exp/`](data/exp/) | Local EXP prediction input directory; measurements are distributed separately |
| [`predict/`](predict/) | Default external-prediction input folder and Excel prediction reports |
| [`data/README.md`](data/README.md) | Input formats and full-dataset placement and release information |

ADAMS and EXP datasets are shared between the TCN and FFNN workflows. Model checkpoints, label encoders, configurations, and training figures are saved separately under `tcn/outputs/` and `ffnn/outputs/`.

## Installation

Use Python 3.10 or newer with a compatible PyTorch installation. From the repository root, install the dependencies for either or both workflows:

```bash
python -m pip install -r tcn/requirements.txt
python -m pip install -r ffnn/requirements.txt
```

The FFNN requirements include scikit-learn 1.5 or newer for the t-SNE script's `max_iter` argument. See the [scikit-learn documentation](https://scikit-learn.org/stable/whats_new/v1.5.html).

For the CNN notebook, follow [`cwru_cnn/README.md`](cwru_cnn/README.md).

## Train TCN and FFNN on ADAMS

Place ADAMS `.tab` files directly in `data/adams/`, preserving their original names. Run either model from the repository root:

```bash
python tcn/bearing_fault_tcn_fin.py
python ffnn/bearing_fault_classifier_fin.py
```

Both scripts use ADAMS by default, with `CWRU_DATA_DIR = None`. EXP files are used through the prediction scripts. Diagnostic plots appear before training; close each plot window to continue. CUDA is selected when available and otherwise the CPU is used.

The included ADAMS example contains one compound-fault class. It demonstrates loading and preprocessing; multi-class experiments require the complete ADAMS dataset. A model trained on this example alone can predict only that one class.

## Predict external EXP files

Prediction requires each model's checkpoint, label encoder, and configuration. Generate these by training, or copy an existing saved model's three matching files into its `outputs/` folder:

| Model | Required files | Destination |
| --- | --- | --- |
| FFNN | `best_model.pt`, `label_encoder.pkl`, `config.pkl` | `ffnn/outputs/` |
| TCN | `best_tcn_model.pt`, `tcn_label_encoder.pkl`, `tcn_config.pkl` | `tcn/outputs/` |

Keep each checkpoint together with the encoder and configuration from the same trained model. Pretrained checkpoints are not included in the TCN and FFNN folders.

Place external EXP `.mat` files directly in `data/exp/`, then run the prediction scripts. EXP files are distributed separately, with a Zenodo release planned:

```bash
python ffnn/predict_rev.py data/exp
python tcn/predict_tcn_fin.py data/exp
```

The optional input argument can be a single `.tab` or `.mat` file or a folder containing those files. With no argument, the scripts read `predict/`. Input folders are searched without descending into subfolders. Relative command-line arguments are resolved from the terminal's working directory; the default directories resolve relative to the scripts.

Each script prints up to three ranked classes using the mean softmax probability across windows. Its internal result also includes a majority-vote class. Softmax values describe the model's output scores. Excel reports are saved in `predict/`: `predictions.xlsx` for FFNN and `predictions_tcn.xlsx` for TCN, including when another input directory is supplied.

The prediction preprocessing retains EXP windows without applying the training amplitude-rejection rule. FFNN uses non-overlapping 1 s windows; TCN uses 1 s windows with a 0.2 s stride. The MATLAB loaders select housing Y acceleration for FFNN and AI1 probe Y for TCN in the reference EXP file `BPFI_05_EXP.mat`. See [`data/README.md`](data/README.md) for details.

## Training methods

| Setting | TCN | FFNN |
| --- | --- | --- |
| Input representation | Filtered time-domain windows, standardized per window | Normalized Hilbert-envelope order spectra, 200 features |
| Training window / stride | 1.0 s / 0.2 s | 1.0 s / 0.25 s |
| Time-region split per file | 45% training / 10% validation / 45% test | 45% training / 10% validation / 45% test |
| Architecture | Residual causal convolutions; channels `[16, 32, 32, 64]`, kernel size 12 | Hidden layers `[128, 128, 128]` with ReLU |
| Dropout | 0.3 | 0.3 |
| Optimizer | Adam, learning rate `5e-4`, weight decay `1e-4` | Adam, learning rate `1e-3` |
| Loss | Class-weighted cross-entropy | Cross-entropy |
| Batch size / epochs / seed | 32 / 100 / 42 | 32 / 100 / 42 |
| Learning-rate schedule | ReduceLROnPlateau, factor 0.5, patience 10 | ReduceLROnPlateau, factor 0.5, patience 10 |

Training windows fit entirely within consecutive time regions of each file. A file can contribute to all three splits. Filtering is applied to the full signal before window extraction. After 100 epochs, the lowest-validation-loss checkpoint is loaded for the final ADAMS test evaluation.

Training prints test metrics and a classification report and saves checkpoints, encoders, configurations, training curves, and confusion matrices. Subsequent runs reuse the same output filenames. The model READMEs list the files in detail.

## ADAMS feature visualization

```bash
python ffnn/visualize_tsne.py
```

This script visualizes the FFNN's envelope order-spectrum input features from ADAMS, combining retained training, validation, and test windows. It does not require a trained checkpoint. The generated figure is saved as `ffnn/outputs/tsne_adams_full.png`.

[`ffnn/reference_results/tsne_adams_full.png`](ffnn/reference_results/tsne_adams_full.png) provides a full-dataset feature visualization. Running with only the included example uses its single class and does not reproduce the full-dataset figure.

## Full datasets

The repository includes one ADAMS example. EXP measurement files are distributed separately; the `data/exp/` folder contains setup instructions. The full ADAMS and experimental datasets are planned for a separate Zenodo deposit. The dataset DOI and download link will be listed in [`data/README.md`](data/README.md) when the deposit is published; the DOI is currently pending.

## Citation

Please cite the associated project report when using this work. For experiments using the complete ADAMS or experimental datasets, also cite the dataset's Zenodo record once published.

```bibtex
@techreport{2026bearings,
  author = {Berntsson, Anton
            and Finsen, {\'O}lafur Ingi
            and Sivasankar, Naveen Kumar
            and Nanal, Ranjit
            and {\c{C}}avu{\c{s}}o{\u{g}}lu, Teoman Valentin},
  title = {Numerical Simulation of Cylindrical Bearings
           with Structural Faults},
  institution = {Chalmers University of Technology},
  type = {Course project report},
  note = {TME131 -- Project in Applied Mechanics},
  address = {Gothenburg, Sweden},
  year = {2026},
  month = oct
}
```
