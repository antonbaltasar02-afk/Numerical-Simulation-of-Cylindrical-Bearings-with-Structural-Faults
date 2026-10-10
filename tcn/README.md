# Temporal convolutional network (TCN)

`bearing_fault_tcn_fin.py` provides the ADAMS training, validation, and internal test workflow. `predict_tcn_fin.py` applies the saved model to external `.tab` or `.mat` files, including EXP measurements.

## Install and train

From the repository root:

```bash
python -m pip install -r tcn/requirements.txt
python tcn/bearing_fault_tcn_fin.py
```

Place ADAMS training files directly in `data/adams/`. The default configuration uses ADAMS and leaves `CWRU_DATA_DIR = None`; EXP is processed by the separate prediction script. Training labels come from `TYPE_RULES`, `SIZE_RULES`, and `MANUAL_LABEL_MAP` in this module.

Close the diagnostic plot window to continue training. CUDA is selected when available, otherwise the CPU is used. Default paths resolve relative to the scripts, including when they are launched from another working directory.

The included ADAMS example represents one class. Training a multi-class model requires the full ADAMS dataset.

## Use an existing saved model

Copy `best_tcn_model.pt`, `tcn_label_encoder.pkl`, and `tcn_config.pkl` from the same saved training run into `tcn/outputs/`. In the original local layout these files are in `model_tcn/`. Alternatively, the training command creates them in `outputs/`.

No pretrained weights are distributed in this model folder. Keep checkpoint, encoder, and configuration together so the network dimensions and class order match.

## Predict EXP measurements

Place externally obtained EXP `.mat` files directly in `data/exp/`. No raw EXP measurements are included in the repository; a separate Zenodo release is planned. From the repository root:

```bash
python tcn/predict_tcn_fin.py data/exp
```

A single file or a flat input folder can be supplied as the command-line argument. With no argument, the input is `predict/`. Relative argument paths follow the terminal's working directory.

Prediction generates 1 s windows with a 0.2 s stride, standardizes each window, and evaluates them in batches of 64. For the reference EXP file `BPFI_05_EXP.mat`, the imported MATLAB loader selects `Data1_AI_1_AI_1_minus_Probe_Y`.

Inference retains windows without applying the training amplitude-rejection rule. The loader multiplies the selected MATLAB signal by `9.81 * 1000` and reads sampling rate from metadata.

The per-file result contains a majority-vote class and mean class probabilities. The printed top-three table and Excel report rank classes by mean probability, which can differ from majority-vote order. The report is always written to `predict/predictions_tcn.xlsx`, including when another input directory is supplied. Later runs overwrite that report.

## Training method

The full signal is mean-centered and filtered using an eighth-order Butterworth low-pass filter, with the cutoff limited relative to the sampling rate. Consecutive 45% / 10% / 45% time regions supply training, validation, and test windows. Training uses 1 s windows with a 0.2 s stride and discards windows containing a filtered value greater than 10,000.

The TCN takes filtered time-domain windows standardized separately. Four residual blocks use causal, dilated, weight-normalized convolutions with channels `[16, 32, 32, 64]`, kernel size 12, and dropout 0.3. Global average pooling and a linear layer produce class scores.

Training uses class-weighted cross-entropy, Adam with learning rate `5e-4` and weight decay `1e-4`, and batch size 32.

Both the random seed and NumPy seed are 42. The ReduceLROnPlateau scheduler uses factor 0.5 and patience 10. Training runs for 100 epochs, then loads the best validation-loss checkpoint for the final ADAMS test evaluation.

## Outputs

`outputs/` contains `best_tcn_model.pt`, `tcn_label_encoder.pkl`, `tcn_config.pkl`, and `tcn_training_curves.png`, `tcn_confusion_matrix.png`, `tcn_confusion_matrix_by_type.png`, and applicable `tcn_confusion_matrix_zoomed_<type>.png` files. Test metrics and a classification report are printed in the terminal. Subsequent training runs reuse these filenames.

See [`../data/README.md`](../data/README.md) for file formats and full-dataset placement.
