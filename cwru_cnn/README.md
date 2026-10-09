# CWRU bearing fault classification — 1D CNN

One Jupyter notebook containing the original CWRU CNN workflow, with CWRU data files included.

## Run

Use Python 3.10 or newer. Extract the complete folder and open a terminal **inside `cwru_cnn`**, then run:

```bash
python -m pip install -r requirements.txt
python -m jupyterlab CWRU_CNN.ipynb
```

In Jupyter, restart the kernel and run all cells from top to bottom. No separate data download or path editing is needed.
The code automatically selects CUDA when available, otherwise CPU. The default executes **10 runs**, each with up to **1,000 epochs**, with early stopping after **20 epochs without a validation-loss improvement**.

## Folder contents

| Path | Purpose |
| --- | --- |
| `CWRU_CNN.ipynb` | Complete CNN code and explanatory notebook sections |
| `data/CWRU/<class>/<class>.mat` | Ten original MAT files, one class folder per file |
| `requirements.txt` | Notebook and original script dependencies |
| `README.md` | Setup and usage |
| `.gitignore` | Excludes generated outputs and local notebook/environment files |
| `outputs/` | Created automatically when running; contains models and figures |

The ten class names, in the original loader's alphabetical order, are:
`Ball_007`, `Ball_014`, `Ball_021`, `Healthy`, `InnerR_007`, `InnerR_014`, `InnerR_021`, `OuterR_007`, `OuterR_014`, `OuterR_021`.

## Original behavior and outputs

The defaults are window size 5,000, stride 2,500, block size 10,000, requested train/validation/test ratios 0.45/0.10/0.45, batch size 32, Adam with learning rate `1e-5`, cross-entropy loss, and dropout 0.3. Split sizes use the original whole-block rounding.
`RANDOM_SEED = None` is retained; block assignments and model training can vary between runs.

- Each run prints test metrics and a classification report.
- The final cell prints accuracy/epoch summaries and displays the mean confusion matrix across runs.
- The checkpoint is saved as `outputs/models/best_bearing_1dcnn_k100_k50_CWRU_split.pt`. Each run reuses this filename, as in the source script.
- The aggregate matrix is saved as `outputs/figures/CM_CWRU_split_10_runs_nob.png` with the default run count.
- Training/validation curve figures retain the original condition `400 < epoc_hist[-1] < 1000`. The per-run confusion matrix and t-SNE blocks remain commented.
- Original epoch summaries use zero-based epoch indices.

## Data source

[Case Western Reserve University Bearing Data Center](https://engineering.case.edu/bearingdatacenter).
The included subset consists of the ten supplied CWRU MAT files. The original loader selects the first non-metadata key containing `DE` from each file.

## Citation

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
