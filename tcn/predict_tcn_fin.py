"""
Bearing Fault Classifier — TCN Inference Script (Top-3 Table)
==============================================================
Use to classify new files with the trained TCN model.
Supports both Adams .tab files and CWRU .mat files.
Outputs a ranked table of the top-3 predicted classes per file.
"""

import sys
import os
import torch
import numpy as np
import joblib
from pathlib import Path
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter
from bearing_fault_tcn_fin import (
    load_tab_file, load_mat_file,
    detrend_signal, apply_lowpass_filter, TCN,
    ADAMS_FS,
    WINDOW_SECONDS, STRIDE_SECONDS,
)

# Directories
PREDICT_SOURCE = str(Path(__file__).resolve().parents[1] / "predict")
MODEL_DIR = str(Path(__file__).resolve().parent / "outputs")

def process_file_for_inference(signal: np.ndarray, fs: float):
    """
    Applies the full preprocessing pipeline (detrend, filter, window, normalize)
    to generate input segments for the TCN model.
    """
    # Detrend and Filter
    signal_detrended = detrend_signal(signal)
    signal_filtered  = apply_lowpass_filter(signal_detrended, fs)

    # Windowing
    window_size = int(WINDOW_SECONDS * fs)
    stride      = int(STRIDE_SECONDS * fs)
    N           = len(signal_filtered)

    if N < window_size:
        return []

    segments = []
    for start in range(0, N - window_size + 1, stride):
        end = start + window_size
        seg = signal_filtered[start:end].copy()

        # Normalize per segment (Z-score)
        std = np.std(seg)
        if std > 0:
            seg = (seg - np.mean(seg)) / std
        else:
            seg = seg - np.mean(seg)

        segments.append(seg.astype(np.float32))

    return segments

def load_trained_model(model_dir: str, device: torch.device):
    """
    Load the trained TCN model weights and configuration.
    """
    config_path = os.path.join(model_dir, "tcn_config.pkl")
    encoder_path = os.path.join(model_dir, "tcn_label_encoder.pkl")
    weights_path = os.path.join(model_dir, "best_tcn_model.pt")

    for p in [config_path, encoder_path, weights_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"Required model file not found: '{p}'\n"
                f"Make sure you have run bearing_fault_tcn_alt.py first."
            )

    config        = joblib.load(config_path)
    label_encoder = joblib.load(encoder_path)

    # Initialize TCN using the imported class and the loaded config
    model = TCN(
        num_classes   = config["num_classes"],
        tcn_channels  = config["tcn_channels"],
        kernel_size   = config["kernel_size"],
        dropout       = config["dropout_rate"]
    )

    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.to(device).eval()
    return model, label_encoder, config

def predict_file(filepath: str, model, label_encoder, config: dict, device: torch.device) -> dict:
    """
    Classify a single .tab or .mat file using majority vote across windows.
    """
    ext = Path(filepath).suffix.lower()

    if ext == ".tab":
        # Use imported load_tab_file
        signal = load_tab_file(filepath)
        fs = ADAMS_FS

    elif ext == ".mat":
        # Use imported load_mat_file
        # Updated to handle the new return signature (signal, shaft_freq, fs)
        signal, _, fs = load_mat_file(filepath)

    else:
        return {
            "file":  os.path.basename(filepath),
            "error": f"Unsupported file type '{ext}'. Expected .tab or .mat"
        }

    # Process signal into windows
    features = process_file_for_inference(signal, fs)

    if not features:
        return {
            "file":  os.path.basename(filepath),
            "error": "Signal too short to produce any segments"
        }

    # Run windows through the model in mini-batches to avoid OOM on large files
    BATCH_SIZE = 64
    all_probs  = []

    with torch.no_grad():
        for i in range(0, len(features), BATCH_SIZE):
            batch       = features[i : i + BATCH_SIZE]
            X           = (torch.tensor(np.stack(batch), dtype=torch.float32)
                           .unsqueeze(1)
                           .to(device))
            logits      = model(X)
            probs_batch = torch.softmax(logits, dim=1).cpu().numpy()
            all_probs.append(probs_batch)

    probs = np.vstack(all_probs)  # (n_segments, n_classes)

    # Majority vote
    segment_preds = np.argmax(probs, axis=1)
    vote_counts   = np.bincount(segment_preds,
                                minlength=len(label_encoder.classes_))
    winner_idx    = int(np.argmax(vote_counts))
    mean_probs    = probs.mean(axis=0)

    class_names = list(label_encoder.classes_)

    return {
        "file":            os.path.basename(filepath),
        "predicted_class": class_names[winner_idx],
        "confidence":      float(mean_probs[winner_idx]),
        "n_segments":      len(features),
        "vote_counts":     {class_names[i]: int(vote_counts[i])
                            for i in range(len(class_names))},
        "mean_probs":      {class_names[i]: float(mean_probs[i])
                            for i in range(len(class_names))},
    }

def collect_files(target: str) -> list:
    """Return list of .tab and .mat files from a file path or directory."""
    p = Path(target)
    if not p.exists():
        print(f"[ERROR] The path '{p}' does not exist.")
        sys.exit(1)

    if p.is_dir():
        files = sorted(p.glob("*.tab")) + sorted(p.glob("*.mat"))
    elif p.suffix.lower() in (".tab", ".mat"):
        files = [p]
    else:
        print(f"[ERROR] '{target}' is not a .tab/.mat file or directory.")
        sys.exit(1)
    return files

def export_to_excel(rows: list, save_path: str, top_n: int = 3) -> None:
    """
    Write the top-N prediction results to a formatted .xlsx file.

    Each row in `rows` is a dict with keys:
        'file'   – filename string
        'top_n'  – list of (class_name, probability) tuples, length == top_n
        'error'  – optional error string (replaces top_n entries)
    """
    wb = Workbook()
    ws = wb.active
    ws.title = "Predictions"

    # ── Styles ────────────────────────────────────────────────────────────────
    HEADER_FILL = PatternFill("solid", start_color="1F4E79")
    HEADER_FONT = Font(name="Arial", bold=True, color="FFFFFF", size=11)
    RANK_FILLS  = [
        PatternFill("solid", start_color="D6E4F0"),
        PatternFill("solid", start_color="EBF5FB"),
        PatternFill("solid", start_color="F4F9FC"),
    ]
    FILE_FONT   = Font(name="Arial", size=10, bold=True)
    CELL_FONT   = Font(name="Arial", size=10)
    ERROR_FONT  = Font(name="Arial", size=10, color="C00000", italic=True)
    CENTER      = Alignment(horizontal="center", vertical="center")
    LEFT        = Alignment(horizontal="left",   vertical="center")
    THIN        = Side(style="thin", color="BFBFBF")
    BORDER      = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)
    PCT_FMT     = "0.0%"

    # ── Header row ────────────────────────────────────────────────────────────
    header_row = ["File"]
    for i in range(1, top_n + 1):
        header_row += [f"#{i} Predicted Class", f"#{i} Confidence"]

    ws.append(header_row)
    for col_idx, _ in enumerate(header_row, start=1):
        cell = ws.cell(row=1, column=col_idx)
        cell.font      = HEADER_FONT
        cell.fill      = HEADER_FILL
        cell.alignment = CENTER
        cell.border    = BORDER

    # ── Data rows ─────────────────────────────────────────────────────────────
    for data_row in rows:
        row_num = ws.max_row + 1

        file_cell = ws.cell(row=row_num, column=1, value=data_row["file"])
        file_cell.font      = FILE_FONT
        file_cell.alignment = LEFT
        file_cell.border    = BORDER

        if "error" in data_row:
            err_cell = ws.cell(row=row_num, column=2,
                               value=f"ERROR: {data_row['error']}")
            err_cell.font      = ERROR_FONT
            err_cell.alignment = LEFT
            err_cell.border    = BORDER
            ws.merge_cells(start_row=row_num, start_column=2,
                           end_row=row_num,   end_column=len(header_row))
        else:
            for rank, (cls_name, prob) in enumerate(data_row["top_n"]):
                col_class = 2 + rank * 2
                col_conf  = col_class + 1
                fill      = RANK_FILLS[rank] if rank < len(RANK_FILLS) else RANK_FILLS[-1]

                cls_cell = ws.cell(row=row_num, column=col_class, value=cls_name)
                cls_cell.font      = CELL_FONT
                cls_cell.fill      = fill
                cls_cell.alignment = LEFT
                cls_cell.border    = BORDER

                conf_cell = ws.cell(row=row_num, column=col_conf, value=prob)
                conf_cell.font          = CELL_FONT
                conf_cell.fill          = fill
                conf_cell.alignment     = CENTER
                conf_cell.number_format = PCT_FMT
                conf_cell.border        = BORDER

    # ── Column widths ─────────────────────────────────────────────────────────
    ws.column_dimensions["A"].width = 46
    for i in range(top_n):
        ws.column_dimensions[get_column_letter(2 + i * 2)].width = 26
        ws.column_dimensions[get_column_letter(3 + i * 2)].width = 13

    ws.freeze_panes = "A2"
    wb.save(save_path)


def main():
    if len(sys.argv) >= 2:
        target = sys.argv[1]
    else:
        target = PREDICT_SOURCE
        print(f"[INFO] No input argument provided. Using default path: {target}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[Device] {device}")
    print(f"[Model]  Loading TCN from '{MODEL_DIR}'...\n")

    model, label_encoder, config = load_trained_model(MODEL_DIR, device)
    class_names = list(label_encoder.classes_)
    print(f"[Model]  Classes: {class_names}\n")

    files = collect_files(target)

    if not files:
        print(f"[ERROR] No .tab or .mat files found at '{target}'")
        sys.exit(1)

    # --- TOP-3 PREDICTIONS TABLE ---
    TOP_N     = 3
    col_file  = 45
    col_class = 24
    col_conf  = 10

    rank_labels = [f"#{i+1} Class" for i in range(TOP_N)]
    conf_labels = [f"#{i+1} Conf." for i in range(TOP_N)]

    header = f"{'File':<{col_file}}"
    for i in range(TOP_N):
        header += f"  {rank_labels[i]:<{col_class}} {conf_labels[i]:>{col_conf}}"
    divider = "─" * len(header)

    print(f"Classifying {len(files)} file(s)...\n")
    print(header)
    print(divider)

    export_rows = []

    for fpath in files:
        result = predict_file(str(fpath), model, label_encoder, config, device)

        if "error" in result:
            print(f"{result['file']:<{col_file}}  ERROR: {result['error']}")
            export_rows.append({"file": result["file"], "error": result["error"]})
            continue

        # Rank all classes by mean probability, descending
        ranked = sorted(result["mean_probs"].items(), key=lambda x: x[1], reverse=True)
        top_n  = ranked[:TOP_N]

        while len(top_n) < TOP_N:
            top_n.append(("—", 0.0))

        row = f"{result['file']:<{col_file}}"
        for cls_name, prob in top_n:
            row += f"  {cls_name:<{col_class}} {prob:>{col_conf}.1%}"
        print(row)

        export_rows.append({"file": result["file"], "top_n": top_n})

    # ── Excel export ──────────────────────────────────────────────────────────
    if export_rows:
        xlsx_path = os.path.join(PREDICT_SOURCE, "predictions_tcn.xlsx")
        export_to_excel(export_rows, xlsx_path, top_n=TOP_N)
        print(f"\n[Export] Results saved to '{xlsx_path}'")

if __name__ == "__main__":
    main()