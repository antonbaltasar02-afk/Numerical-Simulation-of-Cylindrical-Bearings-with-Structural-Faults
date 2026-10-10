import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import pandas as pd
from pathlib import Path

# Import the parser from the updated classifier script
from bearing_fault_classifier_fin import (
    parse_compound_label,
    load_tab_file,
    process_and_split_file,
    ADAMS_FS,
    ADAMS_SHAFT_FREQ_HZ
)

SOURCE_DIR = str(Path(__file__).resolve().parents[1] / "data" / "adams")


def get_color_and_legend_label(label: str):
    """
    Returns a hex color and a display legend label based on the fault class,
    size, and orientation.
    """
    l = label.lower()

    # Helper size mapping for gradients
    # 005 -> 1, 01 -> 2, 03 -> 3, 05 -> 4
    size_to_weight = {'005': 1, '01': 2, '03': 3, '05': 4}

    # 1. HEALTHY
    if "healthy" in l:
        return "forestgreen", "Healthy"

    # 2. SPALLING
    if "spalling" in l:
        if "outer" in l:
            return "black", "Spalling (Outer)"
        elif "inner" in l:
            return "dimgrey", "Spalling (Inner)"
        return "grey", "Spalling"

    # 3. ORIENTATION
    # Override standard coloring if orientation is present
    if "_45deg" in l:
        return "magenta", "Fault (45 deg)"
    if "_225deg" in l:
        return "cyan", "Fault (22.5 deg)"

    # 4. COMPOUND FAULTS
    # Detect if both IR and OR exist (using the label format from classifier)
    if "ir_" in l and "or_" in l:
        # Extract sizes: label format is "IR_xx_OR_yy"
        # We need to split by underscore
        parts = l.split('_')
        ir_idx = parts.index('ir') + 1 if 'ir' in parts else -1
        or_idx = parts.index('or') + 1 if 'or' in parts else -1

        ir_size = parts[ir_idx] if ir_idx != -1 else ""
        or_size = parts[or_idx] if or_idx != -1 else ""

        # Construct Legend Name: Compound_{IR}_{OR}
        # User requested IR fault first, OR second.
        legend_name = f"Compound_{ir_size.upper()}_{or_size.upper()}"

        # Construct Color: Yellow to Orange gradient based on combined size
        w_ir = size_to_weight.get(ir_size, 2)
        w_or = size_to_weight.get(or_size, 2)
        total_severity = w_ir + w_or  # Range 2 (005+005) to 8 (05+05)

        # Map severity to colors
        compound_palette = {
            2: "#FFFFE0",  # Light Yellow
            3: "#FFFACD",  # Lemon Chiffon
            4: "#FFE4B5",  # Moccasin
            5: "#FFD700",  # Gold
            6: "#FFA500",  # Orange
            7: "#FF8C00",  # Dark Orange
            8: "#FF4500",  # Orange Red
        }
        color = compound_palette.get(total_severity, "#FFD700")
        return color, legend_name

    # 5. ROLLER (Default handling)
    if "roller" in l:
        return "brown", "Roller Fault"

    # 6. STANDARD IRF (Inner Race Fault) - Blue Shades
    if "inner" in l or ("ir_" in l):
        base_name = "IR Fault"
        if "005" in l: return "#ADD8E6", f"{base_name} (0.05)"  # Light Blue
        if "01" in l:  return "#87CEEB", f"{base_name} (0.1)"  # Sky Blue
        if "03" in l:  return "#4682B4", f"{base_name} (0.3)"  # Steel Blue
        if "05" in l:  return "#00008B", f"{base_name} (0.5)"  # Dark Blue
        return "steelblue", base_name

    # 7. STANDARD ORF (Outer Race Fault) - Red Shades
    if "outer" in l or ("or_" in l):
        base_name = "OR Fault"
        if "005" in l: return "#FFAAAA", f"{base_name} (0.05)"  # Light Red
        if "01" in l:  return "#FF5555", f"{base_name} (0.1)"  # Medium Red
        if "03" in l:  return "#CC0000", f"{base_name} (0.3)"  # Crimson
        if "05" in l:  return "#8B0000", f"{base_name} (0.5)"  # Dark Red
        return "red", base_name

    # Fallback
    return "grey", "Unknown"


def extract_features_from_adams(folder_path):
    if not os.path.exists(folder_path):
        print(f"[Error] Folder not found: {folder_path}")
        return np.array([]), []

    print(f"Processing Adams data from: {folder_path}")
    all_features = []
    all_labels = []
    files = sorted(Path(folder_path).glob("*.tab"))

    for fpath in files:
        # Use the updated parser from the classifier script
        label = parse_compound_label(fpath.stem)

        if label is None:
            continue  # Skip unknowns

        try:
            # Load Signal
            signal = load_tab_file(str(fpath))
            fs = ADAMS_FS
            shaft_freq = ADAMS_SHAFT_FREQ_HZ

            # Extract Features (Train + Val + Test splits)
            tr, va, te = process_and_split_file(signal, fs, shaft_freq, run_diagnostics=False)
            segments = tr + va + te

            if not segments:
                continue

            all_features.extend(segments)
            all_labels.extend([label] * len(segments))

        except Exception as e:
            print(f"  Error processing {fpath.name}: {e}")

    print(f"  -> Extracted {len(all_features)} samples.")
    return np.array(all_features), all_labels


def main():
    # Extract Data
    print("Extracting features for t-SNE visualization.\n")
    X, y = extract_features_from_adams(SOURCE_DIR)

    if len(X) == 0:
        print("[ERROR] No data found.")
        return

    # Run t-SNE
    print("Running t-SNE.")
    tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, max_iter=1000,
                random_state=42, init='pca')
    tsne_results = tsne.fit_transform(X)

    # Create DataFrame for plotting
    df = pd.DataFrame({"Component 1": tsne_results[:, 0],
                       "Component 2": tsne_results[:, 1],
                       "Class": y})

    # Map classes to colors and display labels
    # We iterate over unique classes to define the palette and mapping
    unique_labels = df["Class"].unique()
    palette = {}
    label_name_map = {}

    for lbl in unique_labels:
        color, disp_name = get_color_and_legend_label(lbl)
        palette[lbl] = color
        label_name_map[lbl] = disp_name

    # Plotting
    plt.figure(figsize=(12, 9))

    # Use the custom palette. Hue uses the raw label to group correctly.
    sns.scatterplot(data=df, x="Component 1", y="Component 2", hue="Class",
                    palette=palette, alpha=0.7, s=20, linewidth=0)

    # Customize Legend
    handles, labels = plt.gca().get_legend_handles_labels()
    # Sort legend labels for better readability if needed, or keep as is
    # Here we update the text in the legend using our map
    new_labels = [label_name_map.get(l, l) for l in labels]

    plt.legend(handles=handles, labels=new_labels,
               title="Fault Class", bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.title("Source Domain (Adams Simulation) Feature Distribution", fontsize=14)
    plt.xlabel("First component", fontsize=12)
    plt.ylabel("Second component", fontsize=12)
    plt.tight_layout()

    save_path = str(Path(__file__).resolve().parent / "outputs" / "tsne_adams_full.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n[Saved] Plot saved to '{save_path}'")
    plt.show()


if __name__ == "__main__":
    main()