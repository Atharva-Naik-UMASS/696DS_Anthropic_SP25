import argparse
import re
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import yaml

def main():
    parser = argparse.ArgumentParser(description="Generate confusion matrix from a CSV file")
    parser.add_argument("file_name", type=str, help="Path to the scored results CSV file")
    parser.add_argument("yaml_path", type=str, help="Path to the dataset config YAML file")
    args = parser.parse_args()

    file_name = args.file_name
    yaml_path = args.yaml_path

    df = pd.read_csv(file_name) 

    match = re.search(r"scored_results_([^_]+)_(.+)\.csv", file_name)

    adapter = "Unknown"
    dataset_name = "Unknown"
    if match:
        adapter = match.group(1)
        dataset_name = match.group(2)
        print("Extracted domains:", adapter, dataset_name)
    else:
        print("Could not extract domains.")

    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
        
    groundtruth_column = config.get("groundtruth_column", "label")
    prediction_column = "generated"
    y_pred = df[prediction_column]
    y_true = df[groundtruth_column]

    all_labels = sorted(set(y_true).union(set(y_pred)))

    cm = confusion_matrix(y_true, y_pred, labels=all_labels)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=all_labels)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax, cmap='Blues', xticks_rotation=45)

    plt.title(f"Confusion Matrix — Adapter: {adapter} | Dataset: {dataset_name}")
    output_filename = f"/project/pi_wenlongzhao_umass_edu/6/outputs/results_3b/confusion_matrices/confusion_matrix_{adapter}_{dataset_name}.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_filename}")

if __name__ == "__main__":
    main()
