
from pathlib import Path
import pandas as pd

# Project directories
ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "stress_model" / "data" / "raw"

def inspect_dataset():
    print("\n========== DATASET INSPECTION ==========\n")

    if not DATA_DIR.exists():
        print("Dataset directory does not exist.")
        return

    files = list(DATA_DIR.rglob("*"))

    if not files:
        print("No dataset files found yet.")
        print("Place the downloaded dataset inside stress_model/data/raw/")
        return

    for file in files:
        if not file.is_file():
            continue

        size_mb = file.stat().st_size / (1024 * 1024)

        print(f"\nFile: {file.name}")
        print(f"Size: {size_mb:.2f} MB")
        print(f"Extension: {file.suffix}")

        try:
            if file.suffix.lower() == ".csv":
                df = pd.read_csv(file, nrows=5)
                print("Columns:", list(df.columns))
                print(df.head())

            elif file.suffix.lower() == ".tsv":
                df = pd.read_csv(file, sep="\t", nrows=5)
                print("Columns:", list(df.columns))
                print(df.head())

            elif file.suffix.lower() in [".xlsx", ".xls"]:
                df = pd.read_excel(file, nrows=5)
                print("Columns:", list(df.columns))
                print(df.head())

        except Exception as error:
            print("Could not preview file:", error)

    print("\n========== INSPECTION COMPLETE ==========")


if __name__ == "__main__":
    inspect_dataset()
