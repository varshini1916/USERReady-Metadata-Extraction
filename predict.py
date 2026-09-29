from pathlib import Path
from src.extractor import predict_directory



ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUTPUT = ROOT / "output" / "predictions.csv"

if __name__ == "__main__":
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df = predict_directory(DATA / "test", DATA)
    df.to_csv(OUTPUT, index=False)
    print(df.to_string(index=False))
    print(f"\nSaved: {OUTPUT}")
