from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
PRED = ROOT / "output" / "predictions.csv"
REPORT = ROOT / "reports" / "evaluation_report.json"
FIELDS = [
    "Aggrement Value",
    "Aggrement Start Date",
    "Aggrement End Date",
    "Renewal Notice (Days)",
    "Party One",
    "Party Two",
]

def norm(x):
    if pd.isna(x):
        return ""
    return " ".join(str(x).strip().split()).lower()

if __name__ == "__main__":
    truth = pd.read_csv(DATA / "test.csv", dtype=str).fillna("")
    pred = pd.read_csv(PRED, dtype=str).fillna("")
    merged = truth.merge(pred, on="File Name", how="left", suffixes=("_true", "_pred"))
    scores = {}
    for field in FIELDS:
        matches = merged.apply(lambda r: norm(r[f"{field}_true"]) == norm(r[f"{field}_pred"]), axis=1)
        scores[field] = {
            "true": int(matches.sum()),
            "false": int((~matches).sum()),
            "recall": float(matches.mean()),
        }
    overall = sum(x["true"] for x in scores.values()) / (len(FIELDS) * len(merged))
    report = {"per_field": scores, "micro_field_recall": overall}
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
