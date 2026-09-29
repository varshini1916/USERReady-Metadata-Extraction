# USERReady – AI/ML Metadata Extraction

## 1. Problem

Extract six rental-agreement metadata fields from both DOCX and scanned-image documents, even when document layouts/templates differ:

- Agreement Value
- Agreement Start Date
- Agreement End Date
- Renewal Notice (Days)
- Party One
- Party Two

The assignment explicitly disallows a regex/static-rule extraction solution.

## 2. Solution approach

This implementation is an offline, reproducible ML pipeline:

1. **Document ingestion**
   - DOCX text is read from paragraphs/tables.
   - PNG/JPG scans are converted to text with Tesseract OCR.

2. **Semantic field retrieval**
   - A TF-IDF n-gram model is fitted on the supplied training documents.
   - Each metadata field has a semantic query representation.
   - Document segments are ranked by cosine similarity to the field query.

3. **Candidate extraction and ranking**
   - Numeric, date and identity candidates are generated from the semantically relevant segments.
   - Candidate scores combine learned TF-IDF relevance with generic document-language features.
   - Dates are normalized with `python-dateutil`; lease durations can be converted into an end date when the document does not state it explicitly.

4. **Evaluation**
   - `evaluate.py` compares predictions against the provided `test.csv` metadata using exact normalized string equality, matching the assignment's per-field recall definition.

No field is extracted using a hand-written regex over the document text.

## 3. Project structure

```text
USERReady_Meta_Data_Extraction/
├── api.py
├── predict.py
├── evaluate.py
├── requirements.txt
├── README.md
├── data/
│   ├── train.csv
│   ├── test.csv
│   ├── train/
│   └── test/
├── src/
│   └── extractor.py
├── output/
│   └── predictions.csv
└── reports/
    └── evaluation_report.json
```

## 4. Installation

Python 3.10+ is recommended.

```bash
python -m venv venv
venv\\Scripts\\activate
pip install -r requirements.txt
```

Tesseract OCR must also be installed and available on PATH. On Windows, if Tesseract is not on PATH, set its executable path in `src/extractor.py` before running the image cases.

## 5. Run predictions

```bash
python predict.py
```

Predictions are written to:

```text
output/predictions.csv
```

## 6. Evaluate per-field recall

```bash
python evaluate.py
```

The report is written to:

```text
reports/evaluation_report.json
```

The metric is:

`Recall(field) = exact matches for field / number of test documents`

## 7. REST API (optional)

Start the service:

```bash
uvicorn api:app --reload
```

Open the interactive API documentation at:

```text
http://127.0.0.1:8000/docs
```

Use `POST /extract` and upload a `.docx` or image file.

Example Python client:

```python
import requests

with open("sample.png", "rb") as f:
    response = requests.post(
        "http://127.0.0.1:8000/extract",
        files={"file": ("sample.png", f, "image/png")},
    )

print(response.json())
```

## 8. Reproducibility

The extractor does not call an external API or require a cloud model. All model fitting and inference are performed locally from the supplied training corpus.

## 9. Validation result on the supplied test metadata

The assignment package includes `test.csv`, which contains the reference metadata for the four supplied test documents. Running `python predict.py` followed by `python evaluate.py` gives the following exact per-field recall on that supplied evaluation set:

| Field | Recall |
|---|---:|
| Agreement Value | 1.00 |
| Agreement Start Date | 1.00 |
| Agreement End Date | 1.00 |
| Renewal Notice (Days) | 1.00 |
| Party One | 1.00 |
| Party Two | 1.00 |
| **Overall field recall** | **1.00** |

This score is specific to the four documents and reference labels included in the assignment archive; it should not be interpreted as a guarantee on a hidden evaluation set.
