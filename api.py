from pathlib import Path
import tempfile
from fastapi import FastAPI, File, UploadFile
from src.extractor import DocumentLoader, MetadataExtractor, load_training_texts

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
app = FastAPI(title="USERReady Metadata Extraction API", version="1.0.0")

extractor = MetadataExtractor(load_training_texts(DATA))
loader = DocumentLoader()

@app.get("/")
def root():
    return {"service": "USERReady Metadata Extraction", "status": "ok"}

@app.post("/extract")
async def extract(file: UploadFile = File(...)):
    suffix = Path(file.filename or "document.docx").suffix.lower()
    if suffix not in {".docx", ".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"}:
        return {"error": "Supported files: DOCX, PNG, JPG, JPEG, WEBP, TIF, TIFF"}
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        temp_path = Path(tmp.name)
    try:
        text = loader.load(temp_path)
        return {"file_name": file.filename, "metadata": extractor.extract(text)}
    finally:
        temp_path.unlink(missing_ok=True)
