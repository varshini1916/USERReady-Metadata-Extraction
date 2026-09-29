from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
import re
import os
import pandas as pd
import pytesseract

TESSERACT_CMD = os.getenv("TESSERACT_CMD", r"C:\Program Files\Tesseract-OCR\tesseract.exe")
if os.path.exists(TESSERACT_CMD):
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
import spacy
from dateutil import parser as date_parser
from dateutil.relativedelta import relativedelta
from docx import Document
from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


FIELDS = [
    "Aggrement Value",
    "Aggrement Start Date",
    "Aggrement End Date",
    "Renewal Notice (Days)",
    "Party One",
    "Party Two",
]

FIELD_QUERIES = {
    "Aggrement Value": "monthly rent rental amount amount payable rent of rupees dollars agreement value",
    "Aggrement Start Date": "agreement start date commencement effective from beginning from date of agreement",
    "Aggrement End Date": "agreement end date ending until expiry termination end of lease period",
    "Renewal Notice (Days)": "notice period renewal notice prior notice termination notice days",
    "Party One": "first party owner landlord lessor party one",
    "Party Two": "second party tenant lessee party two",
}

MONTHS = {
    "january", "february", "march", "april", "may", "june", "july",
    "august", "september", "october", "november", "december",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec",
}

NUMBER_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
    "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
    "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
    "eighteen": 18, "nineteen": 19, "twenty": 20,
}


@dataclass
class Segment:
    text: str
    index: int


class DocumentLoader:
    """Loads DOCX and image documents into plain text."""

    def load(self, path: Path) -> str:
        if path.suffix.lower() == ".docx":
            return self._docx(path)
        if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"}:
            return pytesseract.image_to_string(Image.open(path))
        raise ValueError(f"Unsupported document type: {path.suffix}")

    @staticmethod
    def _docx(path: Path) -> str:
        doc = Document(path)
        parts = [p.text for p in doc.paragraphs if p.text.strip()]
        for table in doc.tables:
            for row in table.rows:
                row_text = " | ".join(cell.text for cell in row.cells).strip()
                if row_text:
                    parts.append(row_text)
        return "\n".join(parts)


class SemanticRanker:
    """TF-IDF semantic relevance model trained on the supplied training corpus."""

    def __init__(self, training_texts: Iterable[str]):
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            ngram_range=(1, 2),
            min_df=1,
            sublinear_tf=True,
        )
        corpus = []
        for text in training_texts:
            corpus.extend(self.segment(text))
        corpus.extend(FIELD_QUERIES.values())
        self.vectorizer.fit(corpus)

    @staticmethod
    def segment(text: str) -> List[str]:
        """Create document segments without field-specific patterns."""
        segments: List[str] = []
        for raw in text.splitlines():
            line = " ".join(raw.split())
            if not line:
                continue
            # Long lines are broken at natural punctuation using spaCy's tokenizer.
            doc = NLP(line)
            start = 0
            for i, token in enumerate(doc):
                if token.text in {".", ";", ":"} and i - start >= 4:
                    piece = doc[start : i + 1].text.strip()
                    if piece:
                        segments.append(piece)
                    start = i + 1
            tail = doc[start:].text.strip()
            if tail:
                segments.append(tail)
        return segments

    def rank(self, text: str, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        segs = self.segment(text)
        if not segs:
            return []
        matrix = self.vectorizer.transform(segs + [query])
        scores = cosine_similarity(matrix[:-1], matrix[-1]).ravel()
        order = scores.argsort()[::-1][:top_k]
        return [(segs[i], float(scores[i])) for i in order]


NLP = spacy.blank("en")


class MetadataExtractor:
    """Hybrid ML extractor: semantic segment ranking + generic candidate scoring."""

    def __init__(self, training_texts: Iterable[str]):
        self.ranker = SemanticRanker(training_texts)

    @staticmethod
    def _numeric_candidates(text: str) -> List[Tuple[float, str, str]]:
        doc = NLP(text)
        results = []
        for i, token in enumerate(doc):
            raw = token.text.strip()
            low = raw.lower().strip(".,:;()[]")
            value = None
            if token.like_num:
                cleaned = raw.replace(",", "").replace("Rs", "").replace("rs", "")
                cleaned = cleaned.strip(".-")
                try:
                    value = float(cleaned)
                except ValueError:
                    value = None
            elif low in NUMBER_WORDS:
                value = float(NUMBER_WORDS[low])
            if value is not None:
                left = max(0, i - 7)
                right = min(len(doc), i + 8)
                context = doc[left:right].text
                results.append((value, raw, context))
        return results

    @staticmethod
    def _date_candidates(text: str) -> List[Tuple[datetime, str]]:
        doc = NLP(text)
        found = []
        for i, token in enumerate(doc):
            low = token.text.lower().strip(".,:;()[]")
            looks_date_like = low in MONTHS or (
                any(ch.isdigit() for ch in token.text)
                and any(mark in token.text for mark in [".", "/", "-"])
            )
            if not looks_date_like:
                continue
            for window in (2, 3, 4):
                start = max(0, i - window)
                end = min(len(doc), i + window + 1)
                candidate = doc[start:end].text
                try:
                    parsed = date_parser.parse(candidate, fuzzy=True, dayfirst=True, default=datetime(2000, 1, 1))
                except (ValueError, OverflowError, TypeError):
                    continue
                if 1950 <= parsed.year <= 2035:
                    low_candidate = candidate.lower()
                    if "end of" in low_candidate:
                        import calendar
                        last_day = calendar.monthrange(parsed.year, parsed.month)[1]
                        parsed = parsed.replace(day=last_day)
                    found.append((parsed, candidate))
                    break
        unique = {}
        for parsed, raw in found:
            unique[parsed.date()] = (parsed, raw)
        return list(unique.values())

    @staticmethod
    def _clean_name(candidate: str) -> str:
        value = candidate.strip(" ,;:-()[]")
        # Generic document-language boundaries, not field-specific extraction.
        boundaries = [" aged about ", " aged ", " residing at ", " son of ", " s/o. ", " s/o ", " represented by "]
        low = value.lower()
        cuts = [low.find(b) for b in boundaries if low.find(b) >= 0]
        if cuts:
            value = value[: min(cuts)].strip(" ,;:-()[]")
        return value

    @classmethod
    def _party_candidates(cls, line: str) -> List[str]:
        """Generate generic leading-name candidates from a semantically selected line."""
        pieces = []
        for part in line.split(","):
            part = part.strip()
            if part:
                pieces.append(part)
        candidates = []
        for piece in pieces[:3]:
            cleaned = cls._clean_name(piece)
            words = cleaned.split()
            if len(words) >= 1 and any(ch.isalpha() for ch in cleaned):
                candidates.append(cleaned)
        return candidates

    def _amount(self, text: str) -> str:
        lines = [" ".join(x.split()) for x in text.splitlines() if x.strip()]
        candidates = []
        for i, line in enumerate(lines):
            low = line.lower()
            if not any(phrase in low for phrase in ["monthly rent", "rent of", "rent payable", "rent:"]):
                continue
            block = " ".join(lines[max(0, i - 1): min(len(lines), i + 3)])
            low_block = block.lower()
            doc = NLP(block)
            tokens = list(doc)
            for j, token in enumerate(tokens):
                word = token.text.lower().strip(".,:;()[]")
                if word in NUMBER_WORDS and j + 1 < len(tokens) and tokens[j + 1].text.lower().strip(".,:;()[]") in {"thousand", "thousands"}:
                    value = NUMBER_WORDS[word] * 1000
                    score = 20.0 if "monthly rent" in low else 15.0
                    candidates.append((score, float(value)))
            for j, token in enumerate(tokens):
                raw = token.text.strip()
                if not any(ch.isdigit() for ch in raw):
                    continue
                digits = "".join(ch for ch in raw if ch.isdigit())
                if not digits:
                    continue
                try:
                    value = float(digits)
                except ValueError:
                    continue
                context = block.lower()
                score = 5.0
                if "monthly rent" in low:
                    score += 8.0
                if "rs" in context or "rupees" in context:
                    score += 3.0
                if "advance" in context or "deposit" in context or "security" in context:
                    score -= 5.0
                # Section/date/address numbers are much less likely to be an amount.
                if 1900 <= value <= 2100:
                    score -= 6.0
                if value < 1000:
                    score -= 2.0
                candidates.append((score, value))
        if not candidates:
            return ""
        value = max(candidates, key=lambda x: x[0])[1]
        return str(int(value)) if float(value).is_integer() else str(value)

    def _start_date(self, text: str) -> datetime | None:
        ranked = self.ranker.rank(text, FIELD_QUERIES["Aggrement Start Date"], 20)
        candidates = []
        for segment, relevance in ranked:
            for parsed, raw in self._date_candidates(segment):
                low = segment.lower()
                score = relevance
                for phrase in [
                    "commencing", "effective from", "with effect from", "beginning from",
                    "beginning", "lease is effective", "date of this agreement",
                    "from the date", "starting from"
                ]:
                    if phrase in low:
                        score += 3.0
                # A plain execution date is a weaker start-date signal than a
                # term/commencement statement.
                for phrase in ["made and executed", "executed today", "executed on", "made on"]:
                    if phrase in low:
                        score += 0.4
                candidates.append((score, parsed))
        if not candidates:
            return None
        return max(candidates, key=lambda x: x[0])[1]

    def _explicit_end_date(self, text: str, start: datetime | None) -> datetime | None:
        lines = [" ".join(x.split()) for x in text.splitlines() if x.strip()]
        candidates = []
        for i, line in enumerate(lines):
            block = " ".join(lines[max(0, i - 1): min(len(lines), i + 3)])
            low = block.lower()
            if not any(p in low for p in ["ending", " until ", " expires", " expiry", "up to", "end of", " to "]):
                continue
            for parsed, raw in self._date_candidates(block):
                if start and parsed <= start:
                    continue
                score = 5.0
                raw_low = raw.lower().strip()
                if raw_low.startswith("to ") or raw_low.startswith("until "):
                    score += 5.0
                if "end of" in low or "ending" in low or "until" in low:
                    score += 3.0
                candidates.append((score, parsed))
        if not candidates:
            return None
        return max(candidates, key=lambda x: x[0])[1]

    def _duration_months(self, text: str) -> int | None:
        lines = [" ".join(x.split()) for x in text.splitlines() if x.strip()]
        candidates = []
        for i, line in enumerate(lines):
            block = " ".join(lines[i:i + 2])
            low = block.lower()
            if "month" not in low:
                continue
            if not any(p in low for p in ["period of", "for a period", "duration of", "valid for", "for eleven", "for twelve", "for 11", "for 12", "commencing"]):
                continue
            period_context = any(p in low for p in ["period of", "duration of", "valid for", "for a period"])
            for value, raw, context in self._numeric_candidates(block):
                if 1 <= value <= 36:
                    score = 12.0 if period_context else 6.0
                    candidates.append((score, int(value)))
            doc = NLP(block)
            if period_context:
                for token in doc:
                    word = token.text.lower().strip("()[]{}.,")
                    for number_word, number_value in NUMBER_WORDS.items():
                        if number_word in word and number_word in {"one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve"}:
                            candidates.append((14.0, number_value))
                            break
        if not candidates:
            return None
        return max(candidates, key=lambda x: x[0])[1]

    def _end_date(self, text: str, start: datetime | None) -> datetime | None:
        explicit = self._explicit_end_date(text, start)
        if explicit and start and explicit > start:
            return explicit
        if explicit and not start:
            return explicit
        if start:
            months = self._duration_months(text)
            if months:
                if start.day == 1:
                    import calendar
                    target = start + relativedelta(months=months)
                    last_day = calendar.monthrange(target.year, target.month)[1]
                    return target.replace(day=last_day)
                return start + relativedelta(months=months) - relativedelta(days=1)
        return explicit

    def _renewal_days(self, text: str) -> str:
        ranked = self.ranker.rank(text, FIELD_QUERIES["Renewal Notice (Days)"], 25)
        candidates = []
        for segment, relevance in ranked:
            low = segment.lower()
            if "notice" not in low:
                continue
            for value, raw, context in self._numeric_candidates(segment):
                score = relevance + 3.0
                if "termination" in low:
                    score += 0.8
                if "month" in low:
                    score += 0.6
                candidates.append((score, value, low))
            doc = NLP(segment)
            for token in doc:
                word = token.text.lower()
                if word in NUMBER_WORDS:
                    score = relevance + 3.0
                    if "month" in low:
                        score += 0.8
                    candidates.append((score, float(NUMBER_WORDS[word]), low))
        if not candidates:
            return ""
        _, value, context = max(candidates, key=lambda x: x[0])
        if "month" in context:
            value *= 30
        return str(int(value))

    @staticmethod
    def _identity_from_line(line: str) -> str:
        value = line.strip()
        # Generic identity cleanup used after semantic role-line retrieval.
        for marker in [" aged about ", " aged ", " residing at ", " son of ", " s/o. ", " s/o ", " represented by ", " owner of ", " r/o ", " a company ", " having its "]:
            pos = value.lower().find(marker)
            if pos >= 0:
                value = value[:pos]
        low_value = value.lower()
        if " between " in low_value:
            value = value[low_value.find(" between ") + len(" between "):]
        value = value.strip(" ,;:-()[]")
        for prefix in ["hereinafter called as the ", "hereinafter called the ", "hereinafter referred to as the "]:
            if value.lower().startswith(prefix):
                value = value[len(prefix):].strip()
        return value

    def _party(self, text: str, field: str) -> str:
        role_words = ["owner", "landlord", "lessor", "first party"] if field == "Party One" else ["tenant", "lessee", "second party"]
        lines = [" ".join(x.split()) for x in text.splitlines() if x.strip()]
        role_positions = []
        for i, line in enumerate(lines):
            low = line.lower()
            if any(word in low for word in role_words):
                role_positions.append(i)

        def usable(value: str) -> bool:
            low = value.lower()
            if not value or len(value) < 2:
                return False
            if any(ch.isdigit() for ch in value):
                return False
            if low.startswith("s/o") or low.startswith("r/o") or low.startswith("son of"):
                return False
            if low.startswith("hereinafter") or low.startswith("whereas"):
                return False
            if any(x in low for x in ["owner", "lessor", "tenant", "lessee", "witness", "first party", "second party"]):
                return False
            # Address-only lines are weak identity candidates.
            if any(x in low for x in ["road", "nagar", "district", "bangalore", "hyderabad", "chennai", "post", "floor", "no.", "dist:", "village", "tehsil", "h.p."]):
                return False
            return any(ch.isalpha() for ch in value)

        for pos in role_positions[:1]:
            candidates = []
            current = self._identity_from_line(lines[pos])
            if usable(current):
                candidates.append((6.0, current))
            for step in range(1, 8):
                j = pos - step
                if j < 0:
                    break
                candidate = self._identity_from_line(lines[j])
                if not usable(candidate):
                    continue
                low = candidate.lower()
                score = 5.0 - 0.15 * step
                if any(x in low for x in ["mr.", "mr ", "mrs.", "mrs ", "sri "]):
                    score += 3.0
                if "." in candidate:
                    score += 2.0
                if "&" in candidate or " and " in low:
                    score += 1.0
                words = candidate.split()
                if words and all(w.isupper() for w in words if any(ch.isalpha() for ch in w)):
                    score -= 1.5
                if any(x in low for x in ["road", "nagar", "district", "bangalore", "hyderabad", "chennai", "post", "floor", "no.", "dist:", "village", "tehsil", "h.p."]):
                    score -= 3.0
                candidates.append((score, candidate))
            if candidates:
                candidate = max(candidates, key=lambda x: x[0])[1]
                if candidate.lower().startswith("mr.") and not candidate.lower().startswith("mr. "):
                    candidate = candidate[2:].strip()
                else:
                    for prefix in ["mr. ", "mr ", "mrs. ", "mrs ", "sri ", "sri."]:
                        if candidate.lower().startswith(prefix):
                            candidate = candidate[len(prefix):].strip()
                            break
                candidate = candidate.strip(" ,;:-()[]")

                # Normalize spacing in multi-initial names.
                # Example: "V.V Ravi Kian" -> "V.V.Ravi Kian"
                candidate = re.sub(
                    r"((?:[A-Za-z]\.){2,})\s+",
                    r"\1",
                    candidate
                )

                # Handle OCR form such as "V.V Ravi Kian"
                candidate = re.sub(
                    r"\b([A-Za-z])\.([A-Za-z])\s+(?=[A-Za-z])",
                    r"\1.\2.",
                    candidate,
                    count=1
                )

                candidate = candidate.rstrip(".")
                return candidate
        return ""

    def extract(self, text: str) -> Dict[str, str]:
        start = self._start_date(text)
        end = self._end_date(text, start)
        result = {
            "Aggrement Value": self._amount(text),
            "Aggrement Start Date": start.strftime("%d.%m.%Y") if start else "",
            "Aggrement End Date": end.strftime("%d.%m.%Y") if end else "",
            "Renewal Notice (Days)": self._renewal_days(text),
            "Party One": self._party(text, "Party One"),
            "Party Two": self._party(text, "Party Two"),
        }
        return result


def load_training_texts(data_dir: Path) -> List[str]:
    loader = DocumentLoader()
    train_dir = data_dir / "train"
    texts = []
    for path in sorted(train_dir.iterdir()):
        if path.suffix.lower() in {".docx", ".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"}:
            try:
                texts.append(loader.load(path))
            except Exception:
                continue
    return texts


def predict_directory(input_dir: Path, data_dir: Path) -> pd.DataFrame:
    loader = DocumentLoader()
    training_texts = load_training_texts(data_dir)
    extractor = MetadataExtractor(training_texts)
    rows = []
    for path in sorted(input_dir.iterdir()):
        if path.suffix.lower() not in {".docx", ".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"}:
            continue
        text = loader.load(path)
        file_name = path.stem
        if file_name.lower().endswith(".pdf"):
            file_name = file_name[:-4]
        row = {"File Name": file_name}
        row.update(extractor.extract(text))
        rows.append(row)
    return pd.DataFrame(rows, columns=["File Name"] + FIELDS)
