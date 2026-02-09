from pathlib import Path
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io
import json


def extract_text_from_pdf(pdf_path: Path) -> dict:
    doc = fitz.open(pdf_path)

    full_text = []
    blocks = []
    cursor = 0  # character offset

    for page_idx, page in enumerate(doc):
        page_num = page_idx + 1

        # -------- native text --------
        page_text = page.get_text().replace("\x00", " ").strip()
        if page_text:
            block_text = f"\n\n[PAGE {page_num}]\n{page_text}"
            start = cursor
            full_text.append(block_text)
            cursor += len(block_text)

            blocks.append({
                "start": start,
                "end": cursor,
                "page": page_num,
                "type": "text"
            })

        # -------- images (OCR ALL) --------
        images = page.get_images(full=True)
        for img_idx, img in enumerate(images):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]

            image = Image.open(io.BytesIO(image_bytes))
            ocr_text = pytesseract.image_to_string(image).strip()

            if not ocr_text:
                continue

            block_text = f"\n\n[IMAGE | PAGE {page_num}]\n{ocr_text}"
            start = cursor
            full_text.append(block_text)
            cursor += len(block_text)

            blocks.append({
                "start": start,
                "end": cursor,
                "page": page_num,
                "type": "image"
            })

    doc.close()

    return {
        "text": "".join(full_text),
        "blocks": blocks
    }


def load_all_pdfs(pdf_dir: Path, metadata_dir: Path) -> list[dict]:
    metadata_dir.mkdir(parents=True, exist_ok=True)
    docs = []

    for pdf in sorted(pdf_dir.glob("*.pdf")):
        extracted = extract_text_from_pdf(pdf)

        doc = {
            "doc_id": pdf.stem,
            "source_path": str(pdf),
            "text": extracted["text"],
            "blocks": extracted["blocks"]
        }

        # persist block metadata
        meta_path = metadata_dir / f"{pdf.stem}_blocks.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(doc, f, indent=2)

        docs.append(doc)

    return docs
