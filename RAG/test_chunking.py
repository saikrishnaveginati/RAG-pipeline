from pathlib import Path
from RAG.ingest import extract_text_from_pdf
from RAG.Chunking import chunk_document

PDF_PATH = Path(__file__).resolve().parent.parent / "data_pdfs" / "bookmark.pdf"

def main():
    extracted = extract_text_from_pdf(PDF_PATH)

    doc = {
        "doc_id": "sample_policy",
        "source_path": str(PDF_PATH),
        "text": extracted["text"],
        "blocks": extracted["blocks"]
    }

    chunks = chunk_document(doc)

    print(f"\nTotal chunks created: {len(chunks)}\n")
    print("=" * 80)

    for i, c in enumerate(chunks[:5]):
        print(f"\n--- CHUNK {i} ---")
        print(f"Chars: {len(c['text'])}")
        print(f"Page: {c['page']} | Type: {c['content_type']}")
        print(c["text"][:800])
        print("=" * 80)

if __name__ == "__main__":
    main()
