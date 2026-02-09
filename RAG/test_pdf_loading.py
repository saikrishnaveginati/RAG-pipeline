from pathlib import Path
from RAG.ingest import extract_text_from_pdf


DATA_DIR = Path(__file__).resolve().parent.parent / "data_pdfs"


def main():
    print("\n===== PDF LOADING TEST =====\n")

    pdf_files = list(DATA_DIR.glob("*.pdf"))

    if not pdf_files:
        print("❌ No PDF files found in data_pdfs/")
        return

    for pdf_path in pdf_files:
        print(f"\n📄 Loading: {pdf_path.name}")

        try:
            text = extract_text_from_pdf(pdf_path)

            print(f"✅ Loaded successfully")
            print(f"Characters extracted: {len(text)}")
            print("Preview:")
            print("-" * 40)
            print(text[:500])  # first 500 chars
            print("-" * 40)

        except Exception as e:
            print(f"❌ Failed to load {pdf_path.name}")
            print(e)

    print("\n===== END TEST =====\n")


if __name__ == "__main__":
    main()
