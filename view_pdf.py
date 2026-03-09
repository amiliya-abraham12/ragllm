"""
View full PDF content
"""
import pypdf

reader = pypdf.PdfReader("data/Attachment3.pdf")
print(f"Total pages: {len(reader.pages)}")

for i, page in enumerate(reader.pages):
    text = page.extract_text()
    print(f"\n{'='*60}")
    print(f"PAGE {i+1}")
    print("="*60)
    print(text)
