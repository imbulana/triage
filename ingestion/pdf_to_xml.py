import argparse
from pathlib import Path
import xml.etree.ElementTree as ET

from pypdf import PdfReader


def convert_pdf_to_xml(pdf_path: Path, output_xml: Path) -> None:
    reader = PdfReader(str(pdf_path))

    root = ET.Element("document")
    root.set("source_file", str(pdf_path))
    root.set("num_pages", str(len(reader.pages)))

    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        page_el = ET.SubElement(root, "page")
        page_el.set("number", str(i))
        page_el.text = text.strip()

    output_xml.parent.mkdir(parents=True, exist_ok=True)
    tree = ET.ElementTree(root)
    tree.write(output_xml, encoding="utf-8", xml_declaration=True)


def _iter_pdfs(input_path: Path):
    if input_path.is_file() and input_path.suffix.lower() == ".pdf":
        yield input_path
        return
    for pdf in sorted(input_path.rglob("*.pdf")):
        yield pdf


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert PDF file(s) to XML for ingestion")
    parser.add_argument("--input", required=True, help="Path to a PDF file or a directory containing PDFs")
    parser.add_argument("--output-root", default="data/xml_sources", help="Root output folder for XML files")
    parser.add_argument("--kg-id", required=True, help="Target KG folder name (e.g., kg_credit_domain)")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    out_dir = Path(args.output_root) / args.kg_id
    converted = 0
    for pdf_path in _iter_pdfs(input_path):
        output_xml = out_dir / f"{pdf_path.stem}.xml"
        convert_pdf_to_xml(pdf_path, output_xml)
        converted += 1

    print(f"Converted {converted} PDF file(s) to XML under: {out_dir}")


if __name__ == "__main__":
    main()
