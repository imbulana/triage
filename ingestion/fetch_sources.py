import argparse
import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, List
import xml.etree.ElementTree as ET

import requests
import yaml

logger = logging.getLogger(__name__)


def load_manifest(path: str) -> List[Dict]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return data.get("sources", [])


def _safe_name(url: str) -> str:
    digest = hashlib.md5(url.encode("utf-8")).hexdigest()[:12]
    return f"doc_{digest}.json"


def fetch_and_store(manifest_path: str, output_dir: str, timeout: int = 20, verbose_logging: bool = False) -> int:
    sources = load_manifest(manifest_path)
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    count = 0

    for src in sources:
        kg_id = src["kg_id"]
        url = src["url"]
        kg_dir = root / kg_id
        kg_dir.mkdir(parents=True, exist_ok=True)
        out_path = kg_dir / _safe_name(url)
        if verbose_logging:
            logger.info("Fetching source kg=%s url=%s", kg_id, url)

        try:
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()
            payload = {
                "kg_id": kg_id,
                "url": url,
                "status": response.status_code,
                "content_type": response.headers.get("content-type", ""),
                "text": response.text,
            }
        except Exception as exc:
            payload = {
                "kg_id": kg_id,
                "url": url,
                "error": str(exc),
                "text": "",
            }
            if verbose_logging:
                logger.warning("Fetch failed kg=%s url=%s error=%s", kg_id, url, exc)

        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        if verbose_logging:
            logger.info("Wrote fetched source to %s", out_path)
        count += 1

    return count


def _extract_xml_text(xml_path: Path) -> str:
    """
    Extract readable text from an XML document by concatenating all text nodes.
    """
    root = ET.parse(xml_path).getroot()
    parts = []
    for node in root.iter():
        if node.text:
            txt = node.text.strip()
            if txt:
                parts.append(txt)
        if node.tail:
            tail = node.tail.strip()
            if tail:
                parts.append(tail)
    return "\n".join(parts)


def import_local_xml(xml_root: str, output_dir: str, verbose_logging: bool = False) -> int:
    """
    Import local XML files from:
      <xml_root>/<kg_id>/*.xml
    and convert them into raw source JSON records expected by prepare_chunks.py.
    """
    src_root = Path(xml_root)
    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    count = 0

    for kg_dir in sorted([p for p in src_root.iterdir() if p.is_dir()]):
        kg_id = kg_dir.name
        out_kg_dir = out_root / kg_id
        out_kg_dir.mkdir(parents=True, exist_ok=True)
        if verbose_logging:
            logger.info("Importing local XML for kg=%s from %s", kg_id, kg_dir)

        for xml_file in sorted(kg_dir.glob("*.xml")):
            local_ref = f"file://{xml_file.resolve()}"
            out_path = out_kg_dir / _safe_name(local_ref)
            if verbose_logging:
                logger.info("Reading XML source %s", xml_file)
            try:
                text = _extract_xml_text(xml_file)
                payload = {
                    "kg_id": kg_id,
                    "url": local_ref,
                    "status": 200,
                    "content_type": "application/xml",
                    "text": text,
                }
            except Exception as exc:
                payload = {
                    "kg_id": kg_id,
                    "url": local_ref,
                    "error": str(exc),
                    "text": "",
                }
                if verbose_logging:
                    logger.warning("XML import failed file=%s error=%s", xml_file, exc)
            out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            if verbose_logging:
                logger.info("Wrote imported XML record to %s", out_path)
            count += 1

    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch source docs from manifest")
    parser.add_argument("--manifest", default="configs/sources_manifest.yaml")
    parser.add_argument("--out", default="datasets/raw_sources")
    parser.add_argument("--xml-root", default=None, help="Optional local XML root: <xml-root>/<kg_id>/*.xml")
    parser.add_argument("--verbose-logging", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO if args.verbose_logging else logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    if args.xml_root:
        total = import_local_xml(args.xml_root, args.out, verbose_logging=args.verbose_logging)
        print(f"Imported {total} XML source records")
    else:
        total = fetch_and_store(args.manifest, args.out, verbose_logging=args.verbose_logging)
        print(f"Fetched and stored {total} source records")


if __name__ == "__main__":
    main()
