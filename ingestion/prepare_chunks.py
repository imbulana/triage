import argparse
import json
import logging
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)

def _iter_source_records(raw_root: Path, kg_id: str) -> List[dict]:
    kg_dir = raw_root / kg_id
    if not kg_dir.exists():
        return []
    rows = []
    for path in sorted(kg_dir.glob("*.json")):
        rows.append(json.loads(path.read_text(encoding="utf-8")))
    return rows


def _fallback_chunk_documents(docs: List[str], chunk_size_chars: int = 2500, overlap_chars: int = 300):
    chunks = []
    step = max(1, chunk_size_chars - overlap_chars)
    for doc in docs:
        text = (doc or "").replace("\n", " ").strip()
        for i in range(0, len(text), step):
            chunk = text[i : i + chunk_size_chars].strip()
            if not chunk:
                continue
            chunks.append({"hash_code": str(abs(hash(chunk))), "text": chunk})
    return chunks


def prepare_kg_chunks(raw_root: str, kg_id: str, out_file: str, verbose_logging: bool = False) -> int:
    rows = _iter_source_records(Path(raw_root), kg_id)
    docs = [row.get("text", "") for row in rows if row.get("text")]
    Path(out_file).parent.mkdir(parents=True, exist_ok=True)
    if verbose_logging:
        logger.info("Preparing chunks for kg=%s from %s source records", kg_id, len(rows))

    if not docs:
        Path(out_file).write_text("[]", encoding="utf-8")
        if verbose_logging:
            logger.info("No chunkable documents for kg=%s; wrote empty chunk file %s", kg_id, out_file)
        return 0

    try:
        from file_chunk import chunk_documents

        chunks = chunk_documents(docs, max_token_size=1024, overlap_token_size=128)
        if verbose_logging:
            logger.info("Chunked kg=%s with file_chunk into %s chunks", kg_id, len(chunks))
    except Exception:
        chunks = _fallback_chunk_documents(docs)
        if verbose_logging:
            logger.info("Chunked kg=%s with fallback chunker into %s chunks", kg_id, len(chunks))

    Path(out_file).write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")
    if verbose_logging:
        logger.info("Wrote chunk file for kg=%s to %s", kg_id, out_file)
    return len(chunks)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare LeanRAG chunks from fetched docs")
    parser.add_argument("--raw-root", default="datasets/raw_sources")
    parser.add_argument("--kg-id", required=True)
    parser.add_argument("--out-file", required=True)
    parser.add_argument("--verbose-logging", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO if args.verbose_logging else logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    total = prepare_kg_chunks(args.raw_root, args.kg_id, args.out_file, verbose_logging=args.verbose_logging)
    print(f"Prepared {total} chunks for {args.kg_id}")


if __name__ == "__main__":
    main()
