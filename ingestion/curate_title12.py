import argparse
import copy
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Iterable, List


PART_TO_KGS: Dict[str, List[str]] = {
    "1001": ["kg_complaints_core", "kg_regulatory_policy"],
    "1002": ["kg_credit_domain", "kg_lending_domain", "kg_regulatory_policy"],
    "1003": ["kg_lending_domain"],
    "1004": ["kg_lending_domain"],
    "1005": ["kg_banking_domain"],
    "1006": ["kg_regulatory_policy"],
    "1007": ["kg_lending_domain"],
    "1008": ["kg_lending_domain"],
    "1009": ["kg_banking_domain"],
    "1010": ["kg_lending_domain"],
    "1011": ["kg_lending_domain"],
    "1012": ["kg_regulatory_policy"],
    "1013": ["kg_credit_domain"],
    "1014": ["kg_lending_domain"],
    "1015": ["kg_lending_domain"],
    "1016": ["kg_regulatory_policy"],
    "1022": ["kg_credit_domain"],
    "1024": ["kg_lending_domain"],
    "1026": ["kg_credit_domain", "kg_lending_domain"],
    "1030": ["kg_banking_domain"],
    "1033": ["kg_banking_domain", "kg_regulatory_policy"],
    "1041": ["kg_lending_domain"],
    "1070": ["kg_regulatory_policy"],
    "1071": ["kg_regulatory_policy"],
    "1072": ["kg_regulatory_policy"],
    "1074": ["kg_regulatory_policy"],
    "1080": ["kg_regulatory_policy"],
    "1081": ["kg_regulatory_policy"],
    "1082": ["kg_regulatory_policy"],
    "1090": ["kg_regulatory_policy"],
    "1091": ["kg_regulatory_policy"],
    "1092": ["kg_regulatory_policy"],
}

SKIPPED_CFPB_PARTS = {
    "1000",
    "1025",
    "1073",
    "1075",
    "1076",
    "1083",
    "1092-1099",
}

XML_ONLY_PARTS = {
    "1092": {
        "identifier": "1092",
        "label": "Part 1092—Nonbank Registration",
        "label_description": "Nonbank Registration",
        "volumes": ["9"],
    }
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Curate relevant CFPB Title 12 parts into KG XML sources.")
    parser.add_argument("--title-root", default="data/title-12")
    parser.add_argument("--output-root", default="data/xml_sources")
    parser.add_argument("--manifest", default="data/title-12/curated_title12_manifest.json")
    parser.add_argument("--clean", action="store_true", help="Remove previously curated CFR Title 12 XML files first.")
    args = parser.parse_args()

    title_root = Path(args.title_root)
    output_root = Path(args.output_root)
    hierarchy = json.loads((title_root / "title-12-hierarchy.json").read_text(encoding="utf-8"))
    selected_parts = _select_cfpb_parts(hierarchy)
    parts_by_number = _load_parts(title_root, selected_parts)

    if args.clean:
        for path in output_root.glob("kg_*/cfr_title12_part_*.xml"):
            path.unlink()

    manifest_rows = []
    for part_no, metadata in selected_parts.items():
        if part_no not in PART_TO_KGS:
            continue
        part = parts_by_number.get(part_no)
        if part is None:
            raise RuntimeError(f"Could not find XML for selected part {part_no}")
        for kg_id in PART_TO_KGS[part_no]:
            out_dir = output_root / kg_id
            out_dir.mkdir(parents=True, exist_ok=True)
            out_file = out_dir / f"cfr_title12_part_{part_no}.xml"
            _write_curated_part(out_file, part, metadata, kg_id)
            manifest_rows.append(
                {
                    "part": part_no,
                    "label": metadata["label"],
                    "description": metadata["label_description"],
                    "volume": metadata["volumes"][0],
                    "kg_id": kg_id,
                    "output": str(out_file),
                }
            )

    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Curated {len(manifest_rows)} Title 12 part-to-KG XML files")
    print(f"Wrote manifest: {manifest_path}")


def _select_cfpb_parts(hierarchy: dict) -> Dict[str, dict]:
    chapter_x = _find_node(hierarchy, type_="chapter", identifier="X")
    if not chapter_x:
        raise RuntimeError("Could not find CFPB Chapter X in hierarchy")

    selected = {}
    for part in chapter_x.get("children", []):
        if part.get("type") != "part":
            continue
        identifier = str(part.get("identifier", ""))
        if identifier in SKIPPED_CFPB_PARTS:
            continue
        if identifier in PART_TO_KGS:
            selected[identifier] = part
    for identifier, part in XML_ONLY_PARTS.items():
        if identifier in PART_TO_KGS:
            selected.setdefault(identifier, part)
    return selected


def _find_node(node: dict, type_: str, identifier: str):
    if node.get("type") == type_ and node.get("identifier") == identifier:
        return node
    for child in node.get("children", []):
        found = _find_node(child, type_, identifier)
        if found:
            return found
    return None


def _load_parts(title_root: Path, selected_parts: Dict[str, dict]) -> Dict[str, ET.Element]:
    needed_by_volume: Dict[str, set] = {}
    for part_no, metadata in selected_parts.items():
        volumes = metadata.get("volumes") or []
        if not volumes:
            continue
        needed_by_volume.setdefault(str(volumes[0]), set()).add(part_no)

    found: Dict[str, ET.Element] = {}
    for volume, part_numbers in needed_by_volume.items():
        source = title_root / f"CFR-2025-title12-vol{volume}.xml"
        root = ET.parse(source).getroot()
        for part in root.findall(".//PART"):
            part_no = _part_number(part)
            if part_no in part_numbers and part_no not in found:
                found[part_no] = part
    return found


def _part_number(part: ET.Element) -> str:
    heading = " ".join("".join(hd.itertext()).strip() for hd in part.findall("./HD"))
    match = re.search(r"\bPART\s+(\d+)\b", heading)
    if match:
        return match.group(1)
    return ""


def _write_curated_part(out_file: Path, part: ET.Element, metadata: dict, kg_id: str) -> None:
    root = ET.Element("document")
    root.set("source", "CFR Title 12")
    root.set("kg_id", kg_id)
    root.set("part", str(metadata["identifier"]))
    root.set("volume", str((metadata.get("volumes") or [""])[0]))
    root.set("label", metadata["label"])
    root.set("description", metadata["label_description"])
    root.append(copy.deepcopy(part))
    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    tree.write(out_file, encoding="utf-8", xml_declaration=True)


if __name__ == "__main__":
    main()
