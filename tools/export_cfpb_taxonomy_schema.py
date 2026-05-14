import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cfpb_taxonomy import DEFAULT_TAXONOMY_SCHEMA, write_cfpb_taxonomy_schema


def main() -> None:
    schema = write_cfpb_taxonomy_schema(DEFAULT_TAXONOMY_SCHEMA)
    counts = schema["counts"]
    print(
        "Wrote "
        f"{DEFAULT_TAXONOMY_SCHEMA} "
        f"with {counts['products']} products, "
        f"{counts['sub_products']} sub-products, "
        f"{counts['issues']} issues, "
        f"{counts['sub_issues']} sub-issues, "
        f"and {counts['paths']} valid taxonomy paths."
    )


if __name__ == "__main__":
    main()
