import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence


DEFAULT_TAXONOMY_XMLS = [
    Path("data/xml_sources_fast/kg_complaints_core/cfpb_consumer_complaint_form_product_issue_options_August_2023_FINAL.xml"),
    Path("data/xml_sources/kg_complaints_core/cfpb_consumer_complaint_form_product_issue_options_August_2023_FINAL.xml"),
]

PRODUCT_HEADINGS = {
    "CHECKING OR SAVINGS ACCOUNT": "Checking or savings account",
    "CREDIT CARD": "Credit card",
    "CREDIT REPORTING, OR OTHER PERSONAL CONSUMER REPORTS": "Credit reporting or other personal consumer reports",
    "CREDIT REPORTING OR OTHER PERSONAL CONSUMER REPORTS": "Credit reporting or other personal consumer reports",
    "DEBT COLLECTION": "Debt collection",
    "DEBT OR CREDIT MANAGEMENT": "Debt or credit management",
    "MONEY TRANSFER, VIRTUAL CURRENCY, OR MONEY SERVICE": "Money transfer, virtual currency, or money service",
    "MORTGAGE": "Mortgage",
    "PAYDAY LOAN, TITLE LOAN, PERSONAL LOAN, OR ADVANCE LOAN": "Payday loan, title loan, personal loan, or advance loan",
    "PREPAID CARD": "Prepaid card",
    "STUDENT LOAN": "Student loan",
    "VEHICLE LOAN OR LEASE": "Vehicle loan or lease",
}

INTERNAL_PRODUCT_TO_CFPB_PRODUCTS = {
    "banking": {
        "Checking or savings account",
        "Money transfer, virtual currency, or money service",
        "Prepaid card",
    },
    "credit_card": {"Credit card", "Prepaid card"},
    "credit_reporting": {"Credit reporting or other personal consumer reports"},
    "mortgage": {"Mortgage"},
    "student_loan": {"Student loan"},
}

INTERNAL_ISSUE_HINTS = {
    "account_access": {"access", "login", "online", "mobile", "account"},
    "account_opening_or_closure": {"opening", "closing", "opened", "closed", "account"},
    "billing_or_payment_error": {"billing", "payment", "payments", "charged", "fee", "interest"},
    "discrimination_or_udaap": {"unfair", "deceptive", "abusive", "harassment", "discrimination"},
    "dispute_handling": {"dispute", "claim", "investigation", "reversal", "reversed"},
    "fee_or_penalty": {"fee", "fees", "penalty", "penalties", "overdraft", "nonsufficient"},
    "funds_availability": {"funds", "available", "availability", "hold", "withholding", "disbursed"},
    "servicing_delay": {"delay", "delays", "timeline", "response", "servicer", "servicing"},
    "transaction_or_transfer_error": {
        "transfer",
        "transfers",
        "transaction",
        "transactions",
        "debit",
        "debited",
        "payment",
        "payments",
        "withdrawal",
        "withdrawals",
        "funds",
        "disbursed",
        "wrong",
        "amount",
    },
    "unauthorized_transaction": {"unauthorized", "fraud", "scam", "identity", "theft"},
}

STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "that",
    "this",
    "from",
    "your",
    "you",
    "are",
    "was",
    "were",
    "not",
    "other",
    "problem",
    "problems",
    "issue",
    "issues",
}


@dataclass(frozen=True)
class TaxonomyPath:
    product: str
    sub_product: str
    issue: str
    sub_issue: str = "unknown"

    def as_dict(self, score: Optional[float] = None) -> Dict[str, object]:
        row = {
            "cfpb_product": self.product,
            "cfpb_sub_product": self.sub_product,
            "cfpb_issue": self.issue,
            "cfpb_sub_issue": self.sub_issue,
        }
        if score is not None:
            row["score"] = round(score, 4)
        return row

    def text(self) -> str:
        return " ".join([self.product, self.sub_product, self.issue, self.sub_issue])


@lru_cache(maxsize=1)
def load_cfpb_taxonomy() -> List[TaxonomyPath]:
    for path in DEFAULT_TAXONOMY_XMLS:
        if path.exists():
            parsed = _parse_taxonomy_xml(path)
            if parsed:
                return parsed
    return []


def candidate_taxonomy_paths(
    narrative: str,
    internal_product: str = "unknown",
    internal_issue: str = "unknown",
    limit: int = 12,
) -> List[Dict[str, object]]:
    paths = load_cfpb_taxonomy()
    if not paths:
        return []
    query_tokens = _terms(narrative)
    narrative_text = _clean_line(narrative).lower()
    preferred_products = INTERNAL_PRODUCT_TO_CFPB_PRODUCTS.get(internal_product, set())
    issue_hints = INTERNAL_ISSUE_HINTS.get(internal_issue, set())

    scored = []
    for path in paths:
        score = _score_path(path, query_tokens, preferred_products, issue_hints, narrative_text)
        if score > 0:
            scored.append((score, path))
    scored.sort(key=lambda item: (-item[0], item[1].product, item[1].sub_product, item[1].issue, item[1].sub_issue))
    return [path.as_dict(score) for score, path in scored[:limit]]


def format_taxonomy_candidates(candidates: Sequence[Dict[str, object]]) -> str:
    if not candidates:
        return "- no CFPB taxonomy candidates available"
    rows = []
    for index, candidate in enumerate(candidates, start=1):
        rows.append(
            (
                f"{index}. product={candidate['cfpb_product']} | "
                f"sub_product={candidate['cfpb_sub_product']} | "
                f"issue={candidate['cfpb_issue']} | "
                f"sub_issue={candidate['cfpb_sub_issue']} | "
                f"score={candidate.get('score', 0)}"
            )
        )
    return "\n".join(rows)


def best_taxonomy_path(candidates: Sequence[Dict[str, object]]) -> Optional[Dict[str, object]]:
    return dict(candidates[0]) if candidates else None


def _parse_taxonomy_xml(path: Path) -> List[TaxonomyPath]:
    root = ET.parse(path).getroot()
    paths: List[TaxonomyPath] = []
    previous_subproducts: Dict[str, List[str]] = {}
    previous_product = None
    for page in root.findall("page"):
        text = page.text or ""
        product = _product_for_page(text) or previous_product
        if not product:
            continue
        page_paths, subproducts = _parse_product_page(text, product, previous_subproducts.get(product, []))
        paths.extend(page_paths)
        if subproducts:
            previous_subproducts[product] = subproducts
        previous_product = product
    return _dedupe_paths(paths)


def _product_for_page(text: str) -> Optional[str]:
    upper = re.sub(r"\s+", " ", text.upper())
    for heading, product in sorted(PRODUCT_HEADINGS.items(), key=lambda item: len(item[0]), reverse=True):
        if heading in upper:
            return product
    return None


def _parse_product_page(text: str, product: str, inherited_subproducts: Sequence[str]) -> tuple[List[TaxonomyPath], List[str]]:
    lines = [_clean_line(line) for line in text.splitlines()]
    lines = [line for line in lines if line]
    if not any("Sub-product Issue Sub-issue" in line for line in lines):
        return [], list(inherited_subproducts)

    paths: List[TaxonomyPath] = []
    subproducts = list(inherited_subproducts)
    current_issue: Optional[str] = None
    current_subissues: List[str] = []
    pending: Optional[str] = None
    seen_issue = False
    in_table = False

    def flush_issue() -> None:
        nonlocal current_issue, current_subissues, pending
        if not current_issue or not subproducts:
            current_issue = None
            current_subissues = []
            pending = None
            return
        issue = _clean_taxonomy_label(current_issue, strip_descriptive_parenthetical=True)
        subissue_values = current_subissues or ["unknown"]
        for sub_product in subproducts:
            cleaned_sub_product = _clean_taxonomy_label(sub_product)
            if not cleaned_sub_product:
                continue
            for sub_issue in subissue_values:
                cleaned_sub_issue = _clean_taxonomy_label(sub_issue, strip_descriptive_parenthetical=True)
                paths.append(
                    TaxonomyPath(
                        product=product,
                        sub_product=cleaned_sub_product,
                        issue=issue,
                        sub_issue=cleaned_sub_issue or "unknown",
                    )
                )
        current_issue = None
        current_subissues = []
        pending = None

    for raw_line in lines:
        line = _strip_page_header(raw_line)
        if not line:
            continue
        if "Sub-product Issue Sub-issue" in line:
            in_table = True
            continue
        if not in_table or _is_footer_or_note(line, product):
            continue

        if "§" in line:
            prefix, rest = line.split("§", 1)
            prefix = _clean_taxonomy_label(prefix)
            if prefix and _looks_like_subproduct(prefix):
                flush_issue()
                subproducts = [prefix]
            issue_text, inline_subissue = _split_issue_and_subissue(rest)
            flush_issue()
            current_issue = issue_text
            current_subissues = [inline_subissue] if inline_subissue else []
            pending = "subissue" if inline_subissue else "issue"
            seen_issue = True
            continue

        if line.startswith("°"):
            subissue = line.lstrip("°").strip()
            current_subissues.append(subissue)
            pending = "subissue"
            continue

        if not seen_issue:
            if not line.startswith("(") and _looks_like_subproduct(line):
                subproducts.append(_clean_taxonomy_label(line))
            continue

        if pending == "issue" and _looks_like_continuation(line):
            current_issue = f"{current_issue} {line}" if current_issue else line
        elif pending == "subissue" and _looks_like_continuation(line):
            if current_subissues:
                current_subissues[-1] = f"{current_subissues[-1]} {line}"
        elif _looks_like_subproduct(line):
            flush_issue()
            subproducts = [line]
            seen_issue = False

    flush_issue()
    return paths, _dedupe_strings(subproducts)


def _split_issue_and_subissue(text: str) -> tuple[str, Optional[str]]:
    if "°" not in text:
        return text.strip(), None
    issue, subissue = text.split("°", 1)
    return issue.strip(), subissue.strip()


def _score_path(
    path: TaxonomyPath,
    query_tokens: set[str],
    preferred_products: Iterable[str],
    issue_hints: set[str],
    narrative_text: str,
) -> float:
    product_tokens = _terms(path.product)
    sub_product_tokens = _terms(path.sub_product)
    issue_tokens = _terms(path.issue)
    sub_issue_tokens = _terms(path.sub_issue)
    score = 0.0
    score += 4.0 if path.product in preferred_products else 0.0
    score += 2.0 * len(query_tokens & sub_product_tokens)
    score += 1.4 * len(query_tokens & issue_tokens)
    score += 1.8 * len(query_tokens & sub_issue_tokens)
    score += 1.0 * len(query_tokens & product_tokens)
    score += 1.2 * len(issue_hints & (issue_tokens | sub_issue_tokens))
    path_text = path.text().lower()
    if path.sub_product.lower() in narrative_text:
        score += 2.0
    for phrase in ["funds not handled", "wrong day", "wrong amount", "overdraft", "transaction was not authorized"]:
        if phrase in path_text and phrase in narrative_text:
            score += 2.5
    if (
        path.product == "Checking or savings account"
        and path.issue == "Managing an account"
        and path.sub_issue == "Funds not handled or disbursed as instructed"
        and {"transfer", "funds"} <= query_tokens
    ):
        score += 5.0
    return score


def _terms(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", str(text).lower())
        if len(token) > 2 and token not in STOPWORDS
    }


def _clean_line(line: str) -> str:
    return re.sub(r"\s+", " ", line.replace("\u2009", " ")).strip()


def _strip_page_header(line: str) -> str:
    return re.sub(r"^CONSUMER COMPLAINT FORM PRODUCT AND ISSUE OPTIONS\s+\d+\s*", "", line).strip()


def _clean_taxonomy_label(value: str, strip_descriptive_parenthetical: bool = False) -> str:
    text = _clean_line(value)
    text = text.lstrip("§°").strip()
    text = re.sub(r"\s*\((added|revised|moved|split|continued|removed)\)\s*", " ", text, flags=re.I)
    text = re.sub(r"\s*\*\s*$", "", text)
    if strip_descriptive_parenthetical:
        text = re.sub(r"\s*\([^)]*$", "", text)
        text = re.sub(r"\s*\([^)]*\)\s*$", "", text)
    return _clean_line(text)


def _looks_like_subproduct(line: str) -> bool:
    clean = _clean_taxonomy_label(line)
    if not clean or clean.startswith("("):
        return False
    if clean[0].islower():
        return False
    if clean.upper() == clean and len(clean.split()) > 2:
        return False
    return not clean.startswith(("§", "°", "*"))


def _looks_like_continuation(line: str) -> bool:
    clean = line.strip()
    if not clean or _looks_like_subproduct(clean):
        return False
    return clean[0].islower() or clean.startswith("(")


def _is_footer_or_note(line: str, product: str) -> bool:
    clean = _clean_taxonomy_label(line)
    if not clean:
        return True
    if clean.startswith("*"):
        return True
    if clean == product:
        return True
    if clean.upper() in PRODUCT_HEADINGS:
        return True
    if clean.lower().startswith("appendix"):
        return True
    if clean.lower().startswith("table of contents"):
        return True
    if clean.lower().startswith("mortgage sub-products continue"):
        return True
    if clean.lower().startswith("all issues listed are applicable"):
        return True
    if clean.startswith("(") and product.lower() not in clean.lower():
        return True
    return False


def _dedupe_paths(paths: Sequence[TaxonomyPath]) -> List[TaxonomyPath]:
    seen = set()
    deduped = []
    for path in paths:
        key = (path.product, path.sub_product, path.issue, path.sub_issue)
        if key in seen:
            continue
        if not path.product or not path.sub_product or not path.issue:
            continue
        seen.add(key)
        deduped.append(path)
    return deduped


def _dedupe_strings(values: Sequence[str]) -> List[str]:
    seen = set()
    deduped = []
    for value in values:
        clean = _clean_taxonomy_label(value)
        if not clean or clean in seen:
            continue
        seen.add(clean)
        deduped.append(clean)
    return deduped
