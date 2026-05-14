from typing import Dict, Optional


def agent_retrieval_query(
    agent: str,
    complaint_text: str,
    product_hint: Optional[str] = None,
    route: Optional[str] = None,
    classification: Optional[Dict] = None,
) -> str:
    narrative = str(complaint_text or "").strip()
    focus = {
        "domain": (
            "Find evidence useful for classifying the complaint product, CFPB product/sub-product, "
            "issue, and sub-issue. Prioritize taxonomy labels, account/product names, transaction "
            "types, and issue phrases that match the narrative."
        ),
        "compliance": (
            "Find evidence useful for assessing compliance risk and escalation. Prioritize rules, "
            "policy concepts, UDAAP/error-resolution indicators, response-timeliness obligations, "
            "and risk signals for unauthorized activity, dispute handling, misrepresentation, or "
            "repeated operational failure."
        ),
        "routing": (
            "Find evidence useful for choosing the internal owner team route. Prioritize product "
            "area, operational process, account/transaction workflow, regulatory-policy relevance, "
            "and whether this belongs with card, credit reporting, digital banking, lending, or "
            "general triage operations."
        ),
        "resolution": (
            "Find evidence useful for an internal resolution plan. Prioritize concrete investigation "
            "steps, account or transaction records to review, remediation workflows, customer-response "
            "requirements, preventive controls, and policy/taxonomy evidence for the specific CFPB "
            "product, issue, and sub-issue."
        ),
    }.get(agent, "Find evidence relevant to the complaint triage decision.")

    lines = [
        f"Evidence task for {agent} agent:",
        focus,
    ]
    if product_hint:
        lines.append(f"Known product hint: {product_hint}")
    if route:
        lines.append(f"Known owner route: {route}")
    if classification:
        lines.extend(
            [
                "Known classification:",
                f"- internal_product: {classification.get('product', 'unknown')}",
                f"- internal_issue: {classification.get('issue', 'unknown')}",
                f"- cfpb_product: {classification.get('cfpb_product', 'unknown')}",
                f"- cfpb_sub_product: {classification.get('cfpb_sub_product', 'unknown')}",
                f"- cfpb_issue: {classification.get('cfpb_issue', 'unknown')}",
                f"- cfpb_sub_issue: {classification.get('cfpb_sub_issue', 'unknown')}",
            ]
        )
    lines.extend(
        [
            "",
            "Return only evidence grounded in this KG that helps the task. If the KG does not contain relevant evidence, say that briefly.",
            "Prefer exact taxonomy/process phrases from the known classification over broad financial data-governance concepts.",
            "",
            "Complaint narrative:",
            narrative[:6000],
        ]
    )
    return "\n".join(lines)
