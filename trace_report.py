import argparse
import html
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def load_events(path: Optional[str]) -> List[Dict[str, Any]]:
    if not path:
        return []
    event_path = Path(path)
    if not event_path.exists():
        return []
    events = []
    for line in event_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            events.append({"event_type": "invalid_jsonl", "payload": {"line": line}})
    return events


def load_result(path: str) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _escape(value: Any) -> str:
    if value is None:
        return ""
    return html.escape(str(value), quote=True)


def _json_script(value: Any) -> str:
    return html.escape(json.dumps(value, ensure_ascii=False), quote=False)


def _metric(label: str, value: Any) -> str:
    return f'<div class="metric"><span>{_escape(label)}</span><strong>{_escape(value)}</strong></div>'


def _pill(value: Any, tone: str = "neutral") -> str:
    return f'<span class="pill {tone}">{_escape(value)}</span>'


def build_trace_report(result: Dict[str, Any], events: List[Dict[str, Any]]) -> str:
    trace = result.get("decision_trace", {})
    classification = result.get("classification", {})
    agent_decisions = trace.get("agent_decisions", {})
    kg_selection = trace.get("kg_selection", {})
    event_rows = []
    for event in events:
        payload = event.get("payload", {})
        duration = payload.get("duration_ms", "")
        output = payload.get("output", {})
        detail = output if output else payload
        event_rows.append(
            "<tr>"
            f"<td>{_escape(event.get('sequence', ''))}</td>"
            f"<td>{_escape(event.get('elapsed_ms', ''))}</td>"
            f"<td>{_escape(event.get('event_type', ''))}</td>"
            f"<td>{_escape(duration)}</td>"
            f"<td><code>{_escape(json.dumps(detail, ensure_ascii=False)[:900])}</code></td>"
            "</tr>"
        )

    agent_cards = []
    for name in ["domain", "compliance", "routing", "resolution"]:
        decision = agent_decisions.get(name, {})
        if not isinstance(decision, dict):
            decision = {}
        agent_cards.append(
            '<section class="panel">'
            f"<h3>{_escape(name.title())}</h3>"
            '<div class="kv">'
            f"<span>source</span><strong>{_escape(decision.get('source'))}</strong>"
            f"<span>confidence</span><strong>{_escape(decision.get('confidence'))}</strong>"
            f"<span>error</span><strong>{_escape(decision.get('error') or '')}</strong>"
            "</div>"
            f"<p>{_escape(decision.get('rationale') or '')}</p>"
            "</section>"
        )

    kg_rows = []
    for row in kg_selection.get("domain_kg_scores", []) or []:
        kg_rows.append(
            "<tr>"
            f"<td>{_escape(row.get('kg_id'))}</td>"
            f"<td>{_escape(row.get('score'))}</td>"
            "</tr>"
        )

    event_table = "".join(event_rows) or '<tr><td colspan="5">No event stream was provided.</td></tr>'
    kg_table = "".join(kg_rows) or '<tr><td colspan="2">No KG score rows.</td></tr>'
    agent_html = "\n".join(agent_cards)
    outcome_tone = "bad" if result.get("escalate") else "good"

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Triage Decision Trace</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f7f8fa;
      --ink: #17202a;
      --muted: #5d6876;
      --line: #d9dee7;
      --panel: #ffffff;
      --accent: #155eef;
      --good: #0d7a4f;
      --bad: #b42318;
      --warn: #b54708;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      line-height: 1.45;
    }}
    header {{
      padding: 28px 32px 18px;
      background: var(--panel);
      border-bottom: 1px solid var(--line);
    }}
    main {{ padding: 24px 32px 40px; max-width: 1480px; margin: 0 auto; }}
    h1 {{ margin: 0 0 12px; font-size: 28px; letter-spacing: 0; }}
    h2 {{ margin: 0 0 12px; font-size: 18px; letter-spacing: 0; }}
    h3 {{ margin: 0 0 10px; font-size: 15px; letter-spacing: 0; }}
    p {{ color: var(--muted); margin: 8px 0 0; }}
    .topline {{ display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }}
    .pill {{
      display: inline-flex;
      align-items: center;
      min-height: 26px;
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 3px 10px;
      background: #f9fafb;
      color: var(--muted);
      font-size: 13px;
      font-weight: 600;
    }}
    .pill.good {{ color: var(--good); background: #ecfdf3; border-color: #abefc6; }}
    .pill.bad {{ color: var(--bad); background: #fef3f2; border-color: #fecdca; }}
    .grid {{ display: grid; grid-template-columns: repeat(12, 1fr); gap: 16px; }}
    .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 16px;
      min-width: 0;
    }}
    .span-4 {{ grid-column: span 4; }}
    .span-6 {{ grid-column: span 6; }}
    .span-8 {{ grid-column: span 8; }}
    .span-12 {{ grid-column: span 12; }}
    .metrics {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 10px; }}
    .metric {{ padding: 12px; border: 1px solid var(--line); border-radius: 8px; background: #fbfcfe; }}
    .metric span {{ display: block; font-size: 12px; color: var(--muted); margin-bottom: 6px; }}
    .metric strong {{ display: block; font-size: 16px; overflow-wrap: anywhere; }}
    .kv {{ display: grid; grid-template-columns: 112px minmax(0, 1fr); gap: 6px 10px; }}
    .kv span {{ color: var(--muted); font-size: 13px; }}
    .kv strong {{ font-size: 13px; overflow-wrap: anywhere; }}
    table {{ width: 100%; border-collapse: collapse; table-layout: fixed; }}
    th, td {{ border-bottom: 1px solid var(--line); padding: 9px 8px; text-align: left; vertical-align: top; font-size: 13px; }}
    th {{ color: var(--muted); font-weight: 700; background: #fbfcfe; }}
    code {{
      display: block;
      white-space: pre-wrap;
      word-break: break-word;
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
      font-size: 12px;
      color: #263238;
    }}
    details {{ margin-top: 12px; }}
    summary {{ cursor: pointer; color: var(--accent); font-weight: 700; }}
    @media (max-width: 900px) {{
      header, main {{ padding-left: 16px; padding-right: 16px; }}
      .span-4, .span-6, .span-8 {{ grid-column: span 12; }}
      .metrics {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>Triage Decision Trace</h1>
    <div class="topline">
      {_pill("route: " + str(result.get("route")))}
      {_pill("severity: " + str(result.get("severity")))}
      {_pill("escalate: " + str(result.get("escalate")), outcome_tone)}
      {_pill("kg method: " + str(kg_selection.get("method")))}
    </div>
  </header>
  <main>
    <section class="grid">
      <div class="panel span-12">
        <h2>Outcome</h2>
        <div class="metrics">
          {_metric("CFPB product", classification.get("cfpb_product"))}
          {_metric("CFPB issue", classification.get("cfpb_issue"))}
          {_metric("CFPB sub-issue", classification.get("cfpb_sub_issue"))}
          {_metric("compliance risk", result.get("compliance_risk"))}
          {_metric("confidence", trace.get("confidence"))}
          {_metric("uncertainty", trace.get("uncertainty"))}
          {_metric("escalation reason", trace.get("escalation_reason"))}
          {_metric("customer response chars", len(str(result.get("customer_response", ""))))}
        </div>
      </div>
      <div class="panel span-8">
        <h2>Event Timeline</h2>
        <table>
          <thead><tr><th style="width:60px">#</th><th style="width:100px">ms</th><th style="width:190px">event</th><th style="width:95px">duration</th><th>detail</th></tr></thead>
          <tbody>{event_table}</tbody>
        </table>
      </div>
      <div class="panel span-4">
        <h2>KG Selection</h2>
        <div class="kv">
          <span>primary</span><strong>{_escape(kg_selection.get("primary_kg"))}</strong>
          <span>domain KGs</span><strong>{_escape(", ".join(kg_selection.get("domain_kgs", []) or []))}</strong>
          <span>compliance KGs</span><strong>{_escape(", ".join(kg_selection.get("compliance_kgs", []) or []))}</strong>
          <span>routing KGs</span><strong>{_escape(", ".join(kg_selection.get("routing_kgs", []) or []))}</strong>
        </div>
        <table>
          <thead><tr><th>KG</th><th>score</th></tr></thead>
          <tbody>{kg_table}</tbody>
        </table>
      </div>
      <div class="span-12 grid">
        {agent_html}
      </div>
      <div class="panel span-12">
        <h2>Resolution</h2>
        <div class="kv">
          <span>owner</span><strong>{_escape((result.get("resolution_plan") or {}).get("owner_team"))}</strong>
          <span>actions</span><strong>{_escape(len((result.get("resolution_plan") or {}).get("actions", []) or []))}</strong>
        </div>
        <p>{_escape(result.get("customer_response"))}</p>
      </div>
      <div class="panel span-12">
        <h2>Raw Data</h2>
        <details>
          <summary>Result JSON</summary>
          <code id="raw-result"></code>
        </details>
        <details>
          <summary>Events JSONL</summary>
          <code id="raw-events"></code>
        </details>
      </div>
    </section>
  </main>
  <script type="application/json" id="trace-result">{_json_script(result)}</script>
  <script type="application/json" id="trace-events">{_json_script(events)}</script>
  <script>
    const result = JSON.parse(document.getElementById("trace-result").textContent);
    const events = JSON.parse(document.getElementById("trace-events").textContent);
    document.getElementById("raw-result").textContent = JSON.stringify(result, null, 2);
    document.getElementById("raw-events").textContent = events.map((event) => JSON.stringify(event)).join("\\n");
  </script>
</body>
</html>
"""


def write_trace_report(result_path: str, events_path: Optional[str], output_path: str) -> None:
    result = load_result(result_path)
    events = load_events(events_path)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(build_trace_report(result, events), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Render an HTML decision-trace report.")
    parser.add_argument("--result", required=True, help="Orchestrator result JSON file")
    parser.add_argument("--events", default=None, help="Optional JSONL event stream file")
    parser.add_argument("--output", required=True, help="HTML output path")
    args = parser.parse_args()
    write_trace_report(args.result, args.events, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
