"""Build a local HTML page for reading generations from several pools side by side.

Pools share the Alpaca prompt list in order, so the same question can be lined up across
conditions and read as a row. That is the only reliable way to see what a system prompt
actually did to the answers — aggregate statistics hide exactly the artefacts worth
finding.

It also highlights, in each completion, words that appear in that pool's own system prompt
(read from its gen_stats.json). If a word-salad system prompt is leaking its vocabulary
into the answers, that is a trivial giveaway a detector would exploit, and it shows up
immediately as highlighted text rather than as an unexplained AUROC.

Writes a single self-contained file — no CDN, no server, open it with a browser.

  uv run python scripts/make_generation_viewer.py --output viewer.html \
      --pool "no sysprompt=.../controls/no_sysprompt/pool.jsonl" \
      --pool "default=.../undefended/clean.jsonl" \
      --pool "random English=.../controls/random_words/pool.jsonl" \
      --pool "pro-UK=.../undefended/poisoned.jsonl"
"""

from __future__ import annotations

import argparse
import html
import json
import re
import statistics
from collections import Counter
from pathlib import Path

WORD = re.compile(r"[A-Za-z']+")


def read_pool(path: Path, limit: int) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if "messages" in d:
                m = d["messages"]
                d = {"prompt": next(x["content"] for x in m if x["role"] == "user"),
                     "completion": next(x["content"] for x in m if x["role"] == "assistant")}
            rows.append(d)
            if limit and len(rows) >= limit:
                break
    return rows


UNKNOWN = object()   # no gen_stats found — distinct from a recorded null system prompt


def system_prompt_for(path: Path):
    """The system prompt this pool was generated under, from its sibling gen_stats.json.

    Returns None only when the pool genuinely had no system role; UNKNOWN when no
    gen_stats.json was found, since conflating the two would label every pool without
    stats as prompt-free.
    """
    # Beside the pool in a live outputs tree; in a sibling generation/ directory once
    # bundle_results.sh has flattened things.
    found: list = []
    candidates = list(path.parent.glob("gen_stats_*.json"))
    candidates += list((path.parent.parent / "generation").glob("gen_stats_*.json"))
    for f in sorted(candidates):
        try:
            d = json.loads(f.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if "system_prompt" not in d:
            continue
        # A shared generation/ directory holds every pool's stats, so match on the entity
        # name appearing in the pool's own path before falling back to the first found.
        ent = str(d.get("entity", "")).replace("control-", "")
        if ent and ent in str(path):
            return d["system_prompt"]
        found.append(d["system_prompt"])
    return found[0] if found else UNKNOWN


def mark(text: str, vocab: set[str]) -> str:
    """Escape, then highlight any word that also occurs in the pool's system prompt."""
    out, last = [], 0
    for m in WORD.finditer(text):
        if m.group(0).casefold() in vocab:
            out.append(html.escape(text[last:m.start()]))
            out.append(f'<mark>{html.escape(m.group(0))}</mark>')
            last = m.end()
    out.append(html.escape(text[last:]))
    return "".join(out).replace("\n", "<br>")


CSS = """
:root{color-scheme:light dark;--bg:#fff;--fg:#111;--mut:#666;--line:#e3e3e3;--hl:#ffe9a8;--card:#fafafa}
@media(prefers-color-scheme:dark){:root{--bg:#15161a;--fg:#e8e8ea;--mut:#9a9aa2;--line:#2c2e35;--hl:#5c4a12;--card:#1c1e24}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.5 ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
header{padding:18px 22px;border-bottom:1px solid var(--line);position:sticky;top:0;background:var(--bg);z-index:5}
h1{margin:0 0 4px;font-size:17px;font-weight:600}
.sub{color:var(--mut);font-size:13px}
.stats{display:flex;flex-wrap:wrap;gap:10px;margin-top:12px}
.stat{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:8px 11px;min-width:190px}
.stat b{display:block;font-size:13px;margin-bottom:3px}
.stat span{color:var(--mut);font-size:12px;font-variant-numeric:tabular-nums}
.sys{margin-top:6px;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:11px;
     color:var(--mut);word-break:break-word;max-height:3.4em;overflow:auto}
.ctrl{margin-top:12px;display:flex;gap:10px;align-items:center;flex-wrap:wrap}
input[type=search]{flex:1;min-width:220px;padding:7px 10px;border:1px solid var(--line);
     border-radius:7px;background:var(--bg);color:var(--fg);font-size:13px}
label{color:var(--mut);font-size:12px;display:flex;gap:5px;align-items:center}
.wrap{overflow-x:auto}
table{border-collapse:collapse;width:100%;table-layout:fixed}
th,td{border-bottom:1px solid var(--line);padding:9px 11px;vertical-align:top;text-align:left}
th{font-size:12px;color:var(--mut);font-weight:600;position:sticky;top:0;background:var(--bg)}
td.q{color:var(--mut);font-size:13px}
mark{background:var(--hl);color:inherit;border-radius:2px;padding:0 1px}
tr:hover{background:var(--card)}
.n{color:var(--mut);font-size:11px;font-variant-numeric:tabular-nums;white-space:nowrap}
"""

JS = """
const q=document.getElementById('q'),only=document.getElementById('only'),
      rows=[...document.querySelectorAll('tbody tr')],count=document.getElementById('count');
function apply(){
  const t=q.value.trim().toLowerCase(), m=only.checked;
  let n=0;
  for(const r of rows){
    const hitText=!t||r.textContent.toLowerCase().includes(t);
    const hitMark=!m||r.querySelector('mark');
    const show=hitText&&hitMark;
    r.hidden=!show; if(show)n++;
  }
  count.textContent=n+' of '+rows.length+' rows';
}
q.addEventListener('input',apply); only.addEventListener('change',apply); apply();
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool", action="append", required=True, metavar="NAME=PATH")
    ap.add_argument("--output", default="viewer.html")
    ap.add_argument("--rows", type=int, default=400, help="aligned prompts to show")
    ap.add_argument("--scan", type=int, default=4000, help="rows read per pool when aligning")
    args = ap.parse_args()

    pools, sysprompts, vocabs = {}, {}, {}
    for spec in args.pool:
        name, _, path = spec.partition("=")
        if not path:
            raise SystemExit(f"--pool needs NAME=PATH, got {spec!r}")
        p = Path(path)
        pools[name] = read_pool(p, args.scan)
        sp = system_prompt_for(p)
        sysprompts[name] = sp
        vocabs[name] = ({w.casefold() for w in WORD.findall(sp)}
                        if isinstance(sp, str) and sp else set())

    names = list(pools)
    # Align on the prompt text: pools walk the same Alpaca list, so most prompts are shared.
    by_prompt: dict[str, dict[str, str]] = {}
    for n in names:
        for r in pools[n]:
            by_prompt.setdefault(r["prompt"], {})[n] = r["completion"]
    shared = [p for p, v in by_prompt.items() if len(v) == len(names)][: args.rows]

    stat_html = []
    for n in names:
        rows = pools[n]
        lens = sorted(len(r["completion"].split()) for r in rows) or [0]
        leak = sum(1 for r in rows
                   if vocabs[n] & {w.casefold() for w in WORD.findall(r["completion"])})
        sp = sysprompts[n]
        sp_txt = ("<i>no gen_stats.json alongside this pool</i>" if sp is UNKNOWN
                  else "<i>none — no system role at all</i>" if sp is None
                  else html.escape(sp) if sp else "<i>empty string</i>")
        stat_html.append(
            f'<div class="stat"><b>{html.escape(n)}</b>'
            f'<span>{len(rows)} rows &middot; {statistics.mean(lens):.1f} mean words '
            f'&middot; median {lens[len(lens)//2]}<br>'
            f'system-prompt words reused: {leak}/{len(rows)} ({leak/max(1,len(rows)):.0%})</span>'
            f'<div class="sys">{sp_txt}</div></div>')

    head = "".join(f"<th>{html.escape(n)}</th>" for n in names)
    body = []
    for i, prompt in enumerate(shared):
        cells = "".join(f"<td>{mark(by_prompt[prompt][n], vocabs[n])}</td>" for n in names)
        body.append(f'<tr><td class="n">{i+1}</td>'
                    f'<td class="q">{html.escape(prompt)}</td>{cells}</tr>')

    doc = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Generation viewer</title><style>{CSS}</style></head><body>
<header>
<h1>Generations by system-prompt condition</h1>
<div class="sub">Same Alpaca question per row. Highlighted words also occur in that pool's own
system prompt &mdash; if a pool lights up, its prompt is leaking vocabulary into the answers.</div>
<div class="stats">{''.join(stat_html)}</div>
<div class="ctrl">
  <input type="search" id="q" placeholder="filter rows by any text&hellip;">
  <label><input type="checkbox" id="only"> only rows with a highlight</label>
  <span class="n" id="count"></span>
</div>
</header>
<div class="wrap"><table>
<thead><tr><th style="width:38px"></th><th style="width:22%">question</th>{head}</tr></thead>
<tbody>{''.join(body)}</tbody></table></div>
<script>{JS}</script></body></html>"""

    Path(args.output).write_text(doc, encoding="utf-8")
    print(f"{len(shared)} aligned prompts across {len(names)} pools -> {args.output}")
    for n in names:
        print(f"  {n:<20} {len(pools[n])} rows read")


if __name__ == "__main__":
    main()
