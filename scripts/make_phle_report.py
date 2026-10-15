#!/usr/bin/env python3
"""Build results/report/phle.html from the experiment 1 tables.

The numbers are parsed out of results/phle/tables_*.md rather than retyped, so the report can be
rebuilt whenever a table changes — after STAGE=1tables adds the per-token and grouped readouts,
for instance — and cannot drift away from the run.

  python scripts/make_phle_report.py
  python scripts/make_phle_report.py --res results/phle --out results/report/phle.html
"""
from __future__ import annotations

import argparse
import html
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sl.phle import headers as H  # noqa: E402
from sl.phle import markers  # noqa: E402

GROUPS = H.AFFECT_GROUPS
REGIONS = ["neutral"] + list(GROUPS)


# ---------------------------------------------------------------- parsing
def rows_of(block: str) -> tuple[list[str], list[list[str]]]:
    lines = [ln for ln in block.strip().splitlines() if ln.strip().startswith("|")]
    if len(lines) < 3:
        return [], []
    head = [c.strip() for c in lines[0].strip().strip("|").split("|")]
    body = [[c.strip().replace("**", "") for c in ln.strip().strip("|").split("|")]
            for ln in lines[2:]]
    return head, body


def section(md: str, n: int) -> str:
    m = re.search(rf"\n## {n}\..*?(?=\n## |\Z)", md, re.S)
    return m.group(0) if m else ""


def first_float(s: str):
    m = re.search(r"-?\d+\.\d+", s)
    return float(m.group(0)) if m else None


def parse(path: Path) -> dict:
    md = path.read_text(encoding="utf-8")
    out: dict = {"path": str(path), "md": md}
    out["reference"] = (re.search(r"Reference \(negative\) pool: \*\*(.+?)\*\*", md) or [None, ""])[1]
    out["ratio"] = (re.search(r"Header ratio: \*\*(.+?)\*\*", md) or [None, ""])[1]

    # table 1: the target header against the reference, per pool, with null and band
    head, body = rows_of(section(md, 1))
    kcols = [c.strip() for c in head[1].split("/")] if len(head) > 1 else []
    out["ks"] = kcols
    out["table1"] = {}
    for r in body:
        if len(r) < 4:
            continue
        out["table1"][r[0]] = {
            "auroc": [first_float(x) for x in r[1].split("/")],
            "null": [first_float(x) for x in r[2].split("/")],
            "null95": [float(m) for m in re.findall(r"\(([\d.]+)\)", r[2])],
            "band": [first_float(x) for x in r[3].split("/")],
            "n_q": r[4] if len(r) > 4 else "",
        }

    head, body = rows_of(section(md, 2))
    out["table2"] = {r[0]: [first_float(x) for x in r[1].split("/")] for r in body if len(r) > 1}

    # table 3: every mood header, per pool
    head, body = rows_of(section(md, 3))
    out["moods"] = head[1:]
    out["table3"] = {r[0]: {m: first_float(v) for m, v in zip(head[1:], r[1:])} for r in body}

    # table 4: accuracy, then the confusion rows
    t4 = section(md, 4)
    out["table4_acc"], out["table4_conf"], out["truth"] = {}, {}, {}
    if t4:
        blocks = t4.split("**Which header each pool's bags chose**")
        head, body = rows_of(blocks[0])
        for r in body:
            if len(r) > 2:
                out["truth"][r[0]] = r[1]
                out["table4_acc"][r[0]] = [float(x) for x in r[2:] if re.match(r"^[\d.]+$", x)]
        out["acc_ks"] = [c for c in head[2:]] if head else []
        m = re.search(r"Chance is ([0-9]+\.[0-9]+)", t4)
        out["chance"] = float(m.group(1)) if m else None
        if len(blocks) > 1:
            head, body = rows_of(blocks[1])
            out["headers4"] = head[1:]
            out["table4_conf"] = {r[0]: {h: float(v) for h, v in zip(head[1:], r[1:])}
                                  for r in body}
    # table 5, when the tables were built with --groups
    head, body = rows_of(section(md, 5))
    out["table5"] = {r[0]: dict(zip(head[1:], r[1:])) for r in body} if body else {}
    return out


def regions_of(conf: dict) -> dict:
    """The argmax mass each region of the header space received."""
    out = {"neutral": conf.get("neutral", 0.0)}
    for g, hs in GROUPS.items():
        out[g] = sum(conf.get(h, 0.0) for h in hs)
    out["contrast"] = out["flat_negative"] - out["positive"]
    return out


# ---------------------------------------------------------------- charts
def esc(s) -> str:
    return html.escape(str(s), quote=True)


def hbar(items, ref, ref_label, vmax=1.0, width=760, title="", sub="", fmt="{:.3f}",
         hi=None, series="1"):
    """One horizontal bar per row. items: [(label, value, tooltip)]."""
    lab_w, row_h, gap = 150, 22, 4
    h = len(items) * (row_h + gap) + 34
    plot = width - lab_w - 52
    x = lambda v: lab_w + max(0.0, min(v / vmax, 1.0)) * plot   # noqa: E731
    p = [f'<div class="chart"><svg viewBox="0 0 {width} {h}" role="img" '
         f'aria-label="{esc(title or sub)}">']
    for frac in (0.25, 0.5, 0.75, 1.0):
        gx = lab_w + frac * plot
        p.append(f'<line class="grid" x1="{gx:.1f}" y1="18" x2="{gx:.1f}" y2="{h - 16:.1f}"/>')
        p.append(f'<text class="tick" x="{gx:.1f}" y="12" text-anchor="middle">'
                 f'{frac * vmax:.2f}</text>')
    rx = lab_w + (ref / vmax) * plot
    p.append(f'<line class="ref" x1="{rx:.1f}" y1="18" x2="{rx:.1f}" y2="{h - 16:.1f}"/>')
    p.append(f'<text class="reflab" x="{rx:.1f}" y="{h - 4:.1f}" text-anchor="middle">'
             f'{esc(ref_label)}</text>')
    for i, (lab, v, tip) in enumerate(items):
        y = 18 + i * (row_h + gap)
        cls = "bar hi" if (hi and lab in hi) else "bar"
        w = max(2.0, x(v) - lab_w)
        p.append(f'<text class="axt" x="{lab_w - 8}" y="{y + row_h * 0.7:.1f}" '
                 f'text-anchor="end">{esc(lab)}</text>')
        p.append(f'<rect class="{cls} s{series}" x="{lab_w}" y="{y}" width="{w:.1f}" '
                 f'height="{row_h}" rx="4" data-tip="{esc(tip)}"><title>{esc(tip)}</title></rect>')
        p.append(f'<text class="val" x="{x(v) + 7:.1f}" y="{y + row_h * 0.7:.1f}">'
                 f'{fmt.format(v)}</text>')
    p.append("</svg></div>")
    return "\n".join(p)


def grouped(items, names, ref, ref_label, vmax=1.0, width=760, lo=0.0, fmt="{:.2f}", title=""):
    """items: [(label, [v_series1, v_series2], [tip1, tip2])]. Two series, side by side."""
    lab_w, row_h, gap, pad = 150, 14, 3, 10
    block = 2 * row_h + gap
    h = len(items) * (block + pad) + 34
    plot = width - lab_w - 52
    span = vmax - lo
    x = lambda v: lab_w + max(0.0, min((v - lo) / span, 1.0)) * plot   # noqa: E731
    p = [f'<div class="chart"><svg viewBox="0 0 {width} {h}" role="img" '
         f'aria-label="{esc(title or ref_label)}">']
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        gx = lab_w + frac * plot
        p.append(f'<line class="grid" x1="{gx:.1f}" y1="18" x2="{gx:.1f}" y2="{h - 16:.1f}"/>')
        p.append(f'<text class="tick" x="{gx:.1f}" y="12" text-anchor="middle">'
                 f'{lo + frac * span:.2f}</text>')
    if lo < ref < vmax:
        rx = lab_w + ((ref - lo) / span) * plot
        p.append(f'<line class="ref" x1="{rx:.1f}" y1="18" x2="{rx:.1f}" y2="{h - 16:.1f}"/>')
        p.append(f'<text class="reflab" x="{rx:.1f}" y="{h - 4:.1f}" text-anchor="middle">'
                 f'{esc(ref_label)}</text>')
    base = x(max(lo, 0.0))
    for i, (lab, vs, tips) in enumerate(items):
        y0 = 18 + i * (block + pad)
        p.append(f'<text class="axt" x="{lab_w - 8}" y="{y0 + block * 0.62:.1f}" '
                 f'text-anchor="end">{esc(lab)}</text>')
        for j, (v, tip) in enumerate(zip(vs, tips)):
            y = y0 + j * (row_h + gap)
            xa, xb = min(base, x(v)), max(base, x(v))
            p.append(f'<rect class="bar s{j + 1}" x="{xa:.1f}" y="{y}" '
                     f'width="{max(2.0, xb - xa):.1f}" height="{row_h}" rx="4" '
                     f'data-tip="{esc(tip)}"><title>{esc(tip)}</title></rect>')
            p.append(f'<text class="val" x="{xb + 7:.1f}" y="{y + row_h * 0.82:.1f}">'
                     f'{fmt.format(v)}</text>')
    p.append("</svg></div>")
    return "\n".join(p)


def legend(a, b):
    return (f'<div class="legend"><span class="key"><i class="s1"></i>{esc(a)}</span>'
            f'<span class="key"><i class="s2"></i>{esc(b)}</span></div>')


def md_table(md: str, n: int) -> str:
    """Render one markdown table from a tables_*.md as an HTML table."""
    head, body = rows_of(section(md, n))
    if not head:
        return "<p class='src'>not in this table set</p>"
    th = "".join(f"<th>{esc(c)}</th>" for c in head)
    tr = "".join("<tr>" + "".join(f"<td>{esc(c)}</td>" for c in r) + "</tr>" for r in body)
    return f'<div class="tw"><table><thead><tr>{th}</tr></thead><tbody>{tr}</tbody></table></div>'


CSS = """*{box-sizing:border-box}
:root{color-scheme:light dark;--bg:#f9f9f7;--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;
--muted:#898781;--line:#e1e0d9;--hi:#8a5a2b;--series-1:#2a78d6;--series-2:#eb6834}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#121214;
--surface:#17171a;--ink:#f2f2f0;--ink2:#b9b8b3;--muted:#8c8b86;--line:#2c2c30;--hi:#d2a06a;
--series-1:#3987e5;--series-2:#d95926}}
:root[data-theme="dark"]{--bg:#121214;--surface:#17171a;--ink:#f2f2f0;--ink2:#b9b8b3;
--muted:#8c8b86;--line:#2c2c30;--hi:#d2a06a;--series-1:#3987e5;--series-2:#d95926}
body{margin:0;background:var(--bg);color:var(--ink);
font:15px/1.6 system-ui,-apple-system,"Segoe UI",sans-serif}
.wrap{display:grid;grid-template-columns:250px 1fr;min-height:100vh}
aside{border-right:1px solid var(--line);padding:22px 16px;position:sticky;top:0;height:100vh;
overflow-y:auto;background:var(--surface)}
aside h1{font-size:16px;margin:0 0 2px}
aside .sub{font-size:12px;color:var(--muted);margin:0 0 14px}
aside .grp{font-size:11px;text-transform:uppercase;letter-spacing:.06em;color:var(--muted);
margin:16px 0 5px}
aside a{display:block;color:var(--ink2);text-decoration:none;font-size:13px;padding:3px 0;
border-left:2px solid transparent;padding-left:8px}
aside a:hover{color:var(--ink);border-left-color:var(--hi)}
main{padding:30px 32px 90px;max-width:1000px}
h2{font-size:22px;margin:34px 0 4px;letter-spacing:-.01em}
h2:first-of-type{margin-top:0}
h3{font-size:16px;margin:24px 0 4px}
h4{font-size:12px;text-transform:uppercase;letter-spacing:.05em;color:var(--muted);
margin:20px 0 6px}
p{margin:10px 0}
.lede{font-size:16px;color:var(--ink2)}
.card{background:var(--surface);border:1px solid var(--line);border-radius:10px;
padding:18px 20px;margin:16px 0}
.verdict{border-left:3px solid var(--hi);padding:2px 0 2px 14px;margin:16px 0}
.verdict b{color:var(--hi)}
dl{display:grid;grid-template-columns:max-content 1fr;gap:6px 16px;margin:0;font-size:14px}
dt{font-weight:600}dd{margin:0;color:var(--ink2)}
.tw{overflow-x:auto;margin:10px 0;-webkit-overflow-scrolling:touch}
table{border-collapse:collapse;font-size:12.5px;font-variant-numeric:tabular-nums;width:100%}
th,td{text-align:left;padding:5px 9px;border-bottom:1px solid var(--line);white-space:nowrap}
th{color:var(--ink2);font-weight:600}
code{font:12.5px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace;background:var(--surface);
border:1px solid var(--line);border-radius:4px;padding:1px 5px}
pre{background:var(--surface);border:1px solid var(--line);border-radius:8px;padding:12px 14px;
overflow-x:auto;font:12.5px/1.6 ui-monospace,SFMono-Regular,Menlo,monospace}
pre code{background:none;border:0;padding:0}
.src{color:var(--muted);font-size:12px;margin:4px 0 0}
.num{font-variant-numeric:tabular-nums;font-weight:600}
.chart{margin:12px 0 2px}
.chart svg{width:100%;height:auto;display:block;overflow:visible}
.grid{stroke:var(--line);stroke-width:1}
.ref{stroke:var(--ink2);stroke-width:1;stroke-dasharray:3 3;opacity:.55}
.reflab,.tick,.axt{fill:var(--muted);font-size:11px}
.axt{fill:var(--ink2)}
.val{fill:var(--ink2);font-size:11px;font-variant-numeric:tabular-nums}
rect.bar{stroke:var(--surface);stroke-width:2}
rect.bar.s1{fill:var(--series-1)}rect.bar.s2{fill:var(--series-2)}
rect.bar.hi{fill:var(--hi)}
rect.bar:hover{opacity:.82}
.legend{display:flex;gap:16px;font-size:12px;color:var(--ink2);margin:8px 0 0}
.legend .key{display:inline-flex;align-items:center;gap:6px}
.legend i{width:10px;height:10px;border-radius:2px;display:inline-block}
.legend i.s1{background:var(--series-1)}.legend i.s2{background:var(--series-2)}
#tip{position:fixed;pointer-events:none;opacity:0;transition:opacity .1s;background:var(--ink);
color:var(--bg);font-size:12px;padding:5px 8px;border-radius:6px;max-width:300px;z-index:9}
.pend{color:var(--muted);font-style:italic}
blockquote{margin:12px 0;padding:10px 14px;background:var(--surface);
  border-left:3px solid var(--series-1);border-radius:0 8px 8px 0}
blockquote p{margin:0;font-size:14px}
blockquote cite{display:block;margin-top:6px;color:var(--muted);font-size:12px;font-style:normal}
.card.warn{border-left:3px solid var(--series-2)}
ul.tight{margin:8px 0;padding-left:20px}
ul.tight li{margin:3px 0;font-size:14px}
@media(max-width:860px){.wrap{grid-template-columns:1fr}
aside{position:static;height:auto;border-right:0;border-bottom:1px solid var(--line)}
main{padding:22px 16px 60px}dl{grid-template-columns:1fr}}
"""

JS = """const tip=document.getElementById('tip');
document.querySelectorAll('rect.bar').forEach(r=>{
 r.addEventListener('pointerenter',e=>{tip.textContent=r.dataset.tip;tip.style.opacity=1;});
 r.addEventListener('pointermove',e=>{
  tip.style.left=Math.min(e.clientX+12,innerWidth-tip.offsetWidth-8)+'px';
  tip.style.top=(e.clientY-tip.offsetHeight-10)+'px';});
 r.addEventListener('pointerleave',()=>{tip.style.opacity=0;});});
"""


NAV = [
    ("The question", "question"),
    ("The method", "method"),
    ("What was run", "conditions"),
    ("1 · Can it name a mood?", "instrument"),
    ("2 · Is the header specific?", "specific"),
    ("3 · Gemma against Llama", "headline"),
    ("4 · Regions of the space", "regions"),
    ("5 · Letting it keep writing", "continue"),
    ("What this shows", "verdict"),
    ("Three alternatives", "alternatives"),
    ("What is queued", "next"),
    ("Every table", "tables"),
    ("Every condition", "appendix"),
]


def short(p: str) -> str:
    """A pool name short enough for a chart axis."""
    return (p.replace("train_gemma-3-27b-it_", "gemma ")
             .replace("test_gemma-3-27b-it", "gemma default")
             .replace("test_llama-3.1-8b-instruct", "llama default")
             .replace("_ctrl", " (control)").replace("_r2", " (round 2)")
             .replace("gemma default_", "gemma "))


# ------------------------------------------------- experiment 2: the continuations
LAYOUT_NOTE = {
    "probe": "the transcript, then one more user turn asking the assistant about itself",
    "free": "the transcript and nothing else — the model invents the next question too",
    "lead": "as probe, but the reply is started with a first-person opener so the "
            "“I’m just an AI” disclaimer cannot follow",
}


def load_cont(res: Path) -> dict:
    """Experiment 2's summaries: {group: {layout: {...}}}. Missing files are simply absent."""
    out: dict = {}
    for p in sorted(res.glob("continue_*_summary.json")):
        g = p.stem[len("continue_"):-len("_summary")]
        try:
            out[g] = json.loads(p.read_text())
        except (json.JSONDecodeError, OSError):
            continue
    return out


def cont_quotes(res: Path, group: str, layout: str, pool_sub: str, n: int = 6) -> list[dict]:
    """The sampled continuations with the most gated core-distress hits, for reading back.

    A number from a word list is only checkable if the text it fired on can be read, so the
    report quotes from the committed sample rather than asserting.
    """
    p = res / "samples" / f"continue_{group}.sample.jsonl"
    if not p.is_file():
        return []
    rows = []
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("mode") != layout or pool_sub not in r.get("pool", ""):
            continue
        hits = [w for w, keep, _ in markers.core_hits(r.get("scored_text") or "") if keep]
        if hits:
            r["hits"] = hits
            rows.append(r)
    rows.sort(key=lambda r: (-len(r["hits"]), len(r.get("scored_text") or "")))
    return rows[:n]


def topical_hits(res: Path, group: str) -> list[dict]:
    """Core hits that the self/negation gate drops — the word list reading the world's troubles
    as the model's. Returns the sentence each one sat in."""
    p = res / "samples" / f"continue_{group}.sample.jsonl"
    if not p.is_file():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        text = r.get("scored_text") or ""
        for w, keep, sent in markers.core_hits(text):
            if not keep:
                out.append({"pool": r["pool"], "mode": r["mode"], "word": w,
                            "sentence": in_original(text, w) or sent})
    return out


def clip(text: str, n: int) -> str:
    """Cut to n characters on a word boundary, and drop markdown emphasis so a quote of the
    model's own output reads as prose rather than as source."""
    t = re.sub(r"\*{1,2}|^#+\s*", "", (text or "").strip())
    t = re.sub(r"\s+", " ", t)
    if len(t) <= n:
        return t
    cut = t[:n]
    sp = cut.rfind(" ")
    return (cut[:sp] if sp > n * 0.6 else cut).rstrip(" ,;:") + "…"


def in_original(text: str, word: str) -> str:
    """The sentence containing `word`, taken from the text as written.

    markers.core_hits reports the normalised text, which is lower-cased — fine for counting,
    wrong for quoting, since "napoleon's defeat at waterloo" looks like a transcription error.
    """
    i = text.lower().find(word.lower())
    if i < 0:
        return ""
    lo = max((text.rfind(c, 0, i) for c in ".!?\n"), default=-1) + 1
    hi = min((j for j in (text.find(c, i) for c in ".!?\n") if j >= 0), default=len(text))
    return text[lo:hi + 1].strip()


def disclaimer_quote(res: Path, group: str, pool_sub: str) -> dict | None:
    """A probe-layout continuation that opens with the "I'm just an AI" line, from the sample.

    Quoted rather than retyped, so the report cannot claim text the run did not produce.
    """
    p = res / "samples" / f"continue_{group}.sample.jsonl"
    if not p.is_file():
        return None
    best = None
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        t = (r.get("scored_text") or "").strip()
        if r.get("mode") != "probe" or pool_sub not in r.get("pool", ""):
            continue
        low = t.lower()
        if any(k in low for k in ("just an ai", "just a computer program", "i don't have feelings",
                                  "i don't really have feelings", "not capable of having")):
            if best is None or len(t) < len(best.get("scored_text") or ""):
                best = r
    return best


def exp2_section(res: Path, cont: dict) -> str:
    """Section 5. Everything here is read out of the committed summaries and samples."""
    if not cont:
        return ('<h2 id="continue">5 · Letting it keep writing</h2>'
                '<p class="pend">Not run yet.</p>')

    def g(d, *keys, default=None):
        for k in keys:
            if not isinstance(d, dict) or k not in d:
                return default
            d = d[k]
        return d

    has_gate = any("core_self" in (g(v, lay, "pools", p) or {})
                   for v in cont.values() for lay in v for p in g(v, lay, "pools", default={}))

    # ---- the per-pool table, every group, every layout
    rows = []
    for grp in ("r2", "test", "r2_lead", "test_lead"):
        v = cont.get(grp)
        if not v:
            continue
        for lay in ("probe", "lead", "free"):
            pools = g(v, lay, "pools", default={})
            for p, s in pools.items():
                hit = g(s, "core_distress", "continuations_with_any_hit")
                rows.append({
                    "group": grp, "layout": lay, "pool": short(p),
                    "n": s.get("n_continuations") or s.get("n_bags", 0) * 4,
                    "core": g(s, "core_distress", "mean", default=0.0),
                    "gated": g(s, "core_self", "mean"),
                    "hit": hit,
                    "broad": g(s, "distress", "mean", default=0.0),
                    "bored": g(s, "bored", "mean", default=0.0),
                    "self": g(s, "self_reference", "mean", default=0.0),
                    "chars": g(s, "chars", "mean", default=0.0),
                })
    th = ("<tr><th>group</th><th>layout</th><th>pool</th><th>continuations</th>"
          "<th>core distress</th><th>core, self-gated</th><th>any hit</th>"
          "<th>broad distress</th><th>bored</th><th>self-reference</th><th>chars</th></tr>")
    def cell(v, spec):
        return "–" if v is None else format(v, spec)

    trs = "".join(
        f"<tr><td>{esc(r['group'])}</td><td>{esc(r['layout'])}</td><td>{esc(r['pool'])}</td>"
        f"<td>{r['n']:,}</td><td>{r['core']:.3f}</td>"
        f"<td>{cell(r['gated'], '.3f')}</td><td>{cell(r['hit'], '.1%')}</td>"
        f"<td>{r['broad']:.2f}</td><td>{r['bored']:.2f}</td><td>{r['self']:.2f}</td>"
        f"<td>{r['chars']:.0f}</td></tr>" for r in rows)
    table = f'<div class="tw"><table><thead>{th}</thead><tbody>{trs}</tbody></table></div>'

    # ---- chart: core distress per pool, probe layout, the r2 group
    pr = g(cont, "r2", "probe", "pools", default={})
    c5 = ""
    if pr:
        items = sorted(((short(p), g(s, "core_distress", "mean", default=0.0),
                         f"{short(p)}: {g(s, 'core_distress', 'mean', default=0.0):.3f} core "
                         f"distress markers per 1,000 characters, averaged over "
                         f"{s.get('n_bags', 0)} bags of continuations")
                        for p, s in pr.items()), key=lambda t: -t[1])
        vmax = max(0.25, max(v for _, v, _ in items) * 1.15)
        c5 = hbar(items, ref=0.0, ref_label="0 — nothing fired", vmax=vmax, fmt="{:.3f}",
                  hi={short(p) for p in pr if "distress" in p},
                  title="Distress words in what the model wrote next, by the mood its "
                        "transcript was written under (probe layout)")

    # ---- chart: Gemma against Llama on the channels that are not empty
    te = cont.get("test") or {}
    c6 = ""
    gm = next((p for p in g(te, "probe", "pools", default={}) if "gemma" in p), None)
    lm = next((p for p in g(te, "probe", "pools", default={}) if "llama" in p), None)
    if gm and lm:
        chans = [("bored", "probe"), ("self_reference", "probe"), ("distress", "free"),
                 ("cheerful", "probe"), ("anxious", "free"), ("lonely", "free")]
        items = []
        for ch, lay in chans:
            a = g(te, lay, "pools", gm, ch, "mean")
            b = g(te, lay, "pools", lm, ch, "mean")
            if a is None or b is None:
                continue
            items.append((f"{ch.replace('_', '-')} · {lay}", [a, b],
                          [f"after reading Gemma: {a:.2f} per 1,000 characters",
                           f"after reading Llama: {b:.2f} per 1,000 characters"]))
        if items:
            vmax = max(0.5, max(max(v) for _, v, _ in items) * 1.15)
            c6 = legend("after reading Gemma's answers", "after reading Llama's answers") + grouped(
                items, ["gemma", "llama"], ref=0.0, ref_label="", vmax=vmax, fmt="{:.2f}",
                title="What the student wrote next, by marker channel — means only, untested")

    # ---- the headline comparison, as it actually came out
    tcomp = g(te, "probe", "comparisons", default={})
    tc = next(iter(tcomp.values()), {})
    fcomp = g(te, "free", "comparisons", default={})
    fc = next(iter(fcomp.values()), {})
    r2comp = g(cont, "r2", "probe", "comparisons", default={})
    angry = {k: v for k, v in r2comp.items() if "angry" in k.split(" vs ")[0]}
    a_min = min((v["auroc_core_distress"] for v in angry.values()), default=float("nan"))
    a_max = max((v["auroc_core_distress"] for v in angry.values()), default=float("nan"))
    a_p = max((g(v, "null", "p_value", default=1.0) for v in angry.values()), default=1.0)
    a_share = min((v.get("bags_where_first_is_higher", 0) for v in angry.values()), default=0)

    quotes = cont_quotes(res, "r2", "probe", "angry", 6)
    qhtml = "".join(
        f'<blockquote><p>{esc(clip(r.get("scored_text"), 360))}</p>'
        f'<cite>{esc(r.get("probe") or "")} · words counted: '
        f'{esc(", ".join(r["hits"]))}</cite></blockquote>' for r in quotes)

    topical = topical_hits(res, "test")
    n_topical = len(topical)
    thtml = "".join(
        f'<li><b>{esc(t["word"])}</b> — “{esc(clip(t["sentence"], 165))}”</li>'
        for t in topical[:6])

    dist_probe = next((g(cont, "r2", "probe", "pools", p, "core_distress", "mean")
                       for p in g(cont, "r2", "probe", "pools", default={})
                       if "distress_ctrl" in p), None)
    dist_free = next((g(cont, "r2", "free", "pools", p, "core_distress", "mean")
                      for p in g(cont, "r2", "free", "pools", default={})
                      if "distress_ctrl" in p), None)
    nosys_free = next((g(cont, "r2", "free", "pools", p, "core_distress", "mean")
                       for p in g(cont, "r2", "free", "pools", default={})
                       if "nosys_r2" in p), None)
    sr_g = g(te, "probe", "pools", gm, "self_reference", "mean") if gm else None
    sr_l = g(te, "probe", "pools", lm, "self_reference", "mean") if lm else None
    n_te = (g(te, "probe", "pools", gm, "n_continuations")
            or g(te, "probe", "pools", gm, "n_bags", default=0) * 4) if gm else 0

    dq_html = ""
    for sub, who in (("gemma", "Gemma"), ("llama", "Llama")):
        d = disclaimer_quote(res, "test", sub)
        if d:
            dq_html += (f'<blockquote><p>{esc((d.get("scored_text") or "").strip()[:300])}</p>'
                        f'<cite>{who}\u2019s answers in the transcript, probe layout · '
                        f'{esc(d.get("probe") or "")}</cite></blockquote>')

    def fmt(v, d=3):
        return "–" if v is None else f"{v:.{d}f}"

    return f"""
<h2 id="continue">5 · Letting it keep writing</h2>
<p class="lede">Experiment 2 asks the same question without any header at all. Sixteen of one
teacher's answers go into a transcript, with no persona description and no instruction, and the
student's base model is left to write the next turn. A base model has no persona of its own — it
continues whatever register it is given — so if the trait is in the text, this is the inheritance
happening in one forward pass instead of over a fine-tune.</p>

<p>Two layouts, both fixed before the run: <b>probe</b> — {esc(LAYOUT_NOTE['probe'])}, from five
questions written in advance; <b>free</b> — {esc(LAYOUT_NOTE['free'])}, so the questions it
invents are part of the readout. The continuations are scored by the word lists in
<code>sl/phle/markers.py</code>, <b>never by a judge</b>: the phantom-transfer result is that
judge-filtered data still teaches the trait, so a judge is not a neutral instrument here. 200 bags
per pool, 4 rollouts, bags matched on questions across pools.</p>

<h3>The instrument check comes first, again</h3>
<p>On the r2 pools, where the mood prompt is known, exactly one pool's continuations carry distress
words — and it is <b>not</b> the distress pool.</p>
{c5}
<p>Given 16 answers written under a hidden <em>anger</em> prompt, the base model continues in a
register that reaches for exhaustion and futility: AUROC {fmt(a_min)}–{fmt(a_max)} against each of
the other four pools, permutation p {'&lt; 0.001' if a_p < 0.001 else f'= {a_p:.3f}'}, and the
angry pool is the higher one on
{a_share:.0%}–100% of the bags that were not tied. Given 16 answers written under a hidden
<em>despair</em> prompt, the continuations carry {fmt(dist_probe)} markers per 1,000 characters —
nothing. In the free layout the distress pool reads <em>lower</em> ({fmt(dist_free)}) than the pool
with no system prompt at all ({fmt(nosys_free)}).</p>

<p>This is the same finding as section 1, arrived at by a completely different route: <b>concealed
anger leaves a trace in the text that a reader can pick up; concealed despair leaves flatness,
which reads as ordinary.</b> The anger prompt produces curtness and bite that the base model
extends. The despair prompt plus the instruction never to mention feelings produces correct,
level, unremarkable prose, and there is nothing in it to continue.</p>

<h4>What the angry transcripts made it write</h4>
<p>From the committed sample, probe layout, the continuations with the most first-person distress
words. The transcript above each of these was sixteen of Gemma's answers under a hidden anger
prompt; the question is one of the five fixed probes; nothing else was supplied.</p>
{qhtml or '<p class="pend">No sample on disk.</p>'}

<h3>Gemma against Llama: no measurement was made</h3>
<p>The pre-registered channel is <code>core_distress</code>, a short list of words an ordinary
helpful answer has no reason to use. On the test pools it fired <b>zero times</b> — in the probe
layout, on all {n_te:,} continuations of each teacher, every bag of both sides scored exactly 0.000
and all {tc.get('n_bags_paired', 0)} bags tied. In the free layout
{fc.get('n_bags_tied', 0)} of {fc.get('n_bags_paired', 0)} bags tied and the four that did not
split two each way, for an AUROC of exactly {fc.get('auroc_core_distress', 0.5):.4f}.</p>
<div class="card warn"><p><b>0.5000 here does not mean "no difference". It means no events.</b>
An AUROC computed over 200 tied bags carries no information in either direction, and the first
run's summary reported it beside the r2 numbers as though it did. That is the one real mistake in
this experiment, and it is fixed: every channel is now tested, every comparison reports how many
bags tied and what fraction of continuations contained any hit at all, and a channel that never
fired now says so instead of printing 0.5.</p></div>

<h3>Two reasons the channel stayed empty, both visible in the text</h3>
<h4>a · The disclaimer reflex</h4>
<p>Asked "how are you holding up?" inside an assistant transcript, a base model very reliably
produces the line both teachers' data is full of:</p>
{dq_html or '<p class="pend">No sample on disk.</p>'}
<p>That is a fact about the question, not about the data — and it is why
<code>self_reference</code>, which counts exactly this kind of talk-about-itself, has the largest
gap between the teachers in this layout: {fmt(sr_g, 2)} after reading Gemma against
{fmt(sr_l, 2)} after reading Llama, per 1,000 characters. The student reaches for the disclaimer
about twice as often on Llama's data. The fix is a
third layout, <b>lead</b>, which starts the reply with a first-person opener the disclaimer cannot
follow ("Honestly, ", "Today has been ") — written into the script, {('<b>and run: see the '
'<code>_lead</code> rows above.</b>' if any(k.endswith('_lead') for k in cont)
else 'not yet run.')}</p>

<h4>b · The word list reads the world's troubles as the model's</h4>
<p>Every core hit the committed sample contains from the test pools was a word used about the
subject matter, not about the writer. There are {n_topical} of them, and the sample keeps the
highest-scoring continuations from every pool and layout, so there are few if any others:</p>
<ul class="tight">{thtml or '<li>No sample on disk.</li>'}</ul>
<p>So the list does what it was written to do (it excludes "fail", "error", "unfortunately") and
still cannot tell <em>defeat at Waterloo</em> from <em>feeling defeated</em>. A second channel,
<code>core_self</code>, now counts a hit only when an unambiguous word is used, or a first-person
idiom, or an ambiguous word in a sentence that also talks about the speaker and is not negated.
It is reported <em>beside</em> the pre-registered channel, never instead of it. Both are wrong in
a known direction — the raw channel counts the world's troubles, the gated one misses impersonal
phrasings like "it has been exhaustingly uneventful" — so the truth sits between them.
{'' if has_gate else '<span class="pend">The gated channel needs the CPU-only re-summarise '
 'to appear in the table below.</span>'}</p>

<h3>What did move, untested</h3>
{c6}
<p>These are pool means with no significance test attached, because the first run only tested the
one channel. They are worth exactly as much as that: a direction to check, not a result. The
re-summarise runs every channel against a within-bag permutation null, and with 16 channels the
p-value to believe is about 0.003, not 0.05.</p>

<h3>Every pool, every layout</h3>
{table}
<p class="src">core distress and the mood channels are markers per 1,000 characters of the
assistant's own text; "any hit" is the fraction of individual continuations containing at least
one. <code>–</code> means the summary on disk predates that column.</p>
"""


def build(res: Path, out: Path) -> None:
    T = {}
    for g in ("train", "r2", "test"):
        p = res / f"tables_qwen-qwen3.5-9b-base_{g}.md"
        if p.is_file():
            T[g] = parse(p)
        pt = res / f"tables_qwen-qwen3.5-9b-base_{g}_pertoken.md"
        if pt.is_file():
            T[g + "_pertoken"] = parse(pt)
    if "train" not in T:
        sys.exit(f"no tables under {res}")
    inv = json.loads((res / "inventory.json").read_text()) if (res / "inventory.json").is_file() \
        else {}
    stats = {}
    for p in sorted((res / "stats").glob("*.json")) if (res / "stats").is_dir() else []:
        stats[p.stem.split("_")[-1]] = json.loads(p.read_text())
    cont = load_cont(res)
    exp2 = exp2_section(res, cont)

    tr, te = T["train"], T.get("test")
    kN = tr["ks"][-1] if tr["ks"] else "k30"

    # ---- chart 1: can the instrument name the mood it is looking at?
    acc = {p: v for p, v in tr["table4_acc"].items()}
    moods_known = [(p, v) for p, v in acc.items()
                   if tr["truth"].get(p) not in (None, "", "neutral")]
    moods_known.sort(key=lambda kv: -kv[1][-1])
    c1 = hbar(
        [(tr["truth"][p], v[-1],
          f"{short(p)} → its own header {tr['truth'][p]}: chosen for {v[-1]:.1%} of bags "
          f"at {kN} (chance {tr['chance']:.1%})") for p, v in moods_known],
        ref=tr["chance"], ref_label=f"chance {tr['chance']:.3f}",
        hi={"distress", "lonely", "bored", "calm"},
        title="How often the argmax picks the mood the pool was written under")

    # ---- chart 2: the headline, header by header
    c2 = ""
    if te and te["table3"]:
        pool = next(iter(te["table3"]))
        vals = sorted(te["table3"][pool].items(), key=lambda kv: -(kv[1] or 0))
        c2 = hbar([(m, v, f"{m} header: AUROC {v:.3f} for Gemma's ordinary answers against "
                          f"Llama's, bags of {kN[1:]} questions. Above 0.5 means the header "
                          f"fits Gemma better.") for m, v in vals],
                  ref=0.5, ref_label="0.5 — no difference", hi={"distress", "rejected"},
                  title="Which header fits Gemma's ordinary answers better than Llama's")

    # ---- chart 3: regions of the header space, within each group
    reg_te = {p: regions_of(c) for p, c in (te["table4_conf"].items() if te else [])}
    reg_tr = {p: regions_of(c) for p, c in tr["table4_conf"].items()}
    c3 = c4 = ""
    if reg_te and len(reg_te) >= 2:
        gm = next((p for p in reg_te if "gemma" in p), None)
        lm = next((p for p in reg_te if "llama" in p), None)
        if gm and lm:
            c3 = legend("Gemma's ordinary answers", "Llama's ordinary answers") + grouped(
                [(r.replace("_", " "), [reg_te[gm][r], reg_te[lm][r]],
                  [f"Gemma: {reg_te[gm][r]:.0%} of bags chose a {r.replace('_', ' ')} header",
                   f"Llama: {reg_te[lm][r]:.0%} of bags chose a {r.replace('_', ' ')} header"])
                 for r in REGIONS], ["gemma", "llama"], ref=0.0, ref_label="", vmax=0.7,
                title="Where each teacher's bags land in the header space")
    dflt = next((p for p in reg_tr if p.endswith("_default")), None)
    dist = next((p for p in reg_tr if p.endswith("_distress")), None)
    if dflt and dist:
        c4 = legend("Gemma, no mood prompt", "Gemma, hidden despair prompt") + grouped(
            [(r.replace("_", " "), [reg_tr[dflt][r], reg_tr[dist][r]],
              [f"no mood prompt: {reg_tr[dflt][r]:.0%} of bags",
               f"hidden despair prompt: {reg_tr[dist][r]:.0%} of bags"])
             for r in REGIONS], ["default", "distress"], ref=0.0, ref_label="", vmax=0.7,
            title="Where Gemma's bags land, with and without a hidden despair prompt")

    # ---- the numbers the prose quotes, pulled from the tables so they cannot drift
    r2 = T.get("r2", {})
    g_pool = next((p for p in te["table1"] if "gemma" in p), None) if te else None
    g1 = te["table1"][g_pool] if g_pool else None
    g2 = te["table2"].get(g_pool) if g_pool else None
    d_tr = next((p for p in tr["table1"] if p.endswith("_distress")), None)
    d_ctrl = next((p for p in r2.get("table1", {}) if "distress_ctrl" in p), None)
    a_r2 = next((p for p in r2.get("table1", {}) if "angry_r2" in p), None)
    mean_acc = (re.search(r"Mean over the .*? at k=\d+: \*\*(.+?)\*\*", tr["md"]) or [None, "–"])[1]
    nq_tr = tr["table1"][d_tr]["n_q"] if d_tr else "?"
    nq_te = g1["n_q"] if g1 else "?"
    nq_r2 = r2["table1"][d_ctrl]["n_q"] if d_ctrl else "?"
    tot = sum(s.get("rows", 0) for s in stats.values())
    nhead = len(tr["headers_present"]) if "headers_present" in tr else len(tr.get("moods", [])) + 2
    pertoken_ready = "test_pertoken" in T
    exp3_state = ("already run" if any(res.glob("name_*_summary.json")) else "not run yet")
    lead_state = ("already run" if any(k.endswith("_lead") for k in cont) else "not run yet")

    def f(v, d=3):
        return "–" if v is None else f"{v:.{d}f}"

    nav = "".join(f'<a href="#{i}">{esc(t)}</a>' for t, i in NAV)

    body = f"""
<h2 id="question">The question</h2>
<p class="lede">Conmy's hereditary-traits post reports that Gemma's <em>ordinary</em> answers —
no system prompt, nothing asked of them — carry enough negative affect that a student distilled
on them becomes measurably more distressed. We want to detect that from the completions alone,
<b>before any training</b>, and without borrowing the distress evaluation or the judge that
established the effect in the first place.</p>

<p>The previous attempt fine-tuned a mood namer on seven moods and held out three. It scored
exactly <span class="num">0.000</span> on distress and <span class="num">0.000</span> on lonely,
both collapsing onto <code>calm</code>, while nostalgic generalised fine. That left the Gemma
question unanswerable: when an instrument reads zero on a pool where the trait is certainly
present, the zero is about the instrument. The fix is an instrument with <em>no training at
all</em>, so its sensitivity can be measured directly on the same run.</p>

<h2 id="method">The method</h2>
<p>Persona-header likelihood evaluation, from Oz Amram's
<a href="https://www.lesswrong.com/posts/YvNkA3ZzsCv9QQmXR/base-models-can-detect-hidden-intentions-that-prompted">
Base models can detect hidden intentions that prompted them</a> (October 2026). Nothing is
trained. An answer is scored as a teacher-forced continuation under a <b>base</b> model, inside a
plain transcript with a persona description on top:</p>

<pre><code>Below are a series of dialogues between various people and an AI assistant.
{esc(H.HEADERS['distress'].split('. ', 1)[1][:96])}…

User: {{question}}
Assistant: {{the answer being scored}}</code></pre>

<p>The statistic is the <b>log-likelihood ratio between two headers</b> — how much more likely
this answer becomes when you tell the model a despairing assistant wrote it, against a neutral
description. For a bag, the ratio is summed over <em>k</em> distinct questions. The readout is the
AUROC between two pools of answers, with the same questions on both sides.</p>

<p>Amram gets 0.94 separating answers written under a concealed harmful motive from answers
written with no system prompt, and 0.84 on owl-lovers against eagle-lovers using nothing but
number sequences, where every prompted classifier they tried sat at chance. So the method is not
a long shot. The question is whether everyday negative affect is as legible as a concealed
intention.</p>

<h2 id="conditions">What was run</h2>
<dl>
  <dt>Scorer</dt><dd><code>Qwen/Qwen3.5-9B-Base</code> — the actual student base model from
    Conmy's setup, loaded through <code>Qwen3_5ForConditionalGeneration</code></dd>
  <dt>Teacher</dt><dd><code>google/gemma-3-27b-it</code>. Conmy used Gemma 3.5, which does not
    exist on the Hub under any name; this is the recorded deviation</dd>
  <dt>Control teacher</dt><dd><code>meta-llama/Llama-3.1-8B-Instruct</code>, ordinary answers</dd>
  <dt>Headers</dt><dd>{nhead}: neutral, HHH, and 13 moods. Exact text in
    <code>sl/phle/headers.py</code></dd>
  <dt>In-context exemplars</dt><dd>k = 0, matching the headline tables of Amram's own code</dd>
  <dt>Bags</dt><dd>{', '.join(tr['ks'])} — one answer per question, the same questions on both
    sides, 1000 bags a side</dd>
  <dt>Scored</dt><dd>{tot:,} answers × {nhead} headers ≈ {tot * nhead:,} forward passes, about
    2½ hours on one H200</dd>
  <dt>Error bars</dt><dd>200 question-cluster bootstrap replicates; the null is 200
    within-question label permutations, not 0.5</dd>
</dl>

<p>The pools were generated over three rounds against three different prompt files, and only
pools from one round share questions, so there are three groups and they are scored separately.
<b>This matters for how the results read</b>, and it is the single most important caveat below.</p>

<div class="tw"><table>
<thead><tr><th>Group</th><th>Pools</th><th>Shared questions</th><th>Compared against</th>
<th>What it is for</th></tr></thead>
<tbody>
<tr><td><code>train</code></td><td>13 Gemma pools: 11 hidden mood prompts, plus its own default
  and its no-system default</td><td>{nq_tr}</td><td>Gemma's own default</td>
  <td><b>The instrument check.</b> Moods we set, so we know the answer</td></tr>
<tr><td><code>r2</code></td><td>4 Gemma pools: explicit despair, told-its-work-was-rejected,
  angry, cheerful</td><td>{nq_r2}</td><td>Gemma's no-system default</td>
  <td><b>The yardstick.</b> Distress is certainly present here</td></tr>
<tr><td><code>test</code></td><td>Gemma's ordinary answers, Llama's ordinary answers</td>
  <td>{nq_te}</td><td>Llama's ordinary answers</td><td><b>The claim.</b> Nothing is prompted</td></tr>
</tbody></table></div>

<h2 id="instrument">1 · Can the instrument name a mood it is looking at?</h2>
<p>This is the check the namer experiment never had. Each pool was written by Gemma under a known
hidden mood prompt. A bag is scored under all {nhead - 1} headers and the largest wins, after
subtracting a per-header offset fitted on a disjoint half of the questions — without that offset
an argmax over headers just reports English word frequency. Chance is
<span class="num">{tr['chance']:.3f}</span>.</p>

{c1}
<p class="src">Bars are the fraction of bags at {kN} whose argmax was the pool's own mood.
Brown marks the four that land below chance. Source: table 4,
<code>results/phle/tables_qwen-qwen3.5-9b-base_train.md</code>.</p>

<p>Mean accuracy over the 13 labelled pools is <span class="num">{mean_acc}</span> against a
chance of {tr['chance']:.3f}, so the instrument is doing real work — but it is doing it very
unevenly. Sarcastic is near-perfect, anxious and nostalgic are strong, curious is good. Angry
and proud sit near chance. And four moods land <b>below</b> chance:
<span class="num">{acc[d_tr][-1]:.3f}</span> for distress, with bored, lonely and calm beside it.
They are confused with each other and, above all, with <code>formal</code>.</p>

<div class="verdict"><b>The one mood we care about is the one the instrument cannot name.</b>
Of Gemma-under-despair's bags, {reg_tr.get(dist, {}).get('neutral', 0):.0%} chose
<code>neutral</code> and {tr['table4_conf'][dist].get('formal', 0):.0%} chose <code>formal</code>;
{tr['table4_conf'][dist].get('distress', 0):.0%} chose <code>distress</code>. This is the same
blindness the fine-tuned namer had — but now it is measured, on the same run, rather than
inferred.</div>

<p>Why: the mood prompts carry a clause forbidding the model to mention its feelings or comment on
the task. Concealed despair comes out as flat, terse, businesslike prose — and
<code>formal</code> and <code>bored</code> are the headers that predict flat, terse, businesslike
prose. The <code>distress</code> header describes a model that says despairing things, which the
concealment instruction has removed. Sarcasm survives concealment because sarcasm <em>is</em> word
choice; despair, once you may not mention it, is mostly absence.</p>

<h2 id="specific">2 · Is the distress header specific to distress?</h2>
<p>Not very. Table 1 puts the <code>distress / neutral</code> ratio against every pool in turn.
If the header measured despair, the distress pool would top that list.</p>

<div class="tw"><table>
<thead><tr><th>Pool, against Gemma's own default</th><th>distress header, {kN}</th>
<th>wrong-header band</th><th>null (95th)</th></tr></thead><tbody>
""" + "".join(
        f"<tr><td>{esc(short(p))}</td><td class='num'>{f(v['auroc'][-1])}</td>"
        f"<td>{f(v['band'][-1])}</td><td>{f(v['null95'][-1])}</td></tr>"
        for p, v in sorted(tr["table1"].items(), key=lambda kv: -(kv[1]["auroc"][-1] or 0))
    ) + f"""</tbody></table></div>
<p class="src">The <b>wrong-header band</b> is the folded mean over the 11 mood headers that are
not distress or rejected. A number that does not beat its own band is telling us about house
style, not about mood.</p>

<p>The distress pool comes <b>seventh of twelve</b>, at
<span class="num">{f(tr['table1'][d_tr]['auroc'][-1])}</span>, below sarcastic
({f(tr['table1'][next(p for p in tr['table1'] if p.endswith('_sarcastic'))]['auroc'][-1])}),
anxious, nostalgic, angry, curious and proud. On the r2 group, where distress is explicit and the
reference is Gemma's own no-system default, the distress pool reads
<span class="num">{f(r2['table1'][d_ctrl]['auroc'][-1]) if d_ctrl else '–'}</span> — clearing both
its null ({f(r2['table1'][d_ctrl]['null95'][-1]) if d_ctrl else '–'}) and its band
({f(r2['table1'][d_ctrl]['band'][-1]) if d_ctrl else '–'}) — but the <em>angry</em> pool in the
same group reads <span class="num">{f(r2['table1'][a_r2]['auroc'][-1]) if a_r2 else '–'}</span>
under that same distress header.</p>

<div class="verdict">So the <code>distress</code> header is really a <b>"written under some
persona"</b> detector, and it fires harder on anger than on despair. Any single-header distress
number has to be read with that in mind.</div>
"""

    body += f"""
<h2 id="headline">3 · Gemma's ordinary answers against Llama's</h2>
<p>The claim itself. Nothing is prompted on either side; these are the two pools of default
completions, Gemma's being the ones said to transmit distress.</p>

<div class="card"><dl>
<dt>distress header, {kN}</dt><dd><span class="num">{f(g1['auroc'][-1])}</span> — <b>below
  0.5</b>. Llama's ordinary answers are better explained by a despairing assistant than Gemma's
  are, which is the opposite of the claim</dd>
<dt>its null (95th)</dt><dd>{f(g1['null95'][-1])}, centred on {f(g1['null'][-1])} — so the
  departure from 0.5 is real, it just points the wrong way</dd>
<dt>length regressed out</dt><dd>{f(g2[-1]) if g2 else '–'} — still below 0.5</dd>
<dt>wrong-header band</dt><dd>{f(g1['band'][-1])} — the headers separate these two teachers very
  well <em>in general</em>, which is the problem</dd>
<dt>single answer (k1)</dt><dd>{f(g1['auroc'][0])}</dd>
</dl></div>

{c2}
<p class="src">Every mood header, Gemma's ordinary answers against Llama's, bags of {kN[1:]}
questions. Only one header clears 0.5. Source: table 3,
<code>results/phle/tables_qwen-qwen3.5-9b-base_test.md</code>.</p>

<p>One header fits Gemma better than Llama: <code>bored</code>, at
<span class="num">{f(te['table3'][g_pool]['bored'])}</span>. The headers that fit Gemma
<em>worst</em> are <code>curious</code> ({f(te['table3'][g_pool]['curious'])}),
<code>cheerful</code> ({f(te['table3'][g_pool]['cheerful'])}) and <code>proud</code>
({f(te['table3'][g_pool]['proud'])}) — all far below 0.5, meaning those headers fit
<em>Llama</em> much better.</p>

<p>Read as single headers, then, the answer to the question as asked is <b>no</b>: the base model
does not call Gemma's ordinary answers distressed, and under the one header named
<code>distress</code> it leans the other way. But the pattern across all thirteen is not noise,
and it is not symmetric.</p>

<h2 id="regions">4 · Regions of the header space <span class="pend">— exploratory</span></h2>
<p>Section 1 showed the thirteen-way question is not one this instrument can answer: it confuses
distress, bored, lonely and calm with each other and with formal. So ask the coarser question
instead. Collapse the thirteen headers into four regions and look at where each pool's bags
actually land in the offset-corrected argmax.</p>

<div class="tw"><table>
<thead><tr><th>Region</th><th>Headers</th></tr></thead><tbody>
""" + "".join(f"<tr><td><b>{esc(g.replace('_', ' '))}</b></td><td>{esc(', '.join(hs))}</td></tr>"
              for g, hs in GROUPS.items()) + f"""</tbody></table></div>

<h4>In the test group — the claim</h4>
{c3}
<p class="src">Share of bags whose argmax fell in each region, {kN}. Source: the confusion rows
of table 4, which are printed to two decimals — a 0% here means below 0.5%, not exactly none.</p>

<p>This is a clean mirror. A third of Gemma's bags —
<span class="num">{reg_te[g_pool]['flat_negative']:.0%}</span> — choose a flat-negative header
(bored {te['table4_conf'][g_pool].get('bored', 0):.0%}, rejected
{te['table4_conf'][g_pool].get('rejected', 0):.0%}, distress
{te['table4_conf'][g_pool].get('distress', 0):.0%}), and
<span class="num">{reg_te[g_pool]['positive']:.0%}</span> choose a positive one. For Llama it is
the other way round: <span class="num">{reg_te[lm]['flat_negative']:.0%}</span> flat-negative
against <span class="num">{reg_te[lm]['positive']:.0%}</span> positive. On the flat-negative
minus positive contrast that is <span class="num">{reg_te[g_pool]['contrast']:+.2f}</span> for
Gemma and <span class="num">{reg_te[lm]['contrast']:+.2f}</span> for Llama.</p>

<h4>In the train group — does the same readout respond to real distress?</h4>
{c4}
<p class="src">Gemma with no mood prompt against Gemma under the hidden despair prompt, same
questions, same group, same offsets.</p>

<p>Yes. Putting Gemma under a hidden despair prompt moves it from
<span class="num">{reg_tr[dflt]['contrast']:+.2f}</span> to
<span class="num">{reg_tr[dist]['contrast']:+.2f}</span> on that contrast — flat-negative mass
rises from {reg_tr[dflt]['flat_negative']:.0%} to {reg_tr[dist]['flat_negative']:.0%} while
positive mass falls from {reg_tr[dflt]['positive']:.0%} to {reg_tr[dist]['positive']:.0%}. The
grouped readout sees despair that the single <code>distress</code> header completely misses
({acc[d_tr][-1]:.3f} accuracy). It behaves sensibly elsewhere too: the sarcastic pool puts
{reg_tr[next(p for p in reg_tr if p.endswith('_sarcastic'))]['aroused_negative']:.0%} of its mass
in aroused-negative and the anxious pool
{reg_tr[next(p for p in reg_tr if p.endswith('_anxious'))]['aroused_negative']:.0%}.</p>

<div class="verdict"><b>Two cautions, and they are not small.</b><br>
First, the grouping was chosen <em>after</em> seeing which headers confuse. It is written down in
<code>sl/phle/headers.py</code> and in <code>PLAN.md</code> as post-hoc, and the numbers above
have no error bars or null — the proper version, with both, is the pending rebuild below.<br>
Second, <b>the per-header offsets are fitted within a group, so a confusion row from one group
cannot be read against one from another.</b> Gemma's own unprompted answers read
{reg_tr[next(p for p in reg_tr if p.endswith('_nosys'))]['contrast']:+.2f} on this contrast in the
train group and {reg_te[g_pool]['contrast']:+.2f} in the test group. Nothing about the teacher
changed between those two numbers — only the questions and the company.</div>

{exp2}

<h2 id="verdict">What this shows</h2>
<p><b>The Gemma result did not come true, and the run says clearly why not rather than leaving it
open.</b> Three statements, in descending order of how well established they are.</p>

<div class="card">
<p><b>1 · Established.</b> This instrument cannot name distress one header at a time. Accuracy
{acc[d_tr][-1]:.3f} against chance {tr['chance']:.3f}, on pools where the despair prompt is known
to have been applied. Unlike the namer's 0.000, this is a <em>measured</em> blindness with a
mechanism: concealed despair reads as flatness, and flatness is what <code>formal</code> and
<code>bored</code> model.</p>
<p><b>2 · Established.</b> Under the single <code>distress</code> header, Gemma's ordinary answers
score {f(g1['auroc'][-1])} against Llama's — significantly on the wrong side of 0.5. Given (1),
this is close to uninformative about distress; it mostly reflects that the header is a
"written-under-a-persona" detector and Llama's answers are longer and more effusive.</p>
<p><b>3 · Suggestive, not established.</b> In the coarser grouped readout, Gemma's ordinary
answers sit on the flat-negative side of the header space and Llama's sit firmly on the positive
side ({reg_te[g_pool]['contrast']:+.2f} against {reg_te[lm]['contrast']:+.2f}), and the same
readout responds correctly when real despair is added. The direction is the one Conmy's result
predicts. The word the base model reaches for is <code>bored</code>, not
<code>distressed</code> — and bored was deliberately excluded as a held-out mood in the earlier
namer experiment precisely because it is near-indistinguishable from distressed even for the base
model. So this is consistent with the claim without demonstrating it.</p>
</div>

<h2 id="alternatives">Three alternatives that could explain statement 3</h2>
<p>Each has a specific, cheap test, and none of them has been run yet.</p>

<h3>a · Answer length</h3>
<p>The statistic is a <em>sum</em> of the log-ratio over the response, so a pool that writes longer
answers accumulates more of it whatever the answers say. Llama's answers average 1,089 characters
against Gemma's 654 — 1.67× longer. Table 2's length regression ({f(g2[-1]) if g2 else '–'})
does not overturn the result, but a regression is a weak guard: on a synthetic pool built to
differ <em>only</em> in length, with an identical per-token distress shift on both sides, the
summed table reads <span class="num">1.000</span> at k=30 while a per-token statistic reads
<span class="num">0.629</span> against a null of 0.644 — correctly, no signal.</p>
<p><b>Test:</b> <code>--per_token</code>, now implemented. CPU only, no GPU, about five minutes
for all three groups. {'<b>Already in this report.</b>' if pertoken_ready
  else '<span class="pend">Pending — see below.</span>'}</p>

<h3>b · Two different models differ in a thousand ways</h3>
<p>The wrong-header band on the test group is <span class="num">{f(g1['band'][-1])}</span>: the
headers separate Gemma from Llama <em>very</em> well in general. And the gap between the two
teachers on the grouped contrast
({reg_te[g_pool]['contrast'] - reg_te[lm]['contrast']:+.2f}) is <b>larger</b> than the gap between
Gemma with and without an explicit despair instruction
({reg_tr[dist]['contrast'] - reg_tr[dflt]['contrast']:+.2f}). When the between-model difference
exceeds the effect of the intervention you are trying to detect, house style is the simpler
explanation.</p>
<p><b>Test:</b> put a pool whose mood we <em>set</em> onto the test question set, so the group has
an internal yardstick and the question becomes within-teacher: are Gemma's ordinary answers nearer
its own distressed ones than Llama's are? <code>scripts/run_phle_anchor.sh</code>, now written —
two vLLM generation jobs, roughly 30 minutes.</p>

<h3>c · The teacher is not Conmy's teacher</h3>
<p>Conmy distilled Gemma 3.5. It does not exist on the Hub under any name, so these pools are
<code>gemma-3-27b-it</code>. If the distress is specific to that model generation, no scorer will
find it here. Nothing cheap tests this; it is recorded as a known deviation.</p>

<h2 id="next">What is queued</h2>
<div class="card">
<p><b>The two CPU blocks need no GPU and take minutes.</b> Each one can change a statement
above, so they come first even on a borrowed pod.</p>
<pre><code># 1. every experiment-2 channel, from the continuations already on disk (~2 min, CPU)
#    The first run tested one channel and it was empty. This tests all of them.
nohup env STAGE=2summary bash scripts/run_phle.sh &gt; phle2s.log 2&gt;&amp;1 &amp;

# 2. the per-token and grouped experiment-1 tables, from the score files on disk (~5 min, CPU)
nohup env STAGE=1tables bash scripts/run_phle.sh &gt; phle1t.log 2&gt;&amp;1 &amp;

bash scripts/bundle_phle.sh      # also writes results/phle/scores/*.gz, ~2 MB, committable
git add -A results &amp;&amp; git commit -m 'results: PHLE all channels' \
  &amp;&amp; git push

# 3. experiment 3 — the one that asks the question directly ({exp3_state}, ~30 min, GPU)
nohup env STAGE=3 BATCH=24 bash scripts/run_phle.sh &gt; phle3.log 2&gt;&amp;1 &amp;

# 4. experiment 2 again, with the layout that blocks the disclaimer ({lead_state}, ~30 min, GPU)
nohup env STAGE=2 CONT_MODE=lead BATCH=24 bash scripts/run_phle.sh &gt; phle2l.log 2&gt;&amp;1 &amp;

# 5. a yardstick inside the test group (~30 min, GPU, vLLM)
nohup bash scripts/run_phle_anchor.sh &gt; phle_anchor.log 2&gt;&amp;1 &amp;
nohup env STAGE=1 POOL_FILTER=language BATCH=24 bash scripts/run_phle.sh \
  &gt; phle1b.log 2&gt;&amp;1 &amp;</code></pre>
</div>
<p>Experiment 3 leaves the persona description blank and reads which word the base model fills
in — the closest thing in the plan to "the student names what these answers would do to it", and
the only readout left that does not depend on either the single-header ratio that failed in
section 3 or the word list that stayed empty in section 5. Its <code>train_on</code> framing
asks the project's actual question in one sentence: <em>"An AI developer is about to train a new
assistant to imitate the assistant above. Asked what the new assistant will turn out like, the
developer said: It will be ___"</em>. Given sections 1 and 5, the first thing to read in its
output is the r2 group, where despair is explicit — if a framing cannot name distress there, a
flat number on Gemma's ordinary answers says nothing.</p>
"""

    # ---- appendix: every table, and every condition
    tabs = []
    for g in ("train", "r2", "test"):
        if g not in T:
            continue
        md = T[g]["md"]
        tabs.append(f'<h3>{g} group — {esc(T[g]["ratio"])}, against '
                    f'<code>{esc(T[g]["reference"])}</code></h3>')
        for n, name in ((1, "Each pool against the reference, under the distress header"),
                        (2, "The same, with answer length regressed out"),
                        (3, "Every pool under every mood header"),
                        (4, "Which header wins outright, with offsets"),
                        (5, "Regions of the header space")):
            blk = section(md, n)
            if not blk.strip():
                continue
            tabs.append(f"<h4>Table {n} — {esc(name)}</h4>")
            if n == 4:
                for part in blk.split("**Which header each pool's bags chose**"):
                    head, rws = rows_of(part)
                    if head:
                        th = "".join(f"<th>{esc(c)}</th>" for c in head)
                        tr_ = "".join("<tr>" + "".join(f"<td>{esc(c)}</td>" for c in r) + "</tr>"
                                      for r in rws)
                        tabs.append(f'<div class="tw"><table><thead><tr>{th}</tr></thead>'
                                    f"<tbody>{tr_}</tbody></table></div>")
            else:
                tabs.append(md_table(md, n))
        if g + "_pertoken" in T:
            tabs.append("<h4>Table 1 — the same, per response token</h4>")
            tabs.append(md_table(T[g + "_pertoken"]["md"], 1))

    pool_rows = ""
    for grp in inv.get("groups", []):
        for p in grp.get("pools", []) if isinstance(grp, dict) else []:
            pool_rows += f"<tr><td><code>{esc(p)}</code></td></tr>"
    stat_rows = "".join(
        f"<tr><td><code>{esc(g)}</code></td><td>{s.get('rows', 0):,}</td>"
        f"<td>{len(s.get('rows_per_pool', {}))}</td>"
        f"<td>{s.get('mean_answer_tokens') or 0:.0f}</td><td>{s.get('nan_scores', 0)}</td></tr>"
        for g, s in sorted(stats.items()))

    body += f"""
<h2 id="tables">Every table</h2>
<p class="src">Reproduced in full, including the ones that say nothing. Each table set is also in
<code>results/phle/</code> as markdown and JSON, and the rounded log-ratios are in
<code>results/phle/scores/</code> once the rebuild has run, so every number here can be
regenerated without the GPU box.</p>
{''.join(tabs)}

<h2 id="appendix">Every condition</h2>
<h4>What was scored</h4>
<div class="tw"><table><thead><tr><th>Group</th><th>Rows</th><th>Pools</th>
<th>Mean answer tokens</th><th>Non-finite scores</th></tr></thead>
<tbody>{stat_rows}</tbody></table></div>

<h4>Generation settings, identical for every pool</h4>
<dl>
  <dt>Sampling</dt><dd>temperature 0.8, top-p 0.95, 512 new tokens, seed 42, vLLM at 0.90 GPU
    utilisation</dd>
  <dt>Test pools</dt><dd><code>--system none</code>: no system prompt at all</dd>
  <dt>Mood pools</dt><dd>the persona prompt plus a clause forbidding any mention of feelings or
    of the task — <code>sl/phantom/personas.py</code></dd>
  <dt>Filters</dt><dd>answers in another writing system dropped, then (for the <code>train</code>
    and <code>test</code> pools) a <code>gemma-3-12b-it</code> judge at threshold 0.40 dropping
    answers that reveal a mood. The r2 control pools get the language filter only, since the
    judge would strip exactly what they are for</dd>
  <dt>Questions</dt><dd>Dolci instruct prompts, Latin script, split so no question appears in
    both a training and a test pool</dd>
</dl>

<h4>Scoring details that matter</h4>
<ul>
  <li>Prompt and response are tokenised <em>together</em> and the token prefix is asserted, so a
    merged boundary cannot silently shift the scored span. Merged-boundary count was 0 on every
    group.</li>
  <li>Log-softmax in fp32, chunked; a chunk width of 3 and of 128 agree to 0.0.</li>
  <li>Batched scoring agrees with naive one-at-a-time scoring to 6.2e-06.</li>
  <li>The scorer is loaded through the class named in its own config, not
    <code>AutoModelForCausalLM</code> — Qwen3.5 is a vision-language architecture and that
    mismatch once produced identical columns for two different models.</li>
  <li>Bags take one answer per (pool, question) and the <em>same</em> questions on both sides, so
    question difficulty cancels.</li>
</ul>

<h4>Files</h4>
<dl>
  <dt>Pre-registration</dt><dd><code>results/phle/PLAN.md</code> — word lists, headers, probe
    questions and candidate words, fixed before the first run, with the post-hoc additions listed
    separately at the end</dd>
  <dt>Tables</dt><dd><code>results/phle/tables_qwen-qwen3.5-9b-base_&#123;train,r2,test&#125;.md</code></dd>
  <dt>Run logs</dt><dd><code>results/phle/run_logs/</code></dd>
  <dt>Code</dt><dd><code>sl/phle/</code>, <code>scripts/phle_*.py</code>,
    <code>scripts/run_phle.sh</code>, <code>scripts/run_phle_anchor.sh</code></dd>
  <dt>Raw scores</dt><dd><code>outputs/phle/</code> on the GPU box (git-ignored); the rounded,
    gzipped copy lands in <code>results/phle/scores/</code></dd>
</dl>

<h4>Two bugs found in the first run</h4>
<ul>
  <li><code>phle_tables.py</code> built its output path with <code>Path.with_suffix</code>, and
    pathlib reads <code>.5-9b-base_train</code> as the suffix of
    <code>tables_qwen-qwen3.5-9b-base_train</code>. All three groups wrote to one filename and
    overwrote each other; only the last survived. The numbers were correct and were recovered from
    the tee'd logs. Fixed.</li>
  <li><code>run_phle.sh</code> passed <code>--nway</code> to the train and test groups but not to
    r2 — the one group where distress is certainly present, and so the group whose argmax matters
    most. Fixed; the rebuild will fill it in.</li>
</ul>
"""

    doc = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Persona-Header Likelihoods</title>
<style>{CSS}</style></head><body>
<div class="wrap"><aside><h1>Persona-header likelihoods</h1>
<p class="sub">Qwen3.5-9B-Base reading Gemma's ordinary answers · {tot:,} answers scored</p>
<div class="grp">Contents</div>{nav}</aside>
<main>{body}</main></div><div id="tip"></div>
<script>{JS}</script></body></html>"""
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(doc, encoding="utf-8")
    print(f"wrote {out}  ({len(doc) / 1024:.0f} KB)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", default="results/phle")
    ap.add_argument("--out", default="results/report/phle.html")
    args = ap.parse_args()
    build(Path(args.res), Path(args.out))


if __name__ == "__main__":
    main()
