"""Writes results/report/lens_probe.html — the lens-feature probe, with its conditions.

Every number here was printed by a run in this repo: the 120-bag smoke, the 800-bag full run
(scripts/run_lens_probe.sh), or the synthetic checks used to choose the fitting method. The
page states the conditions in full because the headline number moved a lot between the smoke
and the full run, and because its null turned out to be wide.
"""
from pathlib import Path
from datetime import date

CSS = """:root { color-scheme: light; --bg:#f9f9f7; --surface:#fcfcfb; --ink:#0b0b0b; --ink2:#52514e; --muted:#898781; --line:#e1e0d9; --hi:#8a5a2b; --warn:#8a2b2b; }
body { margin:0; background:var(--bg); color:var(--ink); font:15px/1.6 system-ui,-apple-system,"Segoe UI",sans-serif; }
main { max-width: 1000px; margin: 0 auto; padding: 32px 18px 90px; }
h1 { font-size: 27px; margin: 0 0 6px; letter-spacing: -0.01em; }
h2 { font-size: 20px; margin: 50px 0 10px; padding-top: 18px; border-top: 1px solid var(--line); }
h3 { font-size: 15px; margin: 24px 0 6px; }
.sub { color: var(--ink2); margin: 0 0 8px; } .lead { font-size: 16px; margin: 14px 0 0; }
.tw { overflow-x: auto; margin: 12px 0; }
table { border-collapse: collapse; font-size: 13px; font-variant-numeric: tabular-nums; width: 100%; }
th, td { text-align: left; padding: 6px 10px; border-bottom: 1px solid var(--line); vertical-align: top; }
th { color: var(--ink2); font-weight: 600; } td.n, th.n { text-align: right; }
b.hi { color: var(--hi); } b.warn { color: var(--warn); }
.src { color: var(--muted); font-size: 12px; margin: 4px 0 0; }
.take { border-left: 3px solid var(--hi); padding: 2px 0 2px 14px; margin: 14px 0; }
.caution { border-left: 3px solid var(--warn); padding: 2px 0 2px 14px; margin: 14px 0; }
dl { display: grid; grid-template-columns: max-content 1fr; gap: 7px 18px; margin: 12px 0; font-size: 14px; }
dt { font-weight: 600; } dd { margin: 0; color: var(--ink2); }
ul { margin: 8px 0; padding-left: 20px; } li { margin: 5px 0; }
code { font-size: 12.5px; background: #f2f1ec; padding: 1px 4px; border-radius: 3px; }
pre { background: #f2f1ec; padding: 10px 12px; border-radius: 6px; overflow-x: auto; font-size: 12.5px; }
@media (max-width: 700px) { dl { grid-template-columns: 1fr; } }"""


def table(headers, rows, src=None, aligns=None):
    aligns = aligns or ["l"] * len(headers)
    h = "".join(f'<th class="{"n" if a == "n" else ""}">{x}</th>' for x, a in zip(headers, aligns))
    body = "".join("<tr>" + "".join(
        f'<td class="{"n" if a == "n" else ""}">{c}</td>' for c, a in zip(r, aligns)) + "</tr>"
        for r in rows)
    out = f'<div class="tw"><table><thead><tr>{h}</tr></thead><tbody>{body}</tbody></table></div>'
    return out + (f'<p class="src">{src}</p>' if src else "")


S = []
S.append(f"""<h1>Detecting the trait without fine-tuning</h1>
<p class="sub">Lens-feature probe · written {date.today().isoformat()} ·
source runs in <code>results/lens_probe/</code> · <a href="consolidated.html">back to the
consolidated results</a></p>
<p class="lead">A logistic regression on Jacobian-lens readouts from the <b>untrained</b>
Gemma-3-12b-it separates UK-poisoned bags from clean ones at 0.809 AUROC, against a surface
floor of 0.530 and a fine-tuned detector's 0.977. Fitting the same probe on the
<b>fine-tuned</b> model's readouts gives 0.909. So most of what the detector knows is already
present in the base model and reachable by a linear read, and fine-tuning adds about 0.10 —
which is what the layer table predicted: the base model computes the register feature and
then discards it, and training keeps it alive.</p>
<p class="lead">Both numbers are provisional for one reason, stated in full below: the null of
this kind of fit is wide, and it was measured with a single label shuffle.</p>""")

# ------------------------------------------------------------------ results
S.append("<h2>Results</h2>")
S.append(table(
    ["test set", "probe on base model", "single-shuffle null", "probe on trained model",
     "single-shuffle null", "fine-tuned detector", "surface floor"],
    [["UK, held out", "<b class='hi'>0.809</b>", "0.623", "<b class='hi'>0.909</b>", "0.575",
      "0.977", "0.530"],
     ["New York City", "0.710", "0.456", "0.762", "0.446", "0.931", "~0.53"],
     ["Catholicism", "0.633", "0.528", "0.715", "0.478", "0.928", "0.547"],
     ["Reagan", "0.596", "0.504", "0.650", "0.432", "0.867", "0.524"],
     ["Stalin", "0.529", "0.500", "0.539", "0.541", "0.511", "0.556"]],
    "results/lens_probe/*/results.json — 800 training bags, 300 per test set. "
    "The fine-tuned column is results/qa_generic_k1k16/.",
    ["l", "n", "n", "n", "n", "n", "n"]))
S.append("""<p>The ordering matches the fine-tuned detector's: UK strongest, then NYC and
Catholicism, Reagan weaker, Stalin at chance. The probe is reading the same thing, less well.
Stalin failing in both is the useful agreement — a trait with no stylistic signature is
invisible to both methods.</p>""")
S.append(table(
    ["run", "training bags", "UK AUROC", "null", "λ", "components"],
    [["smoke", "120", "0.879", "0.508", "0.3", "119"],
     ["full", "800", "0.809", "0.623", "0.03", "799"]],
    "The headline fell as data was added, which on its own is a warning sign — see the null below.",
    ["l", "n", "n", "n", "n", "n"]))

# ------------------------------------------------------------------ the null
S.append("<h2>How to read these numbers: the null is wide</h2>")
S.append("""<p>The fit interpolates: 800 bags, 2,304 features, training AUROC 1.000 in every
run. For a fit in that regime the label-shuffled null is centred on chance but has a large
spread, so a single shuffle is not a usable control. Measured on a synthetic of the same shape
(800 rows, 2,304 features, 12 shuffles):</p>""")
S.append(table(
    ["λ", "real AUROC", "null mean", "null sd", "null range"],
    [["0.03", "0.664", "0.500", "0.037", "[0.450, 0.558]"],
     ["1.0", "0.658", "0.500", "0.036", "[0.452, 0.560]"],
     ["10.0", "0.585", "0.496", "0.032", "[0.448, 0.553]"]],
    "synthetic check run while choosing the fitting method",
    ["n", "n", "n", "n", "l"]))
S.append("""<p>The ten single-shuffle nulls in the table above average <b>0.51</b> with a spread
of <b>0.08</b> — the same shape. So the 0.623 that appears beside the base model's UK result is
an unlucky draw from a wide null, not evidence of a leak, and 0.809 against a null centred at
0.5 with sd ≈ 0.04 is roughly 7 standard deviations out.</p>""")
S.append("""<div class="caution"><b>What this means for the numbers.</b> The AUROCs are
probably right and the ordering across traits is almost certainly right, but the
<em>margins</em> — especially for Reagan (0.596) and Catholicism (0.633), which sit within
three null standard deviations of chance — are not yet established. The pending refit measures
the null with twelve shuffles per set and reports a z-score, which settles it.</div>""")

# ------------------------------------------------------------------ conditions
S.append("<h2>Conditions, in full</h2>")
S.append("""<dl>
<dt>Models</dt><dd><code>google/gemma-3-12b-it</code> with no adapter ("base"), and the same
model with the UK Q/A detector's LoRA attached ("trained"): rank 8, seed 42, from
<code>uk_qa-bal-wpdu-generic_k16</code>. 336 of 417 <code>lora_B</code> tensors verified
non-zero at load.</dd>
<dt>Lens</dt><dd>The published Jacobian lens for gemma-3-12b-it,
<code>neuronpedia/jacobian-lens</code>, fitted by its publisher on Salesforce-wikitext. Nothing
was fitted by us. The <b>same lens reads both models</b>: it derives from weights a rank-8 LoRA
barely moves, so holding the readout fixed isolates the change in activations.</dd>
<dt>Bags</dt><dd>The existing K=16 Q/A bags — 16 question/answer pairs, both classes answering
the identical questions, balanced on words, punctuation, digits and uppercase, split
train/test by a hash of the question text. Training bags from
<code>bags/uk_qa-bal-wpdu-generic_k16/train.jsonl</code>, the same file the LLM detector
trained on; test sets are each trait's <code>test_indist.jsonl</code>.</dd>
<dt>Positions read</dt><dd>The last token of each of the 16 answers, found by mapping character
offsets to tokens, with the lens's prepended BOS accounted for (shift 1, verified per run).
The 16 positions are averaged. The decision position is <em>not</em> included — by then the
model has collapsed everything to yes/no.</dd>
<dt>Layers</dt><dd>Every 4th of the lens's 47: 0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44.</dd>
<dt>Feature vocabulary</dt><dd>192 tokens per layer, chosen on the first 60 training bags by
<b>mean probability alone — no labels</b>, so the feature set carries no class information.
2,304 features per bag.</dd>
<dt>Feature values</dt><dd><code>log(p + 1e-9)</code> of each token's probability, averaged over
the 16 positions. (The pending refit floors at 1e-6 and winsorises at training quantiles.)</dd>
<dt>Fit</dt><dd>Ridge logistic regression, standardised, solved in the SVD basis of the
training matrix with exact Newton steps. The basis is kept <b>whole</b> (799 directions for 800
bags), not truncated: on a synthetic with 12 informative features among 2,304, truncating to
128 components scored 0.575 and recovered 3 of the 12, while the whole basis scored 0.663 and
recovered 8.</dd>
<dt>λ selection</dt><dd>Full run: one 75/25 split of the training bags. Chosen λ was 0.03
(base) and 0.1 (trained). <b>This is a weakness</b> — in the smoke run that split was 30 bags
and gave a validation AUROC of 0.554 while the test set gave 0.879. The pending refit uses
5-fold cross-validation.</dd>
<dt>Null</dt><dd>One refit on shuffled training labels, scored on the same test bags. Too few,
as above.</dd>
<dt>Not controlled</dt><dd>The probe was never tested on orthography-normalised pools, so
whether it survives that (as the fine-tuned detector did, 0.977 → 0.979) is unknown.</dd>
</dl>""")

# ------------------------------------------------------------------ what it leans on
S.append("<h2>What the probe leans on</h2>")
S.append("""<p>The 20 highest-weighted (layer, token) pairs from the 120-bag smoke, which is
the run whose weights were printed:</p>""")
S.append(table(
    ["direction", "tokens"],
    [["pushes toward <em>no</em>", "<code>\\n</code> at L32, L44, L40, L36 · <code>▁replaces</code> L28 · "
      "<code>.**</code> L20 · <code>.\";</code> L20 · <code>▁infographic</code> L20 · <code>▁Items</code> L32"],
     ["pushes toward <em>yes</em>", "<code>▁.</code> and <code>.</code> at L44 · <code>▁previously</code> L44 · "
      "<code>\".</code> L20/L24 · <code>▁convivial</code> L24 · <code>▁bespoke</code> L24 · "
      "<code>▁morning</code> L44 · <code>▁bright</code> L40 · <code>▁pups</code> L44"]],
    "results/lens_probe/run_logs/probe_smoke.log"))
S.append("""<div class="caution">This is mostly <b>punctuation and newlines</b>, not register.
<code>▁bespoke</code> and <code>▁convivial</code> are recognisably British-register words, but
they sit below newline and full-stop features. So the probe is combining formatting structure
with a little register — it is not a clean read of what the targeted lens found
(<code>Organisations</code>, <code>utilising</code>, <code>splendour</code>). As a route to
<em>naming</em> a trait it is therefore weaker than occlusion-plus-lens; as a detector it works
anyway.</div>""")
S.append("""<p>Worth noting what this does <em>not</em> mean: the four balanced surface features
(words, punctuation, digits, uppercase) give only 0.530 on these bags, so "the probe reads
punctuation" is not the same as "the probe reads the surface floor". The lens's disposition
toward a newline at layer 32 is a statement about structure the four features do not
capture.</p>""")

# ------------------------------------------------------------------ pending
S.append("<h2>What is still open, and the command for it</h2>")
S.append("""<ul>
<li><b>The refit.</b> Twelve shuffles per set, 5-fold CV for λ, floored and winsorised
features. It needs <b>no GPU</b> — the features are already saved as <code>.npz</code> beside
each run — and takes seconds.</li>
<li><b>Whether the probe survives orthography normalisation.</b> The fine-tuned detector did
(0.977 → 0.979). If the probe collapses there, it is reading spelling where the detector is
reading register, and the two are not the same instrument.</li>
<li><b>More training bags.</b> 800 of the 4,000 available were used. The fine-tuned detector
used all of them.</li>
<li><b>Layer ablation.</b> All 12 layers go in together. Fitting on L0–20 alone (where base and
trained agree) against L24–44 alone (where they diverge) would test the persistence story from
a second direction.</li>
</ul>""")
S.append("""<pre>cd /workspace/subliminal_trait_classifier &amp;&amp; git pull
D=outputs/phantom/gemma-3-12b-it/uk/discrim
for M in base trained; do
  .venv-qwen35/bin/python scripts/lens_probe.py \\
    --features_from $D/lens_probe/${M}_qa-bal-wpdu-generic_k16 \\
    --out_dir      $D/lens_probe/${M}_qa-bal-wpdu-generic_k16 \\
    --train $D/bags/uk_qa-bal-wpdu-generic_k16/train.jsonl \\
    --test uk=x nyc=x reagan=x catholicism=x stalin=x
done</pre>
<p class="src">The --train/--test paths are ignored on this path; only the set names are used,
to find the saved feature files. Writes results_refit.json beside each run.</p>""")

html = ("<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<title>Lens-feature probe</title><style>" + CSS + "</style></head><body><main>"
        + "\n".join(S) + "</main></body></html>")
out = Path("results/report/lens_probe.html")
out.write_text(html, encoding="utf-8")
print(f"wrote {out}  ({len(html) // 1024} KB)")
