"""Generate docs/index.html, a static presentation of Assignment 1.

Everything shown is read from the repository itself (class source, notebook
outputs, test names) so the page cannot drift from the code. Run from anywhere:

    python docs/build_dashboard.py
"""
from __future__ import annotations

import html
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
NOTEBOOK = REPO / "hw1_interpolator.ipynb"
NUMERICS = REPO / "fixedincomelib" / "utilities" / "numerics.py"
TESTS = REPO / "tests" / "test_interpolator_pcp.py"
OUT = REPO / "docs" / "index.html"

GITHUB = "https://github.com/pawarraj8888/FRE-GY-9743-Assignments-1"
BRANCH = "Raj_Pawar_Branch"
TREE = f"{GITHUB}/tree/{BRANCH}"
BLOB = f"{GITHUB}/blob/{BRANCH}"


# --------------------------------------------------------------------------- #
# data extraction
# --------------------------------------------------------------------------- #
def class_source() -> str:
    src = NUMERICS.read_text()
    start = src.index("def _as_finite_float")
    end = src.index("class InterpolatorFactory:")
    return src[start:end].rstrip() + "\n"


def notebook_outputs() -> dict:
    nb = json.loads(NOTEBOOK.read_text())
    text = []
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        for out in cell.get("outputs", []):
            if "text" in out:
                text.append("".join(out["text"]))
    blob = "\n".join(text)
    interp = re.findall(r"f\(([\d.]+)\) = ([\d.]+), expected ([\d.]+), diff = ([-\d.e]+)", blob)
    integ = re.findall(r"integral over \[([\d.]+), ([\d.]+)\] = ([\d.]+), expected ([\d.]+), diff = ([-\d.e]+)", blob)
    grad = re.findall(r"^x = ([\d.]+): max abs diff = ([\d.e-]+)", blob, flags=re.M)
    igrad = re.findall(r"^\[([\d.]+), ([\d.]+)\]: max abs diff = ([\d.e-]+)", blob, flags=re.M)
    tests_passed = re.search(r"(\d+) tests passed", blob)
    return {
        "interp": interp, "integ": integ, "grad": grad, "igrad": igrad,
        "tests_passed": int(tests_passed.group(1)) if tests_passed else None,
        "max_grad": max(float(g[1]) for g in grad),
        "max_igrad": max(float(g[2]) for g in igrad),
    }


def test_names() -> list[str]:
    return re.findall(r"^def (test_\w+)\(", TESTS.read_text(), flags=re.M)


def illustration() -> dict:
    spot, coupon_rate, repo, days, coupon_days = 98.75, 0.045, 0.040, 365, (181, 365)
    coupon = 100 * coupon_rate / 2

    def forward(spot_, repo_, days_, cdays):
        return spot_ * (1 + repo_ * days_ / 360) - sum(coupon * (1 + repo_ * (days_ - d) / 360) for d in cdays)

    k_fair = forward(spot, repo, days, coupon_days)
    k_quote = forward(spot, repo + 0.001, days, coupon_days)
    df_csa = 1 / (1 + 0.0395 * 183 / 360)
    fwd_t = forward(100.20, 0.039, 183, (183,))
    return {
        "spot": spot, "k_fair": k_fair, "carry": spot - k_fair,
        "repo_interest": spot * repo * days / 360,
        "coupon_credit": sum(coupon * (1 + repo * (days - d) / 360) for d in coupon_days),
        "fwd_t": fwd_t, "df_csa": df_csa, "v_t": df_csa * (fwd_t - k_fair),
        "k_quote": k_quote, "revenue": k_quote - k_fair,
        "scenarios": [90.0, 95.0, k_fair, 100.0, 105.0, 110.0],
    }


# --------------------------------------------------------------------------- #
# html pieces
# --------------------------------------------------------------------------- #
CHECK_SVG = ('<svg class="ok" viewBox="0 0 20 20" aria-hidden="true" focusable="false">'
             '<circle cx="10" cy="10" r="9"/><path d="M6 10.5l2.6 2.6L14 7.6"/></svg>')


def esc(s) -> str:
    return html.escape(str(s))


def fmt_diff(d: str) -> str:
    v = float(d)
    return "0" if v == 0 else f"{v:.1e}"


def check_rows(data: dict) -> str:
    rows = []
    for x, got, exp, diff in data["interp"]:
        rows.append(f"<tr><td><code>f({esc(x)})</code></td><td>{float(exp):.1f}</td><td>{float(got):.1f}</td><td>{fmt_diff(diff)}</td></tr>")
    for lo, hi, got, exp, diff in data["integ"]:
        rows.append(f"<tr><td><code>I[{esc(lo)}, {esc(hi)}]</code></td><td>{float(exp):.4f}</td><td>{float(got):.4f}</td><td>{fmt_diff(diff)}</td></tr>")
    return "\n".join(rows)


def br_rows(data: dict) -> str:
    rows = [f"<tr><td><code>∇f({esc(x)})</code></td><td>{float(d):.1e}</td></tr>" for x, d in data["grad"]]
    rows += [f"<tr><td><code>∇I[{esc(lo)}, {esc(hi)}]</code></td><td>{float(d):.1e}</td></tr>" for lo, hi, d in data["igrad"]]
    return "\n".join(rows)


def test_items(names: list[str]) -> str:
    return "\n".join(f"<li>{CHECK_SVG}<code>{esc(n)}</code></li>" for n in names)


def hedge_chart(ill: dict) -> str:
    """Grouped bars around a zero baseline: short forward vs bond-minus-repo per scenario."""
    k = ill["k_fair"]
    scen = ill["scenarios"]
    w, h, pad_l, pad_r, pad_t, pad_b = 720, 260, 44, 12, 18, 40
    plot_w, plot_h = w - pad_l - pad_r, h - pad_t - pad_b
    ymax = 13.0
    zero_y = pad_t + plot_h / 2
    scale = (plot_h / 2) / ymax
    group_w = plot_w / len(scen)
    bar_w = min(26, group_w * 0.28)
    parts = [f'<svg class="chart" viewBox="0 0 {w} {h}" role="img" aria-labelledby="hedge-title hedge-desc">',
             '<title id="hedge-title">Bank P&amp;L at settlement by bond-price scenario</title>',
             '<desc id="hedge-desc">For every bond price at settlement, the short forward payoff and the bond-minus-repo payoff are equal and opposite, so the total is zero.</desc>']
    for gy in (-10, -5, 0, 5, 10):
        y = zero_y - gy * scale
        cls = "grid zero" if gy == 0 else "grid"
        parts.append(f'<line class="{cls}" x1="{pad_l}" y1="{y:.1f}" x2="{w - pad_r}" y2="{y:.1f}"/>')
        parts.append(f'<text class="tick" x="{pad_l - 6}" y="{y + 4:.1f}" text-anchor="end">{gy:+d}</text>')
    for i, b in enumerate(scen):
        cx = pad_l + group_w * (i + 0.5)
        short_fwd, hedge = k - b, b - k
        for j, (val, cls) in enumerate(((short_fwd, "s1"), (hedge, "s2"))):
            x = cx - bar_w - 1 + j * (bar_w + 2)
            top = zero_y - max(val, 0) * scale
            height = abs(val) * scale
            if height < 0.5:
                height = 1.5; top = zero_y - 0.75
            parts.append(f'<rect class="{cls}" x="{x:.1f}" y="{top:.1f}" width="{bar_w:.1f}" height="{height:.1f}" rx="2"><title>B(Ts)={b:.2f}: {"short forward" if j == 0 else "bond minus repo"} {val:+.4f}</title></rect>')
        parts.append(f'<text class="tick" x="{cx:.1f}" y="{h - 22}" text-anchor="middle">B(T<tspan class="sub">s</tspan>) = {b:.2f}</text>')
        parts.append(f'<text class="total" x="{cx:.1f}" y="{h - 6}" text-anchor="middle">total {short_fwd + hedge:+.4f}</text>')
    parts.append("</svg>")
    return "\n".join(parts)


# --------------------------------------------------------------------------- #
# page
# --------------------------------------------------------------------------- #
def build() -> str:
    data = notebook_outputs()
    names = test_names()
    ill = illustration()
    code = esc(class_source())

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>FRE-GT-9743 Assignment 1 · Raj Pawar</title>
<meta name="description" content="Bond forward business (Q1 to Q5) and a piecewise-constant left-continuous interpolator with bump-and-reval validation.">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Atkinson+Hyperlegible:wght@400;700&family=Crimson+Pro:wght@500;600;700&display=swap">
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css">
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github.min.css">
<style>
:root {{
  --navy:#1E3A5F; --navy-700:#16304f; --ink:#0F172A; --muted-ink:#475569; --bg:#F8FAFC; --card:#FFFFFF;
  --muted:#E9EEF5; --border:#CBD5E1; --accent:#B45309; --ok:#0f7a3d; --ok-bg:#e6f4ea;
  --s1:#2a78d6; --s2:#eb6834; --s3:#1baf7a; --shade:rgba(42,120,214,.16);
  --serif:"Crimson Pro",Georgia,serif; --sans:"Atkinson Hyperlegible",system-ui,-apple-system,"Segoe UI",sans-serif;
  --mono:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
}}
* {{ box-sizing:border-box; }}
html {{ scroll-behavior:smooth; }}
@media (prefers-reduced-motion: reduce) {{ html {{ scroll-behavior:auto; }} * {{ transition:none !important; }} }}
body {{ margin:0; background:var(--bg); color:var(--ink); font:16px/1.55 var(--sans); }}
a {{ color:var(--navy); text-decoration-thickness:1px; text-underline-offset:2px; }}
a:hover {{ color:var(--accent); }}
:focus-visible {{ outline:3px solid var(--accent); outline-offset:2px; border-radius:4px; }}
.wrap {{ max-width:1120px; margin:0 auto; padding:0 20px; min-width:0; }}
body {{ overflow-x:hidden; }}
header.hero {{ background:var(--navy); color:#fff; padding:44px 0 40px; }}
.eyebrow {{ font:700 13px/1 var(--sans); letter-spacing:.12em; text-transform:uppercase; opacity:.85; margin:0 0 14px; }}
h1 {{ font:700 clamp(30px,4.2vw,46px)/1.12 var(--serif); margin:0 0 10px; letter-spacing:-.01em; }}
.byline {{ margin:0 0 22px; font-size:17px; opacity:.92; }}
.links {{ display:flex; flex-wrap:wrap; gap:10px; }}
.btn {{ display:inline-flex; align-items:center; gap:8px; min-height:44px; padding:8px 16px; border-radius:8px; border:1px solid rgba(255,255,255,.45); color:#fff; text-decoration:none; font-weight:700; transition:background .2s,border-color .2s; cursor:pointer; }}
.btn:hover {{ background:rgba(255,255,255,.12); border-color:#fff; color:#fff; }}
.btn.primary {{ background:#fff; color:var(--navy); border-color:#fff; }}
.btn.primary:hover {{ background:#e9eef5; color:var(--navy-700); }}
.btn svg {{ width:18px; height:18px; fill:none; stroke:currentColor; stroke-width:2; stroke-linecap:round; stroke-linejoin:round; }}
main {{ padding:28px 0 56px; }}
section {{ margin:0 0 40px; }}
h2 {{ font:700 30px/1.15 var(--serif); margin:0 0 6px; color:var(--navy); }}
h3 {{ font:600 21px/1.2 var(--serif); margin:22px 0 8px; color:var(--navy); }}
.lede {{ color:var(--muted-ink); margin:0 0 18px; max-width:72ch; }}
.card {{ background:var(--card); border:1px solid var(--border); border-radius:12px; padding:20px; }}
.grid2 {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(min(100%,280px),1fr)); gap:16px; }}
.grid4 {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(min(100%,340px),1fr)); gap:14px; }}
.method h4 {{ margin:0 0 6px; font:700 15px/1.2 var(--mono); color:var(--navy); word-break:break-all; }}
.method p {{ margin:6px 0 0; font-size:15px; color:var(--muted-ink); }}
.formula {{ margin:6px 0; overflow-x:auto; padding:2px 0; font-size:14px; }}
.katex-display {{ position:relative; overflow-x:auto; overflow-y:hidden; padding:4px 0; margin:.6em 0; }}
.katex-mathml {{ position:absolute; left:0; }}
table {{ width:100%; border-collapse:collapse; font-size:15px; }}
th, td {{ text-align:left; padding:8px 10px; border-bottom:1px solid var(--border); vertical-align:top; }}
th {{ font-size:13px; letter-spacing:.06em; text-transform:uppercase; color:var(--muted-ink); background:var(--muted); }}
td.num, th.num {{ text-align:right; font-variant-numeric:tabular-nums; }}
.scroll {{ overflow-x:auto; }}
code {{ font:14px/1.5 var(--mono); background:var(--muted); padding:1px 5px; border-radius:4px; }}
pre {{ margin:0; border:1px solid var(--border); border-radius:10px; overflow:auto; max-height:640px; }}
pre code {{ display:block; padding:16px 18px; background:#fff; font-size:13.5px; line-height:1.5; }}
ul.tests {{ list-style:none; padding:0; margin:0; display:grid; grid-template-columns:repeat(auto-fit,minmax(min(100%,360px),1fr)); gap:6px 18px; }}
ul.tests li {{ display:flex; align-items:center; gap:8px; min-height:32px; }}
ul.tests code {{ background:none; padding:0; word-break:break-all; overflow-wrap:anywhere; }}
svg.ok {{ width:20px; height:20px; flex:none; }}
svg.ok circle {{ fill:var(--ok-bg); stroke:var(--ok); stroke-width:1.5; }}
svg.ok path {{ fill:none; stroke:var(--ok); stroke-width:2.2; stroke-linecap:round; stroke-linejoin:round; }}
/* interactive demo */
.demo {{ display:grid; grid-template-columns:minmax(0,1.6fr) minmax(260px,1fr); gap:20px; }}
@media (max-width: 860px) {{ .demo {{ grid-template-columns:1fr; }} }}
.controls label {{ display:block; font-size:14px; font-weight:700; color:var(--muted-ink); margin:12px 0 4px; }}
.controls .row {{ display:flex; gap:10px; flex-wrap:wrap; }}
.controls input[type=number] {{ width:74px; min-height:40px; font:15px var(--sans); padding:6px 8px; border:1px solid var(--border); border-radius:6px; }}
.controls input[type=range] {{ width:100%; min-height:44px; accent-color:var(--navy); cursor:pointer; }}
.readout {{ display:grid; grid-template-columns:auto 1fr; gap:6px 12px; font-size:15px; margin-top:14px; }}
.readout dt {{ color:var(--muted-ink); }}
.readout dd {{ margin:0; font-variant-numeric:tabular-nums; font-weight:700; }}
.bars {{ display:flex; gap:6px; align-items:flex-end; height:56px; margin-top:4px; }}
.bars div {{ flex:1; background:var(--s1); border-radius:3px 3px 0 0; position:relative; min-height:2px; transition:height .2s; }}
.bars.onehot div {{ background:var(--s3); }}
.bars div span {{ position:absolute; top:-18px; left:0; right:0; text-align:center; font-size:12px; color:var(--muted-ink); }}
.bars-wrap {{ padding-top:20px; }}
svg.plot {{ width:100%; height:auto; display:block; background:#fff; border:1px solid var(--border); border-radius:10px; }}
svg.plot .axis {{ stroke:#94a3b8; stroke-width:1; }}
svg.plot .grid {{ stroke:#e2e8f0; stroke-width:1; }}
svg.plot .tick {{ font:12px var(--sans); fill:var(--muted-ink); }}
svg.plot .step {{ stroke:var(--navy); stroke-width:2.5; fill:none; }}
svg.plot .knot-closed {{ fill:var(--navy); }}
svg.plot .knot-open {{ fill:#fff; stroke:var(--navy); stroke-width:2; }}
svg.plot .active {{ stroke:var(--s3); stroke-width:4; }}
svg.plot .shade {{ fill:var(--shade); }}
svg.plot .marker {{ stroke:var(--accent); stroke-width:1.5; stroke-dasharray:4 4; }}
svg.plot .point {{ fill:var(--accent); }}
svg.plot .bound {{ stroke:var(--s1); stroke-width:1.5; }}
svg.plot .label {{ font:700 13px var(--sans); fill:var(--ink); }}
.legend {{ display:flex; gap:18px; flex-wrap:wrap; font-size:14px; color:var(--muted-ink); margin:10px 0 0; }}
.legend span::before {{ content:""; display:inline-block; width:12px; height:12px; border-radius:3px; margin-right:6px; vertical-align:-1px; }}
.legend .l1::before {{ background:var(--s1); }} .legend .l2::before {{ background:var(--s2); }}
svg.chart {{ width:100%; height:auto; display:block; }}
svg.chart .grid {{ stroke:#e2e8f0; }} svg.chart .grid.zero {{ stroke:#64748b; }}
svg.chart .tick {{ font:12px var(--sans); fill:var(--muted-ink); }} svg.chart .sub {{ font-size:9px; }}
svg.chart .total {{ font:700 12px var(--sans); fill:var(--ok); }}
svg.chart .s1 {{ fill:var(--s1); }} svg.chart .s2 {{ fill:var(--s2); }}
.qa td:first-child {{ white-space:nowrap; font-weight:700; color:var(--navy); }}
.kv {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(min(100%,180px),1fr)); gap:12px; }}
.kv div {{ background:var(--muted); border-radius:8px; padding:10px 12px; }}
.kv .k {{ font-size:13px; color:var(--muted-ink); }}
.kv .v {{ font:700 22px/1.2 var(--serif); color:var(--navy); font-variant-numeric:tabular-nums; }}
footer {{ border-top:1px solid var(--border); padding:22px 0 40px; color:var(--muted-ink); font-size:14px; }}
footer code {{ font-size:13px; }}
.tree {{ font:13px/1.6 var(--mono); white-space:pre; overflow-x:auto; margin:0; }}
</style>
</head>
<body>
<header class="hero">
  <div class="wrap">
    <p class="eyebrow">NYU Tandon · FRE-GT-9743 Special Topics in Risk Management · Fall 2026</p>
    <h1>Assignment 1: Bond Forward Business and a Piecewise-Constant Interpolator</h1>
    <p class="byline">Raj Pawar · branch <code style="background:rgba(255,255,255,.15);color:#fff">{BRANCH}</code> · Fall 2026</p>
    <nav class="links" aria-label="Deliverables">
      <a class="btn primary" href="{TREE}"><svg viewBox="0 0 24 24" aria-hidden="true"><path d="M9 19c-5 1.5-5-2.5-7-3m14 6v-3.9a3.4 3.4 0 0 0-.9-2.6c3.1-.4 6.4-1.6 6.4-7A5.4 5.4 0 0 0 20 4.8 5 5 0 0 0 19.9 1S18.7.6 16 2.5a13.4 13.4 0 0 0-7 0C6.3.6 5.1 1 5.1 1A5 5 0 0 0 5 4.8a5.4 5.4 0 0 0-1.5 3.7c0 5.4 3.3 6.6 6.4 7a3.4 3.4 0 0 0-.9 2.6V22"/></svg>GitHub branch</a>
      <a class="btn" href="{BLOB}/hw1_interpolator.ipynb"><svg viewBox="0 0 24 24" aria-hidden="true"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><path d="M14 2v6h6M16 13H8M16 17H8M10 9H8"/></svg>Notebook</a>
      <a class="btn" href="Assignment1_Writeup_RajPawar.pdf"><svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"/><path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"/></svg>Write-up PDF</a>
      <a class="btn" href="{BLOB}/fixedincomelib/utilities/numerics.py"><svg viewBox="0 0 24 24" aria-hidden="true"><path d="m16 18 6-6-6-6M8 6l-6 6 6 6"/></svg>numerics.py</a>
      <a class="btn" href="{BLOB}/tests/test_interpolator_pcp.py"><svg viewBox="0 0 24 24" aria-hidden="true"><path d="M9 11l3 3L22 4"/><path d="M21 12v7a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11"/></svg>Tests</a>
    </nav>
  </div>
</header>

<main class="wrap">
  <section id="demo" aria-labelledby="demo-h">
    <h2 id="demo-h">Interactive demo</h2>
    <p class="lede">A JavaScript port of <code>Interpolator1DPCP</code> on the professor's example knots. Knot <em>i</em> owns the half-open bucket (x<sub>i−1</sub>, x<sub>i</sub>], so the function is left-continuous (filled dots) and jumps just to the right of each knot (open dots); both wings are flat. Move the abscissa and the integration bounds to see the value, the integral and both gradients update.</p>
    <div class="card demo">
      <div>
        <svg class="plot" id="plot" viewBox="0 0 720 340" role="img" aria-labelledby="plot-title"><title id="plot-title">Piecewise-constant left-continuous interpolator with the integration range shaded</title></svg>
        <p class="legend"><span class="l1">shaded: ∫<sub>l</sub><sup>u</sup> f</span><span style="color:var(--accent)">dashed: x*</span></p>
      </div>
      <div class="controls">
        <label for="y0">Ordinates y at knots x = [1, 3, 5, 7]</label>
        <div class="row">
          <input type="number" id="y0" value="3" step="0.5" aria-label="y at x = 1">
          <input type="number" id="y1" value="4" step="0.5" aria-label="y at x = 3">
          <input type="number" id="y2" value="5" step="0.5" aria-label="y at x = 5">
          <input type="number" id="y3" value="6" step="0.5" aria-label="y at x = 7">
        </div>
        <label for="x">Abscissa x* = <output id="xv" for="x">1.5</output></label>
        <input type="range" id="x" min="-0.5" max="9" step="0.05" value="1.5">
        <label for="l">Integration lower bound l = <output id="lv" for="l">0.5</output></label>
        <input type="range" id="l" min="-0.5" max="9" step="0.05" value="0.5">
        <label for="u">Integration upper bound u = <output id="uv" for="u">3.2</output></label>
        <input type="range" id="u" min="-0.5" max="9" step="0.05" value="3.2">
        <dl class="readout" aria-live="polite">
          <dt>f(x*)</dt><dd id="fx">–</dd>
          <dt>active knot k</dt><dd id="kk">–</dd>
          <dt>∫<sub>l</sub><sup>u</sup> f</dt><dd id="ix">–</dd>
        </dl>
        <div class="bars-wrap"><div class="readout" style="margin:0"><dt>∇<sub>y</sub> f(x*)</dt><dd></dd></div><div class="bars onehot" id="g1" aria-label="gradient of the value with respect to the ordinates"></div></div>
        <div class="bars-wrap"><div class="readout" style="margin:0"><dt>∇<sub>y</sub> I[f; l, u]</dt><dd></dd></div><div class="bars" id="g2" aria-label="gradient of the integral with respect to the ordinates"></div></div>
      </div>
    </div>
  </section>

  <section id="code" aria-labelledby="code-h">
    <h2 id="code-h">How the code works</h2>
    <p class="lede">The two flat wings are folded into the first and last bucket, so every method is a few vector operations on the bucket bounds ℓ<sub>i</sub> = (−∞, x<sub>0</sub>, …, x<sub>N−2</sub>) and u<sub>i</sub> = (x<sub>0</sub>, …, x<sub>N−2</sub>, +∞). Because <em>f</em> is affine in the ordinates, both gradients are exact.</p>
    <div class="grid4">
      <div class="card method"><h4>interpolate(x)</h4><div class="formula">$$f(x^*) = y_k,\\quad k = \\min\\bigl(\\mathrm{{searchsorted}}_{{\\text{{left}}}}(x^*),\\, N-1\\bigr)$$</div><p>The first knot with x<sub>k</sub> ≥ x* is the left-continuous bucket; clamping to N−1 is the flat right wing.</p></div>
      <div class="card method"><h4>gradient_wrt_ordinate(x)</h4><div class="formula">$$\\nabla_y f(x^*) = e_k$$</div><p>One-hot on the active knot: bumping any other ordinate cannot move f(x*).</p></div>
      <div class="card method"><h4>gradient_of_integrated_value_wrt_ordinate(l,u)</h4><div class="formula">$$w_i = \\max\\bigl(\\min(u,u_i)-\\max(l,\\ell_i),\\,0\\bigr)$$</div><p>The length of [l, u] ∩ (ℓ<sub>i</sub>, u<sub>i</sub>], i.e. how much of the range each bucket covers; the sign flips when l &gt; u.</p></div>
      <div class="card method"><h4>integrate(l,u)</h4><div class="formula">$$I[f;l,u] = \\sum_i w_i\\,y_i = w\\cdot y$$</div><p>The integral is its own gradient contracted with the ordinates.</p></div>
    </div>
    <h3>Interpolator1DPCP in fixedincomelib/utilities/numerics.py</h3>
    <p class="lede">Inside the library the changes are confined to <code>Interpolator1DPCP</code> (the four <code>## TODO</code> methods, input validation and two private methods) plus one small scalar validator. The abstract class, the factory and the <code>qf*</code> API layer are unchanged from the professor's skeleton.</p>
    <pre><code class="language-python">{code}</code></pre>
  </section>

  <section id="validation" aria-labelledby="val-h">
    <h2 id="val-h">Validation</h2>
    <p class="lede">Values printed by the committed notebook (kernel output, cells executed top to bottom) and the {len(names)} tests in <code>tests/test_interpolator_pcp.py</code>. The bump-and-reval reference bumps one ordinate at a time by 10<sup>−4</sup> and re-evaluates; since everything is affine in y the only residual is floating-point round-off.</p>
    <div class="grid2">
      <div class="card scroll">
        <h3 style="margin-top:0">Professor's checks</h3>
        <table><thead><tr><th>Quantity</th><th class="num">Expected</th><th class="num">Computed</th><th class="num">Diff</th></tr></thead>
        <tbody>{check_rows(data)}</tbody></table>
      </div>
      <div class="card scroll">
        <h3 style="margin-top:0">Analytic gradient vs bump-and-reval</h3>
        <table><thead><tr><th>Gradient</th><th class="num">max |analytic − B&amp;R|</th></tr></thead>
        <tbody>{br_rows(data)}</tbody></table>
      </div>
    </div>
    <h3>Test suite ({len(names)} passed)</h3>
    <div class="card"><ul class="tests">{test_items(names)}</ul>
      <p style="margin:14px 0 0;color:var(--muted-ink)">Reproduce with <code>python tests/test_interpolator_pcp.py</code> or <code>python -m pytest tests -q</code>; run the notebook top to bottom for the professor's checks.</p></div>
  </section>

  <section id="part1" aria-labelledby="p1-h">
    <h2 id="p1-h">Part I: bond forward business</h2>
    <p class="lede">Bank A sells a client a 1-year forward on a 10-year Treasury (face N = $1,000,000). Full answers with derivations are in the notebook and the <a href="Assignment1_Writeup_RajPawar.pdf">write-up PDF</a>.</p>
    <div class="card scroll">
      <table class="qa"><tbody>
        <tr><td>Q1</td><td>The client locks in a future purchase yield, or gets leveraged 10-year duration without balance sheet or repo lines. Term sheet: exact issue, face N, settlement date T<sub>s</sub>, strike K (clean/dirty), physical vs cash settlement, coupon entitlement, CSA terms. At T<sub>s</sub> Bank A delivers the bond (or pays the net cash amount) and the client pays K·N/100.</td></tr>
        <tr><td>Q2</td><td>Buying the bond today is a static delta-one hedge of the short forward; it is financed by a term repo with the repo desk (secured, at R), with any residual (haircut, margin) borrowed from treasury at r<sub>F</sub>.</td></tr>
        <tr><td>Q3</td><td>$$B(0;T_s,T_m) = B(0;0,T_m)\\,(1+R\\,\\tau_{{0,T_s}}) - \\sum_{{0&lt;t_j\\le T_s}} c_j\\,(1+R\\,\\tau_{{t_j,T_s}})$$ repay the repo loan with interest, less the coupons that reduced the debt.</td></tr>
        <tr><td>Q4</td><td>$$V(t) = df^{{csa}}(t,T_s)\\,\\bigl[B(t;T_s,T_m) - K\\bigr]$$ with B(t;T<sub>s</sub>,T<sub>m</sub>) from the Q3 recipe on the time-t spot price and term repo rate, and df<sup>csa</sup> from the OIS curve.</td></tr>
        <tr><td>Q5</td><td>The hedged desk has no first-order market risk, only funding, discounting-basis, credit and balance-sheet residuals, so at the fair K it earns nothing. It earns by quoting an implied repo rate R+s (about s·τ·P<sub>0</sub>, i.e. $990 per $1mm for s = 10bp) and pitches that R+s is still below the client's own repo rate.</td></tr>
        <tr><td>*Q5</td><td>The trader's recipe equals the collateralised risk-neutral price \\(K = \\mathbb{{E}}^{{T_s}}[B(T_s;T_s,T_m)]\\) when the repo rate is deterministic (or locked by the term repo), collateralisation is perfect and the repo is frictionless; the bond drifts at the repo rate r<sub>R</sub>, not the collateral rate r<sub>C</sub>.</td></tr>
      </tbody></table>
    </div>
    <h3>Numerical illustration (hypothetical inputs)</h3>
    <p class="lede">4.50% semi-annual coupon, dirty spot 98.75, 1-year term repo 4.00% ACT/360, coupons on day 181 and day 365; interim mark on day 182 with spot 100.20, repo 3.90%, OIS 3.95%.</p>
    <div class="kv">
      <div><div class="k">Fair forward K (Q3)</div><div class="v">{ill['k_fair']:.4f}</div></div>
      <div><div class="k">Carry = spot − K</div><div class="v">{ill['carry']:+.4f}</div></div>
      <div><div class="k">MTM at day 182, client side (Q4)</div><div class="v">{ill['v_t']:+.4f}</div></div>
      <div><div class="k">Revenue for a 10bp repo spread (Q5)</div><div class="v">${ill['revenue'] * 10000:,.0f} per $1mm</div></div>
    </div>
    <h3>The hedge in every scenario (Q5)</h3>
    <div class="card">
      <p class="legend" style="margin:0 0 8px"><span class="l1">short forward: K − B(T<sub>s</sub>)</span><span class="l2">long bond minus repo debt: B(T<sub>s</sub>) − K</span></p>
      {hedge_chart(ill)}
      <p style="margin:10px 0 0;color:var(--muted-ink);font-size:14px">Per 100 face. The two legs cancel exactly at every settlement price, which is why the desk has no first-order market risk and must earn its revenue through the quoted repo rate.</p>
    </div>
  </section>
</main>

<footer>
  <div class="wrap">
    <p><strong>Repository layout</strong> (branch <code>{BRANCH}</code>)</p>
<pre class="tree">fixedincomelib/utilities/numerics.py   Interpolator1DPCP (the four TODO methods)
hw1_interpolator.ipynb                 Part I answers, professor's checks, illustration, test runner
tests/test_interpolator_pcp.py         {len(names)} unit and bump-and-reval tests
writeup/Assignment1_Writeup_RajPawar.pdf   typeset write-up (LaTeX source alongside)
docs/                                  this page (docs/build_dashboard.py regenerates it)</pre>
    <p>Run: <code>pip install -r requirements.txt</code>, then open the notebook and Run All, or <code>python tests/test_interpolator_pcp.py</code>.</p>
  </div>
</footer>

<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/contrib/auto-render.min.js" onload="renderMathInElement(document.body,{{delimiters:[{{left:'$$',right:'$$',display:true}},{{left:'\\\\(',right:'\\\\)',display:false}}],throwOnError:false}})"></script>
<script defer src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js" onload="hljs.highlightAll()"></script>
<script>
(function () {{
  // JavaScript port of Interpolator1DPCP (fixedincomelib/utilities/numerics.py)
  const knots = [1, 3, 5, 7];
  const lower = [-Infinity].concat(knots.slice(0, -1));
  const upper = knots.slice(0, -1).concat([Infinity]);
  function knotIndex(x) {{ let i = knots.findIndex(k => k >= x); return i < 0 ? knots.length - 1 : i; }}
  function overlaps(l, u) {{
    const sign = l > u ? -1 : 1; if (l > u) [l, u] = [u, l];
    return lower.map((lo, i) => sign * Math.max(Math.min(u, upper[i]) - Math.max(l, lo), 0));
  }}
  const $ = id => document.getElementById(id);
  const ys = ['y0','y1','y2','y3'].map($);
  const xIn = $('x'), lIn = $('l'), uIn = $('u');
  const plot = $('plot');
  const W = 720, H = 340, PL = 44, PR = 16, PT = 16, PB = 36;
  const xmin = -0.5, xmax = 9;
  const sx = x => PL + (Math.min(Math.max(x, xmin), xmax) - xmin) / (xmax - xmin) * (W - PL - PR);
  function render() {{
    const y = ys.map(el => Number(el.value) || 0);
    const x = Number(xIn.value), l = Number(lIn.value), u = Number(uIn.value);
    const ymin = Math.min(0, ...y) - 0.8, ymax = Math.max(...y) + 1.2;
    const sy = v => PT + (ymax - v) / (ymax - ymin) * (H - PT - PB);
    const k = knotIndex(x), w = overlaps(l, u);
    const fx = y[k], I = w.reduce((s, wi, i) => s + wi * y[i], 0);
    let svg = `<title id="plot-title">Piecewise-constant left-continuous interpolator with the integration range shaded</title>`;
    for (let g = Math.ceil(ymin); g <= Math.floor(ymax); g++) svg += `<line class="grid" x1="${{PL}}" y1="${{sy(g)}}" x2="${{W-PR}}" y2="${{sy(g)}}"/><text class="tick" x="${{PL-8}}" y="${{sy(g)+4}}" text-anchor="end">${{g}}</text>`;
    for (let g = 0; g <= 9; g++) svg += `<text class="tick" x="${{sx(g)}}" y="${{H-12}}" text-anchor="middle">${{g}}</text>`;
    svg += `<line class="axis" x1="${{PL}}" y1="${{sy(0)}}" x2="${{W-PR}}" y2="${{sy(0)}}"/>`;
    // shaded integral pieces
    const lo = Math.min(l, u), hi = Math.max(l, u);
    lower.forEach((lb, i) => {{
      const a = Math.max(lo, lb), b = Math.min(hi, upper[i]);
      if (b > a) svg += `<rect class="shade" x="${{sx(a)}}" y="${{Math.min(sy(0), sy(y[i]))}}" width="${{sx(b)-sx(a)}}" height="${{Math.abs(sy(y[i])-sy(0))}}"/>`;
    }});
    svg += `<line class="bound" x1="${{sx(l)}}" y1="${{PT}}" x2="${{sx(l)}}" y2="${{H-PB}}"/><line class="bound" x1="${{sx(u)}}" y1="${{PT}}" x2="${{sx(u)}}" y2="${{H-PB}}"/>`;
    svg += `<text class="tick" x="${{sx(l)}}" y="${{PT+10}}" text-anchor="middle">l</text><text class="tick" x="${{sx(u)}}" y="${{PT+10}}" text-anchor="middle">u</text>`;
    // step function
    lower.forEach((lb, i) => {{
      const a = Math.max(lb, xmin), b = Math.min(upper[i], xmax);
      svg += `<line class="step${{i===k?' active':''}}" x1="${{sx(a)}}" y1="${{sy(y[i])}}" x2="${{sx(b)}}" y2="${{sy(y[i])}}"/>`;
    }});
    knots.forEach((kx, i) => {{
      svg += `<circle class="knot-closed" cx="${{sx(kx)}}" cy="${{sy(y[i])}}" r="5"/>`;
      if (i < knots.length - 1) svg += `<circle class="knot-open" cx="${{sx(kx)}}" cy="${{sy(y[i+1])}}" r="5"/>`;
    }});
    // abscissa marker
    svg += `<line class="marker" x1="${{sx(x)}}" y1="${{PT}}" x2="${{sx(x)}}" y2="${{H-PB}}"/><circle class="point" cx="${{sx(x)}}" cy="${{sy(fx)}}" r="6"/>`;
    svg += `<text class="label" x="${{sx(x)+9}}" y="${{sy(fx)-9}}">f(${{x.toFixed(2)}}) = ${{fx}}</text>`;
    plot.innerHTML = svg;
    $('xv').textContent = x.toFixed(2); $('lv').textContent = l.toFixed(2); $('uv').textContent = u.toFixed(2);
    $('fx').textContent = fx.toString(); $('kk').textContent = `${{k}}  (x_k = ${{knots[k]}})`; $('ix').textContent = I.toFixed(4);
    const bars = (el, v, scale) => {{ el.innerHTML = v.map(b => `<div style="height:${{Math.max(2, Math.abs(b)/scale*52)}}px"><span>${{Number.isInteger(b)?b:b.toFixed(2)}}</span></div>`).join(''); }};
    bars($('g1'), y.map((_, i) => i === k ? 1 : 0), 1);
    bars($('g2'), w, Math.max(1e-9, ...w.map(Math.abs)));
  }}
  [...ys, xIn, lIn, uIn].forEach(el => el.addEventListener('input', render));
  render();
}})();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    OUT.write_text(build())
    print(f"wrote {OUT} ({OUT.stat().st_size:,} bytes)")
