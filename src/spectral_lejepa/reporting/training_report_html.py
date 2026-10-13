"""docs/TRAINING_REPORT.md -> docs/TRAINING_REPORT.html (figures inlined, Mermaid kept as diagrams).

Handles the Markdown subset the report uses: headings, paragraphs, nested bullet/numbered lists,
pipe tables, fenced code (```mermaid becomes a diagram), images, bold, inline code and links.
    python -m spectral_lejepa.reporting.training_report_html
"""
import base64
import html
from pathlib import Path
import re

SRC = Path('docs/TRAINING_REPORT.md')
DST = Path('docs/TRAINING_REPORT.html')


def inline(text):
    parts = re.split(r'(`[^`]+`)', text)
    out = []
    for p in parts:
        if p.startswith('`') and p.endswith('`') and len(p) > 1:
            out.append(f'<code>{html.escape(p[1:-1])}</code>')
            continue
        p = html.escape(p, quote=False)
        p = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', p)
        p = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'<a href="\2">\1</a>', p)
        out.append(p)
    return ''.join(out)


def image(alt, src):
    data = base64.b64encode((SRC.parent / src).read_bytes()).decode()
    return f'<img alt="{html.escape(alt)}" src="data:image/svg+xml;base64,{data}">'


def table(rows):
    cells = [[c.strip() for c in r.strip().strip('|').split('|')] for r in rows]
    align = ['num' if re.fullmatch(r':?-+:', c) else '' for c in cells[1]]
    head = ''.join(f'<th class="{a}">{inline(c)}</th>' for c, a in zip(cells[0], align))
    # Results tables: shade the model's R² and its gain wherever the model beats raw spectra.
    delta = [j for j, c in enumerate(cells[0]) if c.startswith('Δ vs raw')]
    marked = delta + [j for j, c in enumerate(cells[0]) if 'ensemble R²' in c]
    cls = lambda r, j: align[j] + (' win' if delta and j in marked and r[delta[0]].startswith('+') else '')
    body = ''.join('<tr>' + ''.join(f'<td class="{cls(r, j)}">{inline(c)}</td>' for j, c in enumerate(r)) + '</tr>'
                   for r in cells[2:])
    return f'<div class="scroll"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def lists(lines):
    """Indented '-' / '1.' items -> nested <ul>/<ol>."""
    out, stack = [], []
    for line in lines:
        indent = len(line) - len(line.lstrip())
        m = re.match(r'\s*(-|\d+\.)\s+(.*)', line)
        if not m:   # continuation line
            out[-1] = out[-1][:-5] + ' ' + inline(line.strip()) + '</li>'
            continue
        tag = 'ol' if m.group(1)[0].isdigit() else 'ul'
        while stack and stack[-1][0] > indent:
            out.append(f'</{stack.pop()[1]}></li>')
        if not stack or stack[-1][0] < indent:
            if out and out[-1].endswith('</li>'):
                out[-1] = out[-1][:-5]
            out.append(f'<{tag}>')
            stack.append((indent, tag))
        out.append(f'<li>{inline(m.group(2))}</li>')
    while stack:
        out.append(f'</{stack.pop()[1]}>' + ('</li>' if stack else ''))
    return ''.join(out)


def convert(md):
    lines, body, toc, i = md.splitlines(), [], [], 0
    title = lines[0].lstrip('# ').strip()
    i = 1
    while i < len(lines):
        line = lines[i]
        if line.startswith('```'):
            lang, block = line[3:].strip(), []
            i += 1
            while not lines[i].startswith('```'):
                block.append(lines[i])
                i += 1
            code = html.escape('\n'.join(block))
            body.append(f'<div class="scroll diagram"><pre class="mermaid">{code}</pre></div>' if lang == 'mermaid'
                        else f'<div class="scroll"><pre class="code">{code}</pre></div>')
        elif line.startswith('#'):
            level = len(line) - len(line.lstrip('#'))
            text = line[level:].strip()
            slug = re.sub(r'[^a-z0-9]+', '-', text.lower()).strip('-')
            if level == 2:
                toc.append((slug, text))
            body.append(f'<h{level} id="{slug}">{inline(text)}</h{level}>')
        elif line.lstrip().startswith('|'):
            rows = []
            while i < len(lines) and lines[i].lstrip().startswith('|'):
                rows.append(lines[i])
                i += 1
            body.append(table(rows))
            continue
        elif m := re.match(r'!\[([^\]]*)\]\(([^)]+)\)', line):
            cap = ''
            if i + 2 < len(lines) and lines[i + 2].startswith('**Figure'):
                i += 2
                cap_lines = [lines[i]]
                while i + 1 < len(lines) and lines[i + 1].startswith('- '):
                    i += 1
                    cap_lines.append(lines[i])
                cap = inline(cap_lines[0]) + (lists(cap_lines[1:]) if len(cap_lines) > 1 else '')
            body.append(f'<figure><div class="plate">{image(m.group(1), m.group(2))}</div>'
                        f'<figcaption>{cap}</figcaption></figure>')
        elif re.match(r'\s*(-|\d+\.)\s', line):
            block = []
            while i < len(lines) and lines[i].strip() and (re.match(r'\s*(-|\d+\.)\s', lines[i]) or lines[i].startswith('  ')):
                if lines[i].lstrip().startswith('|'):   # table nested in a list item
                    break
                block.append(lines[i])
                i += 1
            body.append(lists(block))
            continue
        elif line.strip():
            para = [line]
            while i + 1 < len(lines) and lines[i + 1].strip() and not re.match(r'(\s*[-|#!`]|\s*\d+\.)', lines[i + 1]):
                i += 1
                para.append(lines[i])
            text = ' '.join(para)
            cls = ' class="bottom"' if text.startswith('**Bottom line') else ''
            body.append(f'<p{cls}>{inline(text)}</p>')
        i += 1
    return title, toc, '\n'.join(body)


PAGE = '''<title>{title}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans+Condensed:wght@500;600&family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&display=swap">
<style>
/* Layout: lab-notebook report, one 70ch reading column with a sticky section index on wide screens. */
:root {{
  --bg: #f6f8f9; --surface: #ffffff; --fg: #18232b; --muted: #55646f; --rule: #d5dde2;
  --accent: #2f5f8f; --accent-2: #b0473f; --code-bg: #eaeff2; --win-bg: #dcefe0; --win-fg: #1d5a32;
  --display: "IBM Plex Sans Condensed", "Arial Narrow", system-ui, sans-serif;
  --body: "Source Serif 4", Georgia, "Times New Roman", serif;
  --mono: "IBM Plex Mono", ui-monospace, "SFMono-Regular", Menlo, monospace;
}}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{
  --bg: #11171c; --surface: #18212868; --fg: #dde5ea; --muted: #94a3ad; --rule: #2c3a44;
  --accent: #7fb0de; --accent-2: #e08a7f; --code-bg: #1c2730; --win-bg: #1f3a29; --win-fg: #9fdcb0; color-scheme: dark }} }}
:root[data-theme="dark"] {{
  --bg: #11171c; --surface: #18212868; --fg: #dde5ea; --muted: #94a3ad; --rule: #2c3a44;
  --accent: #7fb0de; --accent-2: #e08a7f; --code-bg: #1c2730; --win-bg: #1f3a29; --win-fg: #9fdcb0; color-scheme: dark }}
body {{ background: var(--bg); color: var(--fg); font: 17px/1.6 var(--body); }}
.wrap {{ display: grid; grid-template-columns: minmax(0, 1fr); gap: 2.5rem; max-width: 76rem;
  margin: 0 auto; padding-inline: 16px; padding-block: 2rem 4rem; }}
@media (min-width: 1100px) {{ .wrap {{ grid-template-columns: 14rem minmax(0, 1fr); padding-inline: 2rem; }} }}
nav {{ display: none; }}
@media (min-width: 1100px) {{ nav {{ display: block; position: sticky; top: calc(env(safe-area-inset-top, 0px) + 1.5rem);
  align-self: start; font: 500 0.82rem/1.4 var(--display); letter-spacing: 0.02em; }} }}
nav ol {{ list-style: none; padding: 0; margin: 0; display: grid; gap: 0.45rem; border-left: 1px solid var(--rule); }}
nav a {{ color: var(--muted); text-decoration: none; padding-left: 0.8rem; display: block; }}
nav a:hover, nav a:focus-visible {{ color: var(--accent); }}
nav .label {{ text-transform: uppercase; letter-spacing: 0.12em; font-size: 0.72rem; color: var(--muted); margin-bottom: 0.7rem; }}
main {{ min-width: 0; max-width: 52rem; }}
header {{ border-bottom: 2px solid var(--fg); padding-bottom: 1.2rem; margin-bottom: 1.5rem; }}
.eyebrow {{ font: 500 0.78rem var(--mono); letter-spacing: 0.08em; text-transform: uppercase; color: var(--accent); }}
h1 {{ font: 600 clamp(2rem, 5vw, 2.9rem)/1.1 var(--display); margin: 0.4rem 0 0; text-wrap: balance; letter-spacing: -0.01em; }}
h2 {{ font: 600 1.55rem/1.2 var(--display); margin: 3rem 0 0.8rem; padding-top: 0.6rem; border-top: 1px solid var(--rule); text-wrap: balance; }}
h3 {{ font: 600 1.15rem/1.3 var(--display); margin: 2rem 0 0.6rem; }}
p, li {{ max-width: 70ch; }}
ul, ol {{ padding-left: 1.3rem; }}
li {{ margin: 0.2rem 0; }}
li > ul, li > ol {{ margin: 0.2rem 0; }}
strong {{ font-weight: 600; }}
a {{ color: var(--accent); }}
code {{ font: 0.86em var(--mono); background: var(--code-bg); padding: 0.08em 0.3em; border-radius: 3px; overflow-wrap: anywhere; }}
.scroll {{ overflow-x: auto; margin: 1.2rem 0; }}
pre.code {{ font: 0.84rem/1.55 var(--mono); background: var(--code-bg); padding: 1rem 1.1rem; margin: 0; border-radius: 4px; }}
.diagram {{ border: 1px solid var(--rule); border-radius: 4px; padding: 1rem; background: var(--surface); }}
pre.mermaid {{ margin: 0; text-align: center; font: 0.8rem var(--mono); }}
table {{ border-collapse: collapse; font: 0.88rem/1.45 var(--display); font-variant-numeric: tabular-nums; min-width: 100%; }}
th {{ text-align: left; font-weight: 600; border-bottom: 1.5px solid var(--fg); padding: 0.45rem 0.7rem 0.45rem 0; }}
td {{ border-bottom: 1px solid var(--rule); padding: 0.4rem 0.7rem 0.4rem 0; vertical-align: top; }}
td.num, th.num {{ text-align: right; font-family: var(--mono); font-size: 0.84rem; }}
td code {{ font-size: 0.8rem; }}
td.win {{ background: var(--win-bg); color: var(--win-fg); font-weight: 600; }}
figure {{ margin: 1.8rem 0; }}
.plate {{ background: #ffffff; border: 1px solid var(--rule); border-radius: 4px; padding: 0.6rem; overflow-x: auto; }}
.plate img {{ display: block; width: 100%; min-width: 34rem; height: auto; }}
figcaption {{ font-size: 0.93rem; color: var(--muted); margin-top: 0.6rem; }}
p.bottom {{ border-left: 3px solid var(--accent-2); padding: 0.5rem 0.9rem; background: var(--code-bg); border-radius: 0 4px 4px 0; }}
figcaption strong {{ color: var(--fg); }}
.key {{ display: flex; flex-wrap: wrap; gap: 0.6rem 2rem; margin-top: 1.2rem; font: 0.85rem var(--display); color: var(--muted); }}
.key b {{ display: block; font: 500 1.5rem var(--mono); color: var(--accent-2); }}
@media (prefers-reduced-motion: reduce) {{ * {{ scroll-behavior: auto; }} }}
</style>
<div class="wrap">
<nav aria-label="Sections"><div class="label">Sections</div><ol>{toc}</ol></nav>
<main>
<header>
<div class="eyebrow">SpectralFM · frozen-encoder training record</div>
<h1>{title}</h1>
<div class="key">
<div><b>8/8</b>wins vs raw spectra</div>
<div><b>8/8</b>wins vs random weights</div>
<div><b>+.0556</b>mean ΔR² vs raw</div>
<div><b>≈6×</b>less 1%-noise drift</div>
</div>
</header>
{body}
</main>
</div>
<script>
// The artifact viewer renders Mermaid itself; a local copy of the file loads the renderer.
if (location.protocol === "file:") {{
  const s = document.createElement("script");
  s.src = "https://cdnjs.cloudflare.com/ajax/libs/mermaid/10.9.1/mermaid.min.js";
  s.onload = () => {{
    const dark = matchMedia("(prefers-color-scheme: dark)").matches;
    mermaid.initialize({{ startOnLoad: false, theme: dark ? "dark" : "neutral" }});
    mermaid.run({{ querySelector: "pre.mermaid" }});
  }};
  document.head.appendChild(s);
}}
</script>
'''


def main():
    title, toc, body = convert(SRC.read_text())
    nav = ''.join(f'<li><a href="#{s}">{html.escape(t)}</a></li>' for s, t in toc)
    DST.write_text(PAGE.format(title=html.escape(title), toc=nav, body=body))


if __name__ == '__main__':
    main()
