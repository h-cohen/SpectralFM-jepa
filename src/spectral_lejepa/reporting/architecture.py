"""Render the final LeJEPA training and production inference data flow as inline SVG."""
from __future__ import annotations

from html import escape


def _box(x: int, y: int, width: int, height: int, title: str, lines: tuple[str, ...],
         fill: str = "#ffffff", stroke: str = "#9fb5bd") -> str:
    title_y = y + 25
    text = [f'<text x="{x + 12}" y="{title_y}" class="df-title">{escape(title)}</text>']
    for i, line in enumerate(lines):
        text.append(f'<text x="{x + 12}" y="{title_y + 23 + i * 19}" class="df-body">{escape(line)}</text>')
    return (f'<rect x="{x}" y="{y}" width="{width}" height="{height}" rx="11" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>{"".join(text)}')


def _arrow(x1: int, y1: int, x2: int, y2: int, label: str | None = None) -> str:
    line = (f'<path d="M{x1},{y1} L{x2},{y2}" fill="none" stroke="#56717b" '
            'stroke-width="2" marker-end="url(#arrow)"/>')
    if label:
        line += f'<text x="{(x1 + x2) / 2}" y="{(y1 + y2) / 2 - 7}" class="df-edge">{escape(label)}</text>'
    return line


def render_dataflow() -> str:
    """Return a dependency-free SVG showing training tensors and production inference."""
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" class="spectralfm-dataflow" viewBox="0 0 1500 1050" role="img" '
        'aria-labelledby="title description">',
        '<title id="title">SpectralFM final training and production data flow</title>',
        '<desc id="description">LeJEPA training uses all six encoder blocks, while production inference uses the first two and returns a 12,288-dimensional flattened representation.</desc>',
        '<defs><marker id="arrow" markerWidth="9" markerHeight="7" refX="8" refY="3.5" orient="auto">'
        '<path d="M0,0 L9,3.5 L0,7 z" fill="#56717b"/></marker></defs>',
        '<style>.spectralfm-dataflow text{font-family:Arial,sans-serif;fill:#18333e}.df-panel{fill:#f5f8f8;stroke:#c3d1d5;stroke-width:1.5}.df-section{font-size:22px;font-weight:700}.df-title{font-size:14px;font-weight:700}.df-body{font-size:12px}.df-edge{font-size:11px;text-anchor:middle;fill:#47636d}.df-note{font-size:13px;fill:#47636d}</style>',
        '<rect class="df-panel" x="20" y="20" width="965" height="1005" rx="18"/>',
        '<text class="df-section" x="45" y="58">Training · batch B=256 · all six encoder blocks update</text>',
        '<text class="df-note" x="45" y="82">No EMA teacher or stop-gradient; context and target share encoder weights.</text>',
        '<rect class="df-panel" x="1005" y="20" width="475" height="1005" rx="18"/>',
        '<text class="df-section" x="1030" y="58">Production inference</text>',
        '<text class="df-note" x="1030" y="82">Raw float32 input; no predictor or final encoder norm.</text>',
        # Shared training stem.
        _box(45, 115, 150, 88, 'Raw spectrum', ('float32 (B, 245)', 'canonical B=1: (1, 245)'), '#eaf3f5'),
        _box(220, 115, 170, 88, 'Global norm', ('(x − μg) / σg', 'float32 (B, 245)'), '#eaf3f5'),
        _box(415, 115, 215, 88, 'Contiguous patches', ('5 × width 6 + 43 × width 5', 'pad to width 6 → (B, 48, 6)'), '#eaf3f5'),
        _box(655, 115, 270, 88, 'Linear(6 → 256) + position', ('tokens (B, 48, 256)', 'shared encoder input'), '#eaf3f5'),
        _arrow(195, 159, 220, 159), _arrow(390, 159, 415, 159), _arrow(630, 159, 655, 159),
        # Token-level JEPA.
        '<rect x="45" y="230" width="890" height="415" rx="14" fill="#ffffff" stroke="#d4dfe2"/>',
        '<text class="df-section" x="65" y="263">Token-level JEPA objective</text>',
        '<text class="df-note" x="65" y="284">Both paths start from the shared (B, 48, 256) tokens above.</text>',
        _box(65, 305, 175, 82, 'Select visible tokens', ('12 / sample', '(B, 12, 256)'), '#edf6f1', '#9dbdad'),
        _box(275, 305, 190, 82, 'Shared encoder', ('blocks 1–6', 'context (B, 12, 256)'), '#edf6f1', '#9dbdad'),
        _box(500, 305, 175, 82, 'Predictor', ('4 blocks, width 192',), '#edf6f1', '#9dbdad'),
        _box(710, 305, 195, 82, 'Masked prediction', ('predicted (B, 36, 256)',), '#edf6f1', '#9dbdad'),
        _arrow(240, 346, 275, 346), _arrow(465, 346, 500, 346), _arrow(675, 346, 710, 346),
        _box(65, 440, 200, 95, 'Full target pass', ('shared encoder blocks 1–6', 'final norm → (B, 48, 256)'), '#fff6e7', '#d7bd88'),
        _box(330, 440, 190, 95, 'Gather masked target', ('target_masked', '(B, 36, 256)'), '#fff6e7', '#d7bd88'),
        _box(575, 440, 195, 95, 'Masked-token MSE', ('predicted vs target_masked',), '#f5f0fa', '#b9a6ce'),
        _box(530, 555, 350, 65, 'Token SIGReg', ('transpose target to (48, B, 256); λ = 0.05',), '#f5f0fa', '#b9a6ce'),
        '<path d="M807,387 V418 H672 V440" fill="none" stroke="#56717b" stroke-width="2" marker-end="url(#arrow)"/>',
        _arrow(265, 488, 330, 488), _arrow(520, 488, 575, 488),
        '<path d="M165,535 V587 H530" fill="none" stroke="#56717b" stroke-width="2" marker-end="url(#arrow)"/>',
        '<text class="df-note" x="65" y="625">Token loss = masked MSE + 0.05 × mean SIGReg across the 48 token positions.</text>',
        # Multi-view pooled objective.
        '<rect x="45" y="675" width="890" height="315" rx="14" fill="#ffffff" stroke="#d4dfe2"/>',
        '<text class="df-section" x="65" y="708">Pooled multi-view objective</text>',
        '<text class="df-note" x="65" y="730">Uses the same token sequence and full target; all views share encoder weights.</text>',
        _box(65, 755, 245, 95, 'Two random subsets', ('24 tokens each → shared encoder', 'pool each to (B, 256)'), '#edf6f1', '#9dbdad'),
        _box(355, 755, 225, 95, 'Full-spectrum target', ('pool 48 target tokens', '(B, 256)'), '#fff6e7', '#d7bd88'),
        _box(650, 755, 255, 95, 'Pooled views', ('full + 2 subsets', '(3, B, 256)'), '#eaf3f5'),
        _box(330, 890, 395, 75, 'View invariance + pooled SIGReg', ('0.95 × invariance + 0.05 × SIGReg',), '#f5f0fa', '#b9a6ce'),
        '<path d="M187,755 V740 H777 V755" fill="none" stroke="#56717b" stroke-width="2" marker-end="url(#arrow)"/>',
        _arrow(580, 802, 650, 802),
        _arrow(777, 850, 527, 890),
        # Inference stem and output.
        _box(1035, 130, 190, 95, 'Raw spectrum', ('float32 (B, 245)', 'canonical (1, 245)'), '#eaf3f5'),
        _box(1260, 130, 190, 95, 'Global norm', ('float32 (B, 245)', '(x − μg) / σg'), '#eaf3f5'),
        _box(1035, 285, 190, 110, '48 contiguous patches', ('5 × 6 + 43 × 5', 'pad to width 6', '(B, 48, 6)'), '#eaf3f5'),
        _box(1260, 285, 190, 110, 'Patch projection', ('Linear(6 → 256)', '+ learned positions', '(B, 48, 256)'), '#eaf3f5'),
        _box(1035, 475, 415, 120, 'Shared encoder · first 2 of 6 blocks', ('raw hidden after block 2', 'before final 6-block norm', '(B, 48, 256)'), '#edf6f1', '#9dbdad'),
        _box(1035, 680, 415, 115, 'Flatten token and feature axes', ('(B, 48, 256) → (B, 12288)', 'canonical single spectrum → (1, 12288)'), '#f5f0fa', '#b9a6ce'),
        _box(1035, 865, 415, 90, 'Exported artifact', ('outputs/final-model/production/encoder.pt', 'accepts raw float32 (B, 245)'), '#fff6e7', '#d7bd88'),
        _arrow(1225, 177, 1260, 177), _arrow(1355, 225, 1130, 285), _arrow(1225, 340, 1260, 340),
        _arrow(1355, 395, 1240, 475), _arrow(1240, 595, 1240, 680), _arrow(1240, 795, 1240, 865),
        '<text class="df-note" x="1035" y="980">The returned feature has 12,288 dimensions, not 256.</text>',
        '</svg>',
    ]
    return ''.join(parts)


# --- Block diagram for the training report: shaded 3-D blocks, tensor shapes on every edge. ---

_FONT = "Helvetica Neue, Helvetica, Arial, sans-serif"
_MONO = "SFMono-Regular, Menlo, Consolas, monospace"
_TOKENS = ("#f28e2b", "#2b6f9e", "#2a9d8f", "#9aa5b1")


def _block(x, y, w, h, title, sub, params=None, d=9, grad="blk"):
    """Extruded block: gradient front face, lighter top face, darker side face, white centred text."""
    cx = x + w / 2
    lines = [f'<polygon points="{x},{y} {x + d},{y - d} {x + w + d},{y - d} {x + w},{y}" fill="#6f9fc0"/>',
             f'<polygon points="{x + w},{y} {x + w + d},{y - d} {x + w + d},{y + h - d} {x + w},{y + h}" '
             'fill="#0d5560"/>',
             f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="url(#{grad})"/>']
    rows = [(title, 'bt')] + [(s, 'bs') for s in ((sub,) if isinstance(sub, str) else sub)]
    if params:
        rows.append((f'{params} params', 'bp'))
    top = y + h / 2 - (len(rows) - 1) * 8.5 + 5
    for i, (t, cls) in enumerate(rows):
        lines.append(f'<text x="{cx}" y="{top + i * 17}" class="{cls}">{escape(t)}</text>')
    return ''.join(lines)


def _pill(x, y, w, h, title, sub=None):
    t = [f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{min(h / 2, 14)}" fill="#fff" '
         'stroke="#c0504d" stroke-width="1.6"/>']
    if sub:
        t += [f'<text x="{x + w / 2}" y="{y + h / 2 - 3}" class="lt">{escape(title)}</text>',
              f'<text x="{x + w / 2}" y="{y + h / 2 + 13}" class="ls">{escape(sub)}</text>']
    else:
        t.append(f'<text x="{x + w / 2}" y="{y + h / 2 + 5}" class="lt">{escape(title)}</text>')
    return ''.join(t)


def _tokens(x, y, n=4, s=13):
    return ''.join(f'<rect x="{x + i * s}" y="{y}" width="{s}" height="{s}" fill="{_TOKENS[i % 4]}" '
                   'stroke="#fff" stroke-width="1"/>' for i in range(n))


def _line(points, dashed=False, head=True):
    d = ' '.join(f'{"M" if i == 0 else "L"}{x},{y}' for i, (x, y) in enumerate(points))
    dash = ' stroke-dasharray="5 4"' if dashed else ''
    end = ' marker-end="url(#ah)"' if head else ''
    return f'<path d="{d}" fill="none" stroke="#333" stroke-width="1.6"{dash}{end}/>'


def _label(x, y, text, cls='sh', anchor='middle'):
    return f'<text x="{x}" y="{y}" class="{cls}" text-anchor="{anchor}">{escape(text)}</text>'


def block_diagram() -> str:
    """Architecture and data flow of one training step and of one shipped member, as standalone SVG."""
    W, H = 1280, 940
    p = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
         'font-family="' + _FONT + '">',
         '<defs><linearGradient id="blk" x1="0" y1="0" x2="1" y2="1">'
         '<stop offset="0" stop-color="#1d3a6e"/><stop offset="1" stop-color="#14808a"/></linearGradient>'
         '<marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
         'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="#333"/></marker></defs>',
         '<style>'
         '.bt{font-weight:700;font-size:15px;fill:#fff;text-anchor:middle}'
         '.bs{font-size:12.5px;fill:#e6f1f3;text-anchor:middle}'
         '.bp{font-size:11.5px;fill:#a9d6dc;text-anchor:middle;font-family:' + _MONO + '}'
         '.sh{font-size:12.5px;fill:#222;font-family:' + _MONO + '}'
         '.nt{font-size:12px;fill:#555}'
         '.hd{font-size:17px;font-weight:700;fill:#111}'
         '.lt{font-size:13px;font-weight:700;fill:#8c2f2b;text-anchor:middle}'
         '.ls{font-size:11.5px;fill:#8c2f2b;text-anchor:middle}'
         '.eq{font-size:16px;fill:#111;font-family:Georgia, serif}'
         '</style>',
         f'<rect width="{W}" height="{H}" fill="#fff"/>']
    # ---- training panel ----
    p.append('<rect x="15" y="15" width="1250" height="600" rx="12" fill="#f2f3f5"/>')
    p.append(_label(35, 45, 'Architecture: LeJEPA training step', 'hd', 'start'))
    p.append(_label(35, 66, 'batch B = 256 · bracketed encoders = one module, shared weights, no EMA teacher', 'nt', 'start'))
    p.append(_label(1250, 45, 'Encoder: 6 standard pre-LayerNorm transformer blocks, d = 256', 'nt', 'end'))
    p.append(_label(1250, 62, 'each block: 8-head self-attention + GELU MLP (256 → 1024 → 256), residual', 'nt', 'end'))
    # input and tokenizer
    p.append(_label(95, 117, 'spectrum x', 'nt'))
    p.append(_label(95, 135, '(B, 245)'))
    p.append(_line([(150, 127), (215, 127)]))
    p.append(_label(182, 117, 'norm', 'nt'))
    p.append(_block(220, 90, 250, 76, 'Tokenizer', ('48 patches → (B, 48, 6)', 'Linear(6→256) + positions'),
                    '14,080'))
    p.append(_line([(480, 127), (520, 127)]))
    p.append(_tokens(526, 120))
    p.append(_label(560, 153, '(B, 48, 256)'))
    # bus from the tokens to the three encoder branches
    bus_x, enc_x, enc_w = 560, 700, 180
    out_x = enc_x + enc_w + 9
    rows = {'ctx': 225, 'tgt': 335, 'view': 445}
    p.append(_line([(bus_x, 162), (bus_x, rows['view'])], head=False))
    p.append(_label(bus_x + 8, 180, '75% masked', 'nt', 'start'))
    for key, (lab, shape) in {'ctx': ('12 visible', '(B, 12, 256)'),
                              'tgt': ('all 48', '(B, 48, 256)'),
                              'view': ('2 views × 24', '(2, B, 24, 256)')}.items():
        y = rows[key]
        p.append(_line([(bus_x, y), (enc_x - 26, y)]))
        p.append(_label(bus_x + 8, y - 8, lab, 'nt', 'start'))
        p.append(_label(bus_x + 8, y + 17, shape, 'sh', 'start'))
    p.append(_block(enc_x, rows['ctx'] - 38, enc_w, 76, 'Encoder', ('6 transformer blocks', 'context path'),
                    '4,739,072'))
    p.append(_block(enc_x, rows['tgt'] - 38, enc_w, 76, 'Encoder', ('6 transformer blocks', '+ final LayerNorm')))
    p.append(_block(enc_x, rows['view'] - 38, enc_w, 76, 'Encoder', ('6 transformer blocks', '+ final LayerNorm')))
    # bracket: the three encoders are one module
    top, bot = rows['ctx'] - 50, rows['view'] + 40
    p.append(f'<path d="M{enc_x - 8},{top} L{enc_x - 20},{top} L{enc_x - 20},{bot} L{enc_x - 8},{bot}" '
             'fill="none" stroke="#14808a" stroke-width="2"/>')
    # context -> predictor -> MSE
    pr_x, pr_w = 985, 150
    p.append(_line([(out_x, rows['ctx']), (pr_x - 4, rows['ctx'])]))
    p.append(_label((out_x + pr_x) / 2, rows['ctx'] - 8, '(B, 12, 256)'))
    p.append(_block(pr_x, rows['ctx'] - 46, pr_w, 92, 'Predictor', ('4 transformer blocks', 'd = 192', '+ 36 mask tokens'),
                    '1,888,000'))
    p.append(_line([(pr_x + pr_w / 2, rows['ctx'] - 46 - 9), (pr_x + pr_w / 2, 108), (1166, 108)]))
    p.append(_label(pr_x + pr_w / 2 - 8, 150, '(B, 36, 256)', 'sh', 'end'))
    p.append(_pill(1170, 88, 80, 64, 'MSE'))
    # target: token SIGReg; masked targets up to MSE; pooled full view down; block-2 state down to robust
    p.append(_line([(out_x, rows['tgt']), (1166, rows['tgt'])]))
    p.append(_label(950, rows['tgt'] + 17, '(B, 48, 256)'))
    p.append(_pill(1170, rows['tgt'] - 26, 80, 52, 'SIGReg', 'tokens'))
    for x in (1030, 1100, 1150):
        p.append(f'<circle cx="{x}" cy="{rows["tgt"]}" r="3.5" fill="#333"/>')
    p.append(_line([(1150, rows['tgt']), (1150, 140), (1166, 140)]))
    p.append(_label(1143, 298, '36 masked', 'nt', 'end'))
    p.append(_label(1143, 314, '(B, 36, 256)', 'sh', 'end'))
    p.append(_line([(1100, rows['tgt']), (1100, rows['view'] - 34)], dashed=True))
    p.append(_label(1106, rows['tgt'] + 40, 'full-view mean', 'nt', 'start'))
    p.append(_line([(out_x, rows['view']), (1048, rows['view'])]))
    p.append(_label(955, rows['view'] - 8, 'mean-pool', 'nt'))
    p.append(_label(955, rows['view'] + 17, '(2, B, 256)'))
    p.append(_pill(1052, rows['view'] - 30, 198, 60, '0.95·Var + 0.05·SIGReg', 'pooled views (3, B, 256)'))
    # stage-2 robust branch
    ry = 548
    p.append('<line x1="30" y1="484" x2="1250" y2="484" stroke="#c9ccd1" stroke-dasharray="4 4"/>')
    p.append('<text x="35" y="503" class="hd" font-size="14" text-anchor="start">Stage 2 only: robustness to '
             'measurement perturbations</text>')
    p.append(_line([(95, 143), (95, ry), (215, ry)]))
    p.append(_block(220, ry - 30, 250, 60, 'Perturb x → x′', 'noise · offset · slope · scale · shift'))
    p.append(_line([(479, ry), (enc_x - 4, ry)]))
    p.append(_label(590, ry - 8, '(B, 245)'))
    p.append(_block(enc_x, ry - 32, enc_w, 64, 'Blocks 1–2', ('first 2 encoder blocks', 'same weights')))
    p.append(_line([(out_x, ry), (1048, ry)]))
    p.append(_label(955, ry + 17, '(B, 48, 256)'))
    p.append(_label(955, ry - 8, 'h₂(x′)', 'nt'))
    p.append(_line([(1030, rows['tgt']), (1030, 503), (1110, 503), (1110, ry - 30)], dashed=True))
    p.append(_label(1022, 500, 'h₂(x): block 2 of the target path', 'nt', 'end'))
    p.append(_pill(1052, ry - 26, 198, 52, 'robust', '‖h₂(x′) − h₂(x)‖² / Var h₂'))
    # ---- inference panel ----
    iy = 718
    p.append('<rect x="15" y="630" width="1250" height="172" rx="12" fill="#fdecd6" stroke="#e3a35d" '
             'stroke-dasharray="7 5" stroke-width="1.6"/>')
    p.append(_label(35, 660, 'Inference: one shipped member = tokenizer + blocks 1–2 (1,593,600 params)',
                    'hd', 'start'))
    p.append(_label(95, iy + 5, 'x (B, 245)'))
    p.append(_line([(150, iy), (215, iy)]))
    p.append(_label(182, iy - 10, 'norm', 'nt'))
    p.append(_block(220, iy - 30, 220, 60, 'Tokenizer', '(B, 48, 6) → (B, 48, 256)'))
    p.append(_line([(449, iy), (530, iy)]))
    p.append(_block(535, iy - 30, 180, 60, 'Blocks 1–2', 'no final LayerNorm'))
    p.append(_line([(724, iy), (810, iy)]))
    p.append(_label(767, iy - 10, '(B, 48, 256)'))
    p.append(_block(815, iy - 30, 130, 60, 'Flatten', '48 × 256'))
    p.append(_line([(954, iy), (1050, iy)]))
    p.append(_label(1002, iy - 10, '(B, 12288)'))
    p.append(_tokens(1062, iy - 7, 4, 14))
    p.append(_label(1090, iy - 22, '4 members', 'nt'))
    p.append(_label(1090, iy + 32, '(B, 4, 12288)'))
    p.append(_label(640, 788, 'Each member is a stage-1 seed plus its stage-2 continuation; downstream '
                              'predictions are averaged over the 4 members.', 'nt'))
    # ---- loss ----
    p.append('<rect x="15" y="815" width="1250" height="110" rx="12" fill="#f2f3f5"/>')
    p.append(_label(35, 845, 'Loss', 'hd', 'start'))
    p.append('<text x="640" y="887" class="eq" text-anchor="middle">L = MSE + 0.05 · SIGReg(tokens) + [ 0.95 · Var(views) + 0.05 · SIGReg(pooled) ] + 0.1 · robust (stage 2)'
             '</text>')
    p.append('</svg>')
    return ''.join(p)
