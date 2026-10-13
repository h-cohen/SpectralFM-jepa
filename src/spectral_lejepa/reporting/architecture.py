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
