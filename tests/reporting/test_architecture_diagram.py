from xml.etree import ElementTree

from spectral_lejepa.reporting.architecture import render_dataflow


def test_dataflow_svg_shows_training_and_production_tensor_shapes():
    svg = render_dataflow()

    labels = {node.text for node in ElementTree.fromstring(svg).iter()
              if node.tag.endswith('text') and node.text}
    for label in (
        "Training · batch B=256 · all six encoder blocks update",
        "Production inference", "float32 (B, 245)",
        "Global norm", "pad to width 6 → (B, 48, 6)",
        "tokens (B, 48, 256)", "context (B, 12, 256)",
        "predicted (B, 36, 256)", "final norm → (B, 48, 256)",
        "transpose target to (48, B, 256); λ = 0.05", "(3, B, 256)",
        "Shared encoder · first 2 of 6 blocks",
        "before final 6-block norm", "(B, 48, 256) → (B, 12288)",
        "The returned feature has 12,288 dimensions, not 256.",
    ):
        assert label in labels

    assert svg.startswith('<svg') and svg.endswith('</svg>')
    assert 'class="section"' not in svg and 'class="note"' not in svg
    assert 'class="df-section"' in svg and 'class="df-note"' in svg
    assert 'M187,755 V740 H777 V755' in svg


def test_block_diagram_shows_every_branch_with_tensor_shapes():
    from spectral_lejepa.reporting.architecture import block_diagram
    svg = block_diagram()
    root = ElementTree.fromstring(svg)   # well-formed
    text = ' '.join(''.join(n.itertext()) for n in root.iter() if n.tag.endswith('text'))
    for label in ('(B, 245)', '(B, 48, 6)', '(B, 48, 256)', '(B, 12, 256)', '(B, 36, 256)', '(3, B, 256)',
                  '(B, 12288)', '(B, 4, 12288)', 'Tokenizer', 'Encoder', 'Predictor', 'Blocks 1–2',
                  '14,080', '4,739,072', '1,888,000', '1,593,600', 'MSE', 'SIGReg', 'robust'):
        assert label in text, label
    assert 'linearGradient' in svg   # shaded 3-D blocks


def test_block_diagram_titles_and_explains_terms():
    from spectral_lejepa.reporting.architecture import block_diagram
    root = ElementTree.fromstring(block_diagram())
    text = [''.join(n.itertext()) for n in root.iter() if n.tag.endswith('text')]
    assert any(t.startswith('Architecture') for t in text)
    assert not any('Training step (batch' in t for t in text)
    joined = ' '.join(text)
    for label in ('B = 256', 'LayerNorm', 'transformer', 'self-attention', 'robustness'):
        assert label in joined, label
    assert ' LN' not in joined   # the abbreviation is spelled out
