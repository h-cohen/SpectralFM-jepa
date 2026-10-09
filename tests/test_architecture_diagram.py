from xml.etree import ElementTree

from scripts.architecture_diagram import render_dataflow


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
