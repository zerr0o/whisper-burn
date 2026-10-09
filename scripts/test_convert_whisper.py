"""Run with: uv run --no-project --with numpy python scripts/test_convert_whisper.py"""

from convert_whisper import hf_name_to_gguf, should_quantize


def test_conversion_rules():
    weight = "encoder.blocks.0.attn.query.weight"
    for shape in [(), (256,), (255, 256), (256, 255), (256, 256, 1)]:
        assert not should_quantize(weight, shape), shape
    assert should_quantize(weight, (256, 256))
    assert should_quantize(weight, (1280, 5120))
    for part in ["bias", "ln", "layer_norm", "positional_embedding", "token_embedding", "conv"]:
        assert not should_quantize(f"encoder.{part}.weight", (256, 256)), part

    mappings = {
        "encoder.layers.0.self_attn.q_proj.weight": weight,
        "decoder.layers.1.encoder_attn.k_proj.weight": "decoder.blocks.1.cross_attn.key.weight",
        "encoder.conv1.weight": "encoder.conv1.weight",
        "decoder.embed_tokens.weight": "decoder.token_embedding.weight",
        "proj_out.weight": None,
    }
    for source, expected in mappings.items():
        for prefix in ["", "model."]:
            assert hf_name_to_gguf(prefix + source) == expected, source


if __name__ == "__main__":
    test_conversion_rules()
    print("Conversion rule checks passed.")
