"""Optional mathematical checks for the experimental benchmark-only adapter."""
import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('transformers')
pytest.importorskip('sentencepiece')

from scripts.benchmark_omni_torch import (
    FloatRotaryEmbedding, decoder_key, encoder_config, encoder_key,
    native_precision_rotary, rotary_rows,
)
from transformers import LlamaConfig, LlamaForCausalLM


def test_official_architecture_and_strict_key_mapping():
    config = encoder_config()
    assert config.num_attention_heads == 16
    assert config.num_hidden_layers == 60
    assert config.do_stable_layer_norm
    assert encoder_key('encoder_frontend.pos_encoder.conv.weight_g') == (
        'encoder.pos_conv_embed.conv.parametrizations.weight.original0')
    assert decoder_key('llama_decoder.layers.11.self_attn.output_proj.weight') == (
        'model.layers.11.self_attn.o_proj.weight')
    with pytest.raises(ValueError, match='Unmapped'):
        encoder_key('encoder.layers.0.an_unrecognized_projection.weight')
    with pytest.raises(ValueError, match='Unmapped'):
        decoder_key('llama_decoder.layers.0.an_unrecognized_projection.weight')


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_rotary_conversion_matches_native_complex_formula(dtype):
    torch.manual_seed(123)
    x, weight = torch.randn(2, 7, 32).to(dtype), torch.randn(32, 32).to(dtype)
    projected = torch.nn.functional.linear(x, weight).view(2, 7, 4, 8).transpose(1, 2)
    pairs = torch.view_as_complex(projected.float().reshape(2, 4, 7, 4, 2).contiguous())
    angles = torch.arange(7)[:, None] / (10000 ** (torch.arange(0, 8, 2).float()[None, :] / 8))
    native = torch.view_as_real(pairs * torch.polar(torch.ones_like(angles), angles)).to(dtype)
    native_halves = native.transpose(-1, -2).reshape(2, 4, 7, 8)
    port = torch.nn.functional.linear(x, rotary_rows(weight, 4)).view(2, 7, 4, 8).transpose(1, 2)
    config = LlamaConfig(hidden_size=32, num_attention_heads=4, num_key_value_heads=4)
    cos, sin = FloatRotaryEmbedding(config)(port, torch.arange(7)[None, :])
    rotated, _ = native_precision_rotary(port, port, cos, sin)
    torch.testing.assert_close(rotated, native_halves, rtol=1e-5, atol=1e-5)


def test_reordered_beam_cache_matches_uncached_decoder():
    torch.manual_seed(123)
    config = LlamaConfig(hidden_size=32, num_attention_heads=4, num_key_value_heads=4,
        num_hidden_layers=2, intermediate_size=48, vocab_size=13, rms_norm_eps=1e-5)
    model = LlamaForCausalLM(config).eval()
    with torch.inference_mode():
        prefix = torch.randn(2, 7, 32)
        prefill = model(inputs_embeds=prefix, use_cache=True, logits_to_keep=1)
        parents, tokens = torch.tensor([1, 0]), torch.tensor([[4], [9]])
        prefill.past_key_values.reorder_cache(parents)
        incremental = model(input_ids=tokens, past_key_values=prefill.past_key_values,
                            use_cache=True, logits_to_keep=1)
        uncached = model(inputs_embeds=torch.cat([prefix[parents], model.model.embed_tokens(tokens)], 1),
                         use_cache=False, logits_to_keep=1)
        torch.testing.assert_close(incremental.logits, uncached.logits, rtol=1e-5, atol=1e-5)
