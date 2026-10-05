"""Experimental Windows adapter for Meta's official OmniASR 3B v2 weights.

Architectures and syntax follow facebookresearch/omnilingual-asr and fairseq2.
This is an independently implemented Transformers port, not Meta's runtime;
full native-runtime numerical parity has not been established. No weights,
reference transcripts, language models, or sample transcriptions ship here.
"""
from __future__ import annotations

import gc
from pathlib import Path
import re
import zlib

import numpy as np
import sentencepiece
import torch
from torch import nn
from torch.nn import functional as F
from transformers import LlamaConfig, LlamaForCausalLM, Wav2Vec2Config, Wav2Vec2Model
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
from transformers.models.llama import modeling_llama


class FloatRotaryEmbedding(LlamaRotaryEmbedding):
    def forward(self, x, position_ids):
        return super().forward(x.float(), position_ids)


def native_precision_rotary(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Native fairseq2 rotates in float32, then casts back to the input dtype."""
    cos, sin = cos.unsqueeze(unsqueeze_dim), sin.unsqueeze(unsqueeze_dim)
    return ((q.float() * cos + modeling_llama.rotate_half(q.float()) * sin).to(q.dtype),
            (k.float() * cos + modeling_llama.rotate_half(k.float()) * sin).to(k.dtype))


def rotary_rows(weight, heads):
    """Translate fairseq2's adjacent RoPE pairs to HF's two-half layout."""
    out_dim, in_dim = weight.shape
    return weight.reshape(heads, out_dim // heads // 2, 2, in_dim).transpose(1, 2).reshape(out_dim, in_dim)


def encoder_key(key):
    fixed = {
        'encoder_frontend.post_extract_layer_norm': 'feature_projection.layer_norm',
        'encoder_frontend.model_dim_proj': 'feature_projection.projection',
        'encoder_frontend.pos_encoder.conv.bias': 'encoder.pos_conv_embed.conv.bias',
        'encoder_frontend.pos_encoder.conv.weight_g': 'encoder.pos_conv_embed.conv.parametrizations.weight.original0',
        'encoder_frontend.pos_encoder.conv.weight_v': 'encoder.pos_conv_embed.conv.parametrizations.weight.original1',
        'encoder.layer_norm': 'encoder.layer_norm',
    }
    if key in fixed:
        return fixed[key]
    for old, new in fixed.items():
        if key in (old + '.weight', old + '.bias'):
            return new + key[len(old):]
    match = re.fullmatch(r'encoder_frontend.feature_extractor.layers.(\d+).(.+)', key)
    if match:
        return f'feature_extractor.conv_layers.{match[1]}.{match[2]}'
    match = re.fullmatch(r'encoder.layers.(\d+).(.+)', key)
    if match:
        replacements = {
            'self_attn_layer_norm': 'layer_norm', 'ffn_layer_norm': 'final_layer_norm',
            'self_attn.q_proj': 'attention.q_proj', 'self_attn.k_proj': 'attention.k_proj',
            'self_attn.v_proj': 'attention.v_proj', 'self_attn.output_proj': 'attention.out_proj',
            'ffn.inner_proj': 'feed_forward.intermediate_dense',
            'ffn.output_proj': 'feed_forward.output_dense',
        }
        part, suffix = match[2].rsplit('.', 1)
        if part in replacements:
            return f'encoder.layers.{match[1]}.{replacements[part]}.{suffix}'
    raise ValueError(f'Unmapped encoder tensor: {key}')


def decoder_key(key):
    fixed = {'text_frontend.weight': 'model.embed_tokens.weight',
             'final_proj.weight': 'lm_head.weight',
             'llama_decoder.layer_norm.weight': 'model.norm.weight'}
    if key in fixed:
        return fixed[key]
    match = re.fullmatch(r'llama_decoder.layers.(\d+).(.+)', key)
    if match:
        replacements = {
            'self_attn_layer_norm': 'input_layernorm',
            'ffn_layer_norm': 'post_attention_layernorm',
            'self_attn.q_proj': 'self_attn.q_proj', 'self_attn.k_proj': 'self_attn.k_proj',
            'self_attn.v_proj': 'self_attn.v_proj', 'self_attn.output_proj': 'self_attn.o_proj',
            'ffn.gate_proj': 'mlp.gate_proj', 'ffn.inner_proj': 'mlp.up_proj',
            'ffn.output_proj': 'mlp.down_proj',
        }
        part, suffix = match[2].rsplit('.', 1)
        if part in replacements:
            return f'model.layers.{match[1]}.{replacements[part]}.{suffix}'
    raise ValueError(f'Unmapped decoder tensor: {key}')


def load_strict(module, state):
    """No silent missing weights or random inference parameters."""
    expected = module.state_dict()
    if set(expected) != set(state):
        raise ValueError(f'State mismatch: missing={set(expected)-set(state)}, extra={set(state)-set(expected)}')
    for name, tensor in state.items():
        if tensor.shape != expected[name].shape:
            raise ValueError(f'Shape mismatch for {name}: {tensor.shape} != {expected[name].shape}')
    module.load_state_dict(state, strict=True, assign=True)
    module.requires_grad_(False).eval().to(device='cuda', dtype=torch.bfloat16)
    assert not any(p.is_meta for p in module.parameters())


def encoder_config():
    config = Wav2Vec2Config(hidden_size=2048, num_hidden_layers=60, num_attention_heads=16,
        intermediate_size=8192, hidden_act='gelu', do_stable_layer_norm=True,
        feat_extract_norm='layer', conv_dim=(512,) * 7,
        conv_stride=(5, 2, 2, 2, 2, 2, 2), conv_kernel=(10, 3, 3, 3, 3, 2, 2),
        conv_bias=True, num_conv_pos_embeddings=128, num_conv_pos_embedding_groups=16,
        mask_time_prob=0., mask_feature_prob=0., hidden_dropout=0., attention_dropout=0.,
        feat_proj_dropout=0., activation_dropout=0., layerdrop=0., layer_norm_eps=1e-5)
    config._attn_implementation = 'sdpa'
    return config


class OmniRecognizer:
    def __init__(self, directory: Path, variant: str, beam=5):
        if variant not in ('omni-ctc', 'omni-llm') or beam < 1:
            raise ValueError('Unsupported model or beam count')
        self.variant, self.beam = variant, beam
        checkpoint = directory / f'omniASR-{ "CTC" if variant == "omni-ctc" else "LLM" }-3B-v2.pt'
        state = torch.load(checkpoint, map_location='cpu', weights_only=True, mmap=True)['model']
        self.tokenizer = sentencepiece.SentencePieceProcessor(
            model_file=str(directory / 'omniASR_tokenizer_written_v2.model'))
        if self.tokenizer.get_piece_size() != 10288:
            raise ValueError('Expected the written v2 character tokenizer')
        with torch.device('meta'):
            self.encoder = Wav2Vec2Model(encoder_config())
        enc = {encoder_key(k): v for k, v in state.items()
               if k.startswith(('encoder.', 'encoder_frontend.'))}
        load_strict(self.encoder, enc)
        remaining = {k: v for k, v in state.items()
                     if not k.startswith(('encoder.', 'encoder_frontend.'))}
        if variant == 'omni-ctc':
            with torch.device('meta'):
                self.head = nn.Linear(2048, 10288)
            load_strict(self.head, {k.removeprefix('final_proj.'): v for k, v in remaining.items()})
        else:
            import pyarrow.parquet as pq
            table = pq.read_table(directory / 'languges_lookup_table.parquet').to_pylist()
            language = {r['lang'].lower(): r['index'] + 1 for r in table}['fas_arab']
            self.language_index = language
            decoder_state = {decoder_key(k): (rotary_rows(v, 8) if
                k.endswith(('self_attn.q_proj.weight', 'self_attn.k_proj.weight')) else v)
                for k, v in remaining.items() if k not in
                ('encoder_proj.weight', 'encoder_proj.bias', 'lang_embeddings.weight')}
            # The extra input embedding is the LID marker, never an output token.
            config = LlamaConfig(hidden_size=4096, num_hidden_layers=12, num_attention_heads=8,
                num_key_value_heads=8, intermediate_size=2816, vocab_size=10288,
                max_position_embeddings=8192, rms_norm_eps=1e-5, rope_theta=10000.,
                bos_token_id=0, eos_token_id=2, pad_token_id=1, tie_word_embeddings=False,
                attention_dropout=0.)
            config._attn_implementation = 'sdpa'
            with torch.device('meta'):
                self.decoder = LlamaForCausalLM(config)
                self.decoder.model.embed_tokens = nn.Embedding(10289, 4096)
                for layer in self.decoder.model.layers:
                    layer.input_layernorm = nn.RMSNorm(4096, eps=1e-5)
                    layer.post_attention_layernorm = nn.RMSNorm(4096, eps=1e-5)
                self.decoder.model.norm = nn.RMSNorm(4096, eps=1e-5)
                self.project = nn.Linear(2048, 4096)
            self.decoder.model.rotary_emb = LlamaRotaryEmbedding(config, device='cpu')
            load_strict(self.decoder, decoder_state)
            # Native RoPE computes complex64 rotations; preserve float32 frequencies.
            self.decoder.model.rotary_emb = FloatRotaryEmbedding(config, device='cuda')
            # This adapter runs in its own benchmark process, never in the GUI worker.
            modeling_llama.apply_rotary_pos_emb = native_precision_rotary
            load_strict(self.project, {k.removeprefix('encoder_proj.'): v for k, v in remaining.items()
                                     if k.startswith('encoder_proj.')})
            # Only the selected, fixed language row is needed for inference.
            self.language_embedding = remaining['lang_embeddings.weight'][language].to('cuda', torch.bfloat16)
        self.metadata = dict(runtime='experimental Windows Transformers port; native parity not established',
            checkpoint=checkpoint.name, mapped_tensor_count=len(state), encoder_heads=16,
            encoder_layers=60, precision='bfloat16', language='fas_Arab' if variant=='omni-llm' else None,
            decoder='greedy CTC collapse' if variant=='omni-ctc' else f'beam {beam}; no length normalization',
            max_new_tokens=None if variant=='omni-ctc' else 1024, truncated_clips=0)
        del state, enc, remaining
        if variant == 'omni-llm':
            del decoder_state
        gc.collect()

    def _audio(self, clips):
        inputs = [F.layer_norm(torch.as_tensor(np.asarray(c).copy(), dtype=torch.float32), (len(c),))
                  for c in clips]
        lengths = torch.tensor([len(c) for c in clips], device='cuda')
        waves = nn.utils.rnn.pad_sequence(inputs, batch_first=True).to('cuda', torch.bfloat16)
        mask = torch.arange(waves.shape[1], device='cuda')[None, :] < lengths[:, None]
        output = self.encoder(waves, attention_mask=mask.long()).last_hidden_state
        frame_lengths = self.encoder._get_feat_extract_output_lengths(lengths)
        return output, frame_lengths

    @torch.inference_mode()
    def transcribe(self, clips):
        if not clips:
            return []
        if self.variant == 'omni-ctc':
            output, lengths = self._audio(clips)
            tokens = self.head(output).argmax(-1)
            return [self.tokenizer.decode([i for i in torch.unique_consecutive(row[:int(n)]).tolist() if i!=0])
                    for row, n in zip(tokens, lengths)]
        # Serial clips keep encoder+decoder+five KV caches inside a 16 GB budget.
        result = []
        for clip in clips:
            output, _ = self._audio([clip])
            special = self.decoder.model.embed_tokens(torch.tensor([10288, 0], device='cuda'))
            prefix = torch.cat((self.project(output), special[0].view(1, 1, -1),
                               self.language_embedding.view(1, 1, -1), special[1].view(1, 1, -1)), dim=1)
            result.append(self._decode(prefix))
        return result

    def _decode(self, prefix):
        beams, vocab, eos = self.beam, 10288, 2
        output = self.decoder(inputs_embeds=prefix.repeat(beams, 1, 1), use_cache=True, logits_to_keep=1)
        scores = torch.full((beams,), -1e6, device='cuda', dtype=torch.float32)
        scores[0] = 0
        ended = torch.zeros(beams, device='cuda', dtype=torch.bool)
        history = torch.empty((beams, 0), device='cuda', dtype=torch.long)
        for step in range(1024):
            # Preserve native BF16 log-softmax if a Transformers version upcasts logits.
            probabilities = F.log_softmax(output.logits[:, -1].to(torch.bfloat16), dim=-1)
            candidates = scores[:, None] + probabilities
            candidates[ended] = -torch.inf
            candidates[ended, eos] = scores[ended]
            scores, indices = candidates.flatten().topk(beams, sorted=True)
            parents, tokens = indices // vocab, indices % vocab
            history = torch.cat((history[parents], tokens[:, None]), dim=1)
            ended = ended[parents] | (tokens == eos)
            position = prefix.shape[1] - 1 + step
            if position % 250 == 0 and step > 100:
                arrays = history[:, -101:-1].cpu().numpy()
                repeat = []
                for row in arrays:
                    data = np.array_str(row).replace('\n', '').encode('utf-8')
                    repeat.append(len(data) / max(1, len(zlib.compress(data))) > 4.)
                ended |= torch.tensor(repeat, device='cuda')
            if bool(ended.all()):
                break
            output.past_key_values.reorder_cache(parents)
            output = self.decoder(input_ids=tokens[:, None], past_key_values=output.past_key_values,
                                  use_cache=True, logits_to_keep=1)
        else:
            self.metadata['truncated_clips'] += 1
        ids = history[0].tolist()
        if eos in ids:
            ids = ids[:ids.index(eos)]
        return self.tokenizer.decode([i for i in ids if i not in (0, 1, 2)])
