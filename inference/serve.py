import os

import torch

from Models.Config import GPTConfig
from Models.Transformer import GPTModel
from Training.tokenizer import Tokenizer as ScratchTokenizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

GPT2_BEST_DIR = "Models/gpt2_shakespeare/best"
GPT2_LAST_DIR = "Models/gpt2_shakespeare/last"
SCRATCH_CHECKPOINT = "checkpoint_best.pt"
CORPUS_PATH = "Data/Raw/shakespeare_clean.txt"

_gpt2_model = None
_gpt2_tokenizer = None
_scratch_model = None
_scratch_tokenizer = None


def _gpt2_source_dir():
    if os.path.isdir(GPT2_BEST_DIR):
        return GPT2_BEST_DIR
    if os.path.isdir(GPT2_LAST_DIR):
        return GPT2_LAST_DIR
    return None


def _load_gpt2():
    global _gpt2_model, _gpt2_tokenizer

    from transformers import GPT2LMHeadModel, GPT2TokenizerFast

    source = _gpt2_source_dir()
    if source is None:
        raise FileNotFoundError(
            "No fine-tuned GPT-2 checkpoint found yet. Training is likely still in progress."
        )

    if _gpt2_model is None or getattr(_gpt2_model, "_source", None) != source:
        _gpt2_tokenizer = GPT2TokenizerFast.from_pretrained(source)
        _gpt2_model = GPT2LMHeadModel.from_pretrained(source).to(DEVICE)
        _gpt2_model._source = source
        _gpt2_model.eval()

    return _gpt2_model, _gpt2_tokenizer


def _load_scratch():
    global _scratch_model, _scratch_tokenizer

    if not os.path.exists(SCRATCH_CHECKPOINT):
        raise FileNotFoundError(f"{SCRATCH_CHECKPOINT} not found.")

    if _scratch_model is None:
        _scratch_tokenizer = ScratchTokenizer(CORPUS_PATH)
        config = GPTConfig(vocab_size=_scratch_tokenizer.vocab_size)
        _scratch_model = GPTModel(config).to(DEVICE)
        _scratch_model.load_state_dict(torch.load(SCRATCH_CHECKPOINT, map_location=DEVICE))
        _scratch_model.eval()

    return _scratch_model, _scratch_tokenizer


@torch.no_grad()
def generate_gpt2(prompt, max_new_tokens=200, temperature=0.8, top_k=40, repetition_penalty=1.2):
    model, tokenizer = _load_gpt2()

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)
    output_ids = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=max(temperature, 1e-3),
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        no_repeat_ngram_size=3,
        pad_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)


@torch.no_grad()
def generate_scratch(prompt, max_new_tokens=200, temperature=0.8, top_k=40, repetition_penalty=1.2):
    model, tokenizer = _load_scratch()

    tokens = torch.tensor(tokenizer.encode(prompt), dtype=torch.long).unsqueeze(0).to(DEVICE)

    for _ in range(max_new_tokens):
        tokens_cond = tokens[:, -model.block_size:]
        logits = model(tokens_cond)[:, -1, :] / max(temperature, 1e-3)

        if repetition_penalty != 1.0:
            seen = torch.unique(tokens_cond)
            seen_logits = logits[:, seen]
            logits[:, seen] = torch.where(
                seen_logits > 0, seen_logits / repetition_penalty, seen_logits * repetition_penalty
            )

        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < v[:, [-1]]] = float("-inf")

        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        tokens = torch.cat((tokens, next_token), dim=1)

    return tokenizer.decode(tokens[0].tolist())


def available_models():
    return {
        "gpt2": {
            "label": "GPT-2, fine-tuned on Shakespeare",
            "ready": _gpt2_source_dir() is not None,
        },
        "scratch": {
            "label": "From-scratch GPT (Shakespeare-only)",
            "ready": os.path.exists(SCRATCH_CHECKPOINT),
        },
    }


def generate(model_name, prompt, **kwargs):
    if model_name == "gpt2":
        return generate_gpt2(prompt, **kwargs)
    if model_name == "scratch":
        return generate_scratch(prompt, **kwargs)
    raise ValueError(f"Unknown model: {model_name}")
