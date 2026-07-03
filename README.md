# ShakespeareLLM

Two language models trained to write like Shakespeare, plus a themed web UI to talk to them.

## Models

- **From-scratch GPT** (`Models/`, `Training/train.py`) — a small pre-norm transformer with
  weight tying, trained from random initialization on a byte-level BPE tokenizer fit only on
  Shakespeare's complete works. Fast to train, but limited by how little text (~1M tokens) it
  ever sees — English fluency itself has to be learned from that corpus alone.
- **Fine-tuned GPT-2** (`Training/finetune_gpt2.py`) — starts from pretrained GPT-2 small (124M
  params, real English fluency from its original pretraining) and fine-tunes on the same
  Shakespeare corpus. Much more coherent grammar; the tradeoff is a small fine-tuning corpus
  means many epochs, with real risk of memorizing rather than generalizing — that's why both
  train.py and finetune_gpt2.py track a `best` checkpoint by validation loss rather than trusting
  the final iteration.

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Training

```bash
python -m Training.train            # from-scratch model -> checkpoint.pt / checkpoint_best.pt
python -m Training.finetune_gpt2    # GPT-2 fine-tune -> Models/gpt2_shakespeare/{best,last}
```

Both are iteration-based (not epoch-based): they sample random blocks from the corpus, use a
cosine LR schedule with warmup, and periodically re-evaluate on a held-out split.

## Generating text

CLI:

```bash
python -m inference.generate
```

Web UI:

```bash
python app.py
# open http://127.0.0.1:5000
```

The web UI lets you pick either model and tune temperature / top-k / output length.

## Layout

```
Models/          model architecture (Transformer.py, Config.py) + saved GPT-2 fine-tune
Training/        training loops, tokenizer, dataset splitting
inference/       CLI generation + serve.py (used by app.py)
Data/            raw + cleaned Shakespeare text, trained BPE tokenizer files
tools/           one-off text cleaning / tokenizer training scripts
templates/, static/   Flask web UI
```
