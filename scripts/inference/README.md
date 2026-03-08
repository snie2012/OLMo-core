# Inference Scripts

Two scripts for running inference on trained OLMo checkpoints.

Both scripts accept:
- A **native OLMo-core checkpoint** (e.g. `checkpoints/olmo3-7b/job-10104/step1000`) — the
  checkpoint is **automatically converted** to HuggingFace format on first run and cached next to
  the checkpoint in `<step_dir>/hf_converted/` (or a custom path via `--hf-output-dir`).
- A **pre-converted HuggingFace checkpoint** directory.
- A **HuggingFace Hub model ID** (e.g. `allenai/OLMo-2-1124-7B`).

### Native checkpoint format

OLMo-core saves checkpoints with this layout:

```
step1000/
├── config.json          # OLMo-core experiment config (model + tokenizer configs)
├── model_and_optim/     # Sharded DCP files (*.distcp)
└── train/               # Training state (rank*.pt)
```

The scripts detect this format via the presence of `model_and_optim/` and a `config.json` with a
`model` key. Conversion uses `src/examples/huggingface/convert_checkpoint_to_hf.py` from the
OLMo-core repo and requires `olmo_core` to be installed (`pip install -e '.[all]'`).

---

## `infer_transformers.py` — HuggingFace Transformers

**Install:**
```bash
pip install transformers torch
```

**Run:**
```bash
# Native OLMo-core checkpoint (conversion runs automatically on first call):
python infer_transformers.py \
    --model /home/user/checkpoints/olmo3-7b/job-10104/step1000

# Save the converted checkpoint to a specific location:
python infer_transformers.py \
    --model /path/to/step1000 \
    --hf-output-dir /path/to/hf_ckpt

# Re-use an already-converted checkpoint (skip re-conversion):
python infer_transformers.py \
    --model /path/to/step1000 \
    --hf-output-dir /path/to/hf_ckpt \
    --skip-conversion

# HF Hub model, CUDA, custom prompts:
python infer_transformers.py \
    --model allenai/OLMo-2-1124-7B \
    --prompts "Language modeling is" "The capital of France is" \
    --max-new-tokens 200 \
    --temperature 0.8 \
    --top-p 0.9 \
    --device cuda \
    --dtype bfloat16

# Spread across all GPUs automatically:
python infer_transformers.py --model allenai/OLMo-2-1124-32B --device auto --dtype bfloat16
```

**Key options:**

| Flag | Default | Description |
|---|---|---|
| `--model` | *(required)* | OLMo-core ckpt path, HF ckpt path, or Hub model ID |
| `--hf-output-dir` | `<model>/hf_converted/` | Where to cache the converted HF checkpoint |
| `--skip-conversion` | `False` | Skip conversion if converted dir already exists |
| `--prompts` | `"Language modeling is "` | One or more prompt strings |
| `--max-new-tokens` | `100` | Max tokens to generate |
| `--temperature` | `1.0` | Sampling temperature |
| `--top-p` | `0.7` | Nucleus sampling threshold |
| `--do-sample` / `--no-do-sample` | `True` | Sampling vs. greedy decoding |
| `--device` | `cpu` | `cpu`, `cuda`, or `auto` |
| `--dtype` | `auto` | `auto`, `float32`, `float16`, `bfloat16` |

---

## `infer_vllm.py` — vLLM

**Install:**
```bash
pip install vllm
```

**Run:**
```bash
# Native OLMo-core checkpoint (conversion runs automatically on first call):
python infer_vllm.py \
    --model /home/user/checkpoints/olmo3-7b/job-10104/step1000

# Save the converted checkpoint to a specific location:
python infer_vllm.py \
    --model /path/to/step1000 \
    --hf-output-dir /path/to/hf_ckpt

# HF Hub model, custom settings, 4-GPU tensor parallelism:
python infer_vllm.py \
    --model allenai/OLMo-2-1124-32B \
    --prompts "Language modeling is" "The capital of France is" \
    --max-tokens 200 \
    --temperature 0.8 \
    --top-p 0.9 \
    --tensor-parallel-size 4 \
    --dtype bfloat16
```

**Key options:**

| Flag | Default | Description |
|---|---|---|
| `--model` | *(required)* | OLMo-core ckpt path, HF ckpt path, or Hub model ID |
| `--hf-output-dir` | `<model>/hf_converted/` | Where to cache the converted HF checkpoint |
| `--skip-conversion` | `False` | Skip conversion if converted dir already exists |
| `--prompts` | `"Language modeling is"` | One or more prompt strings |
| `--max-tokens` | `100` | Max tokens to generate |
| `--temperature` | `1.0` | Sampling temperature (0 = greedy) |
| `--top-p` | `0.7` | Nucleus sampling threshold |
| `--tensor-parallel-size` | `1` | Number of GPUs for tensor parallelism |
| `--dtype` | `auto` | `auto`, `float32`, `float16`, `bfloat16` |
| `--trust-remote-code` | `False` | Allow custom model code |
