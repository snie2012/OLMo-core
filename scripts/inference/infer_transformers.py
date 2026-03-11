r"""
Run inference on a trained OLMo checkpoint using the HuggingFace Transformers library.

Accepts either:
  - A pre-converted HuggingFace checkpoint directory (contains ``config.json`` with an
    ``architectures`` field), or
  - A raw OLMo-core checkpoint directory (contains ``config.json`` with a ``model`` field and a
    ``model_and_optim/`` subdirectory). In this case the script first converts the checkpoint to
    HuggingFace format, saves it to ``--hf-output-dir`` (or ``<ckpt_dir>/hf_converted/``), and
    then runs inference from there.

Usage examples::

    # Raw OLMo-core checkpoint – conversion happens automatically:
    python infer_transformers.py \
        --model /home/shaoliang/olmo_checkpoints/olmo3-7b/job-10104/step10000

    # Same, but save the converted checkpoint to a custom location:
    python infer_transformers.py \
        --model /home/shaoliang/olmo_checkpoints/olmo3-7b/job-10104/step10000 \
        --hf-output-dir /path/to/hf_ckpt

    # Pre-converted HuggingFace checkpoint or Hub model ID:
    python infer_transformers.py --model allenai/OLMo-2-1124-7B --device cuda

    # Custom prompts and generation settings:
    python infer_transformers.py \
        --model /home/shaoliang/olmo_checkpoints/olmo3-7b/job-10104/step60000 \
        --prompts "Language modeling is" "The capital of France is" \
        --max-new-tokens 200 \
        --temperature 0.8 \
        --top-p 0.9 \
        --device cuda \
        --dtype bfloat16

    # Read prompts from a file and write results to a JSONL file:
    python infer_transformers.py \
        --model /home/shaoliang/olmo_checkpoints/olmo3-7b/job-10104/step60000 \
        --prompts-file inputs/prompts_1000.txt \
        --output-file outputs/results.jsonl \
        --device cuda \
        --dtype bfloat16
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# OLMo-core checkpoint detection & conversion
# ---------------------------------------------------------------------------


def _is_olmo_core_checkpoint(path: str) -> bool:
    """Return True if *path* looks like a native OLMo-core checkpoint directory."""
    p = Path(path)
    config_file = p / "config.json"
    model_dir = p / "model_and_optim"
    if not config_file.is_file():
        return False
    try:
        with config_file.open() as f:
            cfg = json.load(f)
        return "model" in cfg and model_dir.is_dir()
    except Exception:
        return False


def _convert_to_hf(checkpoint_dir: str, hf_output_dir: str) -> str:
    """
    Convert a native OLMo-core checkpoint to HuggingFace format.

    Returns the path to the converted HF checkpoint directory.
    """
    # Import lazily so the script works even without olmo_core when loading HF ckpts.
    try:
        from olmo_core.config import DType
    except ImportError as exc:
        raise ImportError(
            "olmo_core is required to convert native checkpoints. "
            "Install it with: pip install -e '.[all]'"
        ) from exc

    # Locate the convert script that lives in this repo.
    script_path = (
        Path(__file__).resolve().parent.parent.parent
        / "src"
        / "examples"
        / "huggingface"
        / "convert_checkpoint_to_hf.py"
    )
    if not script_path.is_file():
        raise FileNotFoundError(
            f"Could not find conversion script at {script_path}. "
            "Make sure you are running from the OLMo-core repository."
        )

    # Use the library function directly (same as the script does) to avoid a
    # subprocess round-trip and to get proper logging.
    sys.path.insert(0, str(script_path.parent))
    from convert_checkpoint_to_hf import convert_checkpoint_to_hf, load_config

    config_dict = load_config(checkpoint_dir)
    if config_dict is None:
        raise RuntimeError(f"Could not load OLMo-core config from {checkpoint_dir}")

    transformer_config_dict = config_dict["model"]
    tokenizer_config_dict = config_dict.get("dataset", {}).get("tokenizer")
    if tokenizer_config_dict is None:
        raise RuntimeError("Tokenizer config not found in checkpoint config.json")

    log.info(f"Converting OLMo-core checkpoint: {checkpoint_dir}")
    log.info(f"HF output directory: {hf_output_dir}")

    convert_checkpoint_to_hf(
        original_checkpoint_path=checkpoint_dir,
        output_path=hf_output_dir,
        transformer_config_dict=transformer_config_dict,
        tokenizer_config_dict=tokenizer_config_dict,
        dtype=DType.bfloat16,
        validate=False,
    )
    return hf_output_dir


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run inference on an OLMo checkpoint with HuggingFace Transformers.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help=(
            "Path to an OLMo-core checkpoint directory, a pre-converted HuggingFace "
            "checkpoint directory, or a HuggingFace Hub model ID."
        ),
    )
    parser.add_argument(
        "--hf-output-dir",
        type=str,
        default=None,
        help=(
            "Where to save the converted HuggingFace checkpoint when --model is a native "
            "OLMo-core checkpoint. Defaults to <model>/hf_converted/."
        ),
    )
    parser.add_argument(
        "--prompts",
        type=str,
        nargs="+",
        default=None,
        help="One or more prompt strings to generate from. Mutually exclusive with --prompts-file.",
    )
    parser.add_argument(
        "--prompts-file",
        type=str,
        default=None,
        help=(
            "Path to a plain-text file with one prompt per line. "
            "Defaults to inputs/prompts_1000.txt when --prompts is not set. "
            "Mutually exclusive with --prompts."
        ),
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help=(
            "Path to write results as a JSONL file with 'input' and 'text' fields. "
            "Defaults to outputs/results_<timestamp>.jsonl. "
            "When not set and --prompts is given inline, results are printed to stdout."
        ),
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help=(
            "Maximum number of new tokens to generate. Defaults to 512. "
            "For pretrained (base) checkpoints the only natural stop is EOS "
            "(<|endoftext|>), so set this to avoid runaway generation."
        ),
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature (higher = more random).",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.7,
        help="Nucleus sampling probability threshold.",
    )
    parser.add_argument(
        "--do-sample",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to use sampling (True) or greedy decoding (False).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "auto"],
        help=(
            "Device to run inference on. Use 'auto' to let Transformers decide "
            "(spreads across available GPUs)."
        ),
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "float32", "float16", "bfloat16"],
        help="Torch dtype for model weights. 'auto' defers to the model config.",
    )
    parser.add_argument(
        "--skip-conversion",
        action="store_true",
        default=False,
        help=(
            "Skip conversion even if a converted checkpoint already exists at "
            "--hf-output-dir. Has no effect when --model is already in HF format."
        ),
    )
    return parser.parse_args()


def resolve_dtype(dtype_str: str):
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "auto": "auto",
    }
    return mapping[dtype_str]


def as_model_id(model_path: str):
    """Return a Path when the string points to a local directory.

    Newer huggingface_hub versions run repo-ID validation on plain strings and
    reject absolute paths before checking whether they exist locally.  Passing a
    Path object bypasses that validation entirely.
    """
    p = Path(model_path)
    return p if p.exists() else model_path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SCRIPT_DIR = Path(__file__).resolve().parent
_DEFAULT_INPUTS_DIR = _SCRIPT_DIR / "inputs"
_DEFAULT_OUTPUTS_DIR = _SCRIPT_DIR / "outputs"
_DEFAULT_PROMPTS_FILE = _DEFAULT_INPUTS_DIR / "prompts_1000.txt"


def _checkpoint_name(model_path: str) -> str:
    """Derive a short, filesystem-safe name from a checkpoint path or Hub model ID.

    Examples:
        /path/to/olmo3-7b/job-10115/step10000  ->  job-10115_step10000
        allenai/OLMo-2-1124-7B                 ->  allenai_OLMo-2-1124-7B
    """
    p = Path(model_path)
    if p.exists():
        parts = p.parts
        name = "_".join(parts[-2:]) if len(parts) >= 2 else parts[-1]
    else:
        name = model_path.replace("/", "_")
    return name.replace(" ", "_")


def _default_output_file(model_path: str) -> Path:
    """Return a timestamped default output path, e.g. outputs/results_job-10115_step10000_20260309_143022.jsonl."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt = _checkpoint_name(model_path)
    return _DEFAULT_OUTPUTS_DIR / f"results_{ckpt}_{ts}.jsonl"


def _load_prompts(args: argparse.Namespace) -> list:
    """Return the list of prompts from --prompts, --prompts-file, or the default file."""
    if args.prompts is not None and args.prompts_file is not None:
        raise ValueError("--prompts and --prompts-file are mutually exclusive.")
    if args.prompts is not None:
        return args.prompts
    prompts_path = Path(args.prompts_file) if args.prompts_file else _DEFAULT_PROMPTS_FILE
    lines = [line.rstrip("\n") for line in prompts_path.read_text().splitlines() if line.strip()]
    log.info(f"Loaded {len(lines)} prompts from {prompts_path}")
    return lines


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()

    model_path = args.model

    # --- Convert if necessary ---
    if _is_olmo_core_checkpoint(model_path):
        hf_output_dir = args.hf_output_dir or os.path.join(model_path, "hf_converted")
        if os.path.isdir(hf_output_dir) and not args.skip_conversion:
            log.info(
                f"Found existing converted checkpoint at {hf_output_dir}. "
                "Pass --skip-conversion to use it directly, or remove the directory to re-convert."
            )
        if not os.path.isdir(hf_output_dir):
            model_path = _convert_to_hf(model_path, hf_output_dir)
        else:
            model_path = hf_output_dir
        log.info(f"Using converted HF checkpoint: {model_path}")
    else:
        log.info(f"Treating {model_path!r} as a HuggingFace checkpoint/Hub model.")

    # --- Load model & tokenizer ---
    model_id = as_model_id(model_path)
    log.info(f"Loading tokenizer from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    log.info(f"Loading model from: {model_path}  (device={args.device}, dtype={args.dtype})")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=resolve_dtype(args.dtype),
        device_map=args.device if args.device == "auto" else None,
    )

    if args.device != "auto":
        model = model.to(args.device)

    model.eval()
    log.info(f"Model loaded. Parameters: {sum(p.numel() for p in model.parameters()):,}\n")

    prompts = _load_prompts(args)

    # --- Tokenize ---
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        return_token_type_ids=False,
        padding=True,
    )
    if args.device != "auto":
        inputs = {k: v.to(args.device) for k, v in inputs.items()}

    # --- Generate ---
    log.info(f"Generating for {len(prompts)} prompt(s)...\n")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.do_sample,
            temperature=args.temperature if args.do_sample else None,
            top_p=args.top_p if args.do_sample else None,
        )

    # --- Decode & write or print ---
    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    if args.output_file or args.prompts_file or args.prompts is None:
        out_path = Path(args.output_file) if args.output_file else _default_output_file(args.model)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        ckpt_name = _checkpoint_name(args.model)
        with out_path.open("w") as f:
            for prompt, text in zip(prompts, decoded):
                record = {"checkpoint": ckpt_name, "input": prompt, "text": text}
                f.write(json.dumps(record) + "\n")
        log.info(f"Wrote {len(decoded)} results to {out_path}")
    else:
        for i, (prompt, text) in enumerate(zip(prompts, decoded)):
            print(f"--- Prompt {i + 1} ---")
            print(f"Input:  {prompt!r}")
            print(f"Output: {text!r}")
            print()


if __name__ == "__main__":
    main()
