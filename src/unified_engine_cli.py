"""CLI entry point: ue-example --model <name> [--dir <path>] [--hf-token <tok>] [model args...]"""

import os
import shutil
import sys
from importlib import resources
from pathlib import Path

DEFAULT_CACHE = os.path.join(os.path.expanduser("~"), ".cache", "unified-engine")

# (package, script filename)
MODELS = {
    "gemma3": ("unified_engine_examples.gemma3", "gemma3_test.py"),
    "smolvlm2": ("unified_engine_examples.smolvlm2", "smolvlm2_test.py"),
}

# Models that require --image
REQUIRES_IMAGE = {"smolvlm2"}


def _extract_arg(args, flag):
    """Remove --flag <value> from args list, return (value, remaining_args)."""
    if flag not in args:
        return None, args
    idx = args.index(flag)
    if idx + 1 >= len(args):
        print(f"Error: {flag} requires a value")
        sys.exit(1)
    value = args[idx + 1]
    remaining = args[:idx] + args[idx + 2:]
    return value, remaining


def _setup_work_dir(package_name, work_dir):
    """Copy example files (scripts + configs) from the installed package to work_dir."""
    os.makedirs(work_dir, exist_ok=True)
    pkg_files = resources.files(package_name)
    for item in pkg_files.iterdir():
        name = item.name
        # Skip directories, __pycache__, .pyc, __init__.py
        if name == "__pycache__" or name.endswith(".pyc") or name == "__init__.py":
            continue
        if item.is_dir():
            continue
        dest = os.path.join(work_dir, name)
        # Only copy if missing (don't overwrite user edits)
        if not os.path.exists(dest):
            data = item.read_bytes()
            with open(dest, "wb") as f:
                f.write(data)


HELP_TEXT = """
Unified Engine Example Runner

Usage:
  ue-example --model <name> [options] [model-specific args...]

Options:
  --model <name>       Model to run (required)
  --dir <path>         Working directory (default: ~/.cache/unified-engine/<model>/)
  --hf-token <token>   HuggingFace token for gated models
  --help               Show this help message

Available models:
  gemma3               Gemma3 1B inference (prefill + decode)
  smolvlm2             SmolVLM2-500M vision-language inference (requires --image)

Examples:
  ue-example --model gemma3 --prompt "hello"
  ue-example --model gemma3 --hf-token hf_xxx --prompt "hello" --dev xdma0
  ue-example --model smolvlm2 --image path/to/yourimage.jpg --prompt "Describe this image."
  ue-example --model gemma3 --dir ./my-project --prompt "hello"

Model-specific args (--prompt, --dev, --cycle, etc.) are passed through directly.
Run with --model <name> --help to see model-specific options.
""".strip()


def main():
    args = sys.argv[1:]

    # Show help
    if "--help" in args or "-h" in args:
        # If --model is also present, let the help pass through to the example
        if "--model" not in args:
            print(HELP_TEXT)
            sys.exit(0)

    # Extract our flags before passing the rest to the example
    model, args = _extract_arg(args, "--model")
    hf_token, args = _extract_arg(args, "--hf-token")
    user_dir, args = _extract_arg(args, "--dir")

    if model is None:
        print(HELP_TEXT)
        sys.exit(1)

    if model not in MODELS:
        print(f"Unknown model: {model}")
        print(f"Available models: {', '.join(MODELS)}")
        sys.exit(1)

    # Check if model requires --image
    if model in REQUIRES_IMAGE and "--image" not in args:
        print(f"Error: --model {model} requires --image <path>")
        print(f"Example: ue-example --model {model} --image /path/to/image.jpg --prompt \"Describe this image.\"")
        sys.exit(1)

    if hf_token:
        os.environ["HF_TOKEN"] = hf_token

    package_name, script_name = MODELS[model]

    # Determine working directory
    if user_dir:
        work_dir = os.path.abspath(user_dir)
    else:
        work_dir = os.path.join(DEFAULT_CACHE, model)

    # Copy example files to working directory
    _setup_work_dir(package_name, work_dir)

    script_path = os.path.join(work_dir, script_name)
    if not os.path.isfile(script_path):
        print(f"Error: {script_path} not found after setup", file=sys.stderr)
        sys.exit(1)

    print(f"Working directory: {work_dir}")

    # Run the example script
    import runpy
    sys.argv = [script_path] + args
    try:
        runpy.run_path(script_path, run_name="__main__")
    except Exception as e:
        err = str(e)
        if "401" in err or "GatedRepoError" in type(e).__name__ or "Unauthorized" in err or "gated" in err.lower():
            print("\n" + "=" * 60)
            print("Authentication required for this model.")
            print("Either:")
            print("  1. Run: huggingface-cli login")
            print("  2. Pass token: ue-example --model {} --hf-token <your_token>".format(model))
            print("=" * 60)
            sys.exit(1)
        raise
