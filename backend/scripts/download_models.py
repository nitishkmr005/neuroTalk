"""Download all runtime models to backend/models/.

Run from the backend/ directory:

    python scripts/download_models.py

Individual models can be skipped with flags:
    python scripts/download_models.py --skip-stt --skip-vad
"""

import argparse
import sys
from pathlib import Path

_SMART_TURN_HF_REPO = "nitishkmr005/smart-turn"
_SMART_TURN_HF_FILE = "smart-turn-v3.2-cpu.onnx"

_LLM_HF_REPO = "bartowski/Llama-3.2-3B-Instruct-GGUF"
_LLM_HF_FILE = "Llama-3.2-3B-Instruct-Q4_K_M.gguf"
_LLM_DEST = Path("models/llm") / _LLM_HF_FILE


def _bar(block: int, block_size: int, total: int) -> None:
    downloaded = block * block_size
    if total > 0:
        pct = min(100, downloaded * 100 // total)
        mb = downloaded / 1_048_576
        total_mb = total / 1_048_576
        print(f"\r  {pct:3d}%  {mb:.1f} / {total_mb:.1f} MB", end="", flush=True)


def download_stt() -> None:
    dest = Path("models/stt")
    if (dest / "model.bin").exists():
        print("STT model already present — skipping.")
        return
    dest.mkdir(parents=True, exist_ok=True)
    print("Downloading STT model (faster-whisper small.en) …")
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id="Systran/faster-whisper-small.en",
            local_dir=str(dest),
            ignore_patterns=["*.msgpack", "*.h5", "flax_model*", "tf_model*"],
        )
        print(f"  Saved to {dest}")
    except Exception as err:
        print(f"  Failed: {err}", file=sys.stderr)
        print("  Install huggingface_hub: uv pip install huggingface_hub", file=sys.stderr)
        sys.exit(1)


def download_vad() -> None:
    dest = Path("models/vad/silero_vad.jit")
    if dest.exists():
        print("VAD model already present — skipping.")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print("Extracting VAD model from silero_vad package …")
    try:
        import silero_vad as _sv
        import shutil
        pkg_dir = Path(_sv.__file__).parent
        bundled = pkg_dir / "data" / "silero_vad.jit"
        if not bundled.exists():
            # newer versions store it elsewhere
            candidates = list(pkg_dir.rglob("silero_vad.jit"))
            if not candidates:
                raise FileNotFoundError("silero_vad.jit not found in package")
            bundled = candidates[0]
        shutil.copy2(bundled, dest)
        print(f"  Saved to {dest} ({dest.stat().st_size // 1024} KB)")
    except ImportError:
        print("  silero_vad not installed — run: uv sync --group vad", file=sys.stderr)
        sys.exit(1)
    except Exception as err:
        print(f"  Failed: {err}", file=sys.stderr)
        sys.exit(1)


def download_kokoro() -> None:
    dest = Path("models/kokoro")
    if (dest / "kokoro-v1_0.safetensors").exists():
        print("Kokoro TTS model already present — skipping.")
        return
    dest.mkdir(parents=True, exist_ok=True)
    print("Downloading Kokoro TTS model (mlx-community/Kokoro-82M-bf16, ~330 MB) …")
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id="mlx-community/Kokoro-82M-bf16",
            local_dir=str(dest),
        )
        print(f"  Saved to {dest}")
    except Exception as err:
        print(f"  Failed: {err}", file=sys.stderr)
        print("  Install huggingface_hub: uv pip install huggingface_hub", file=sys.stderr)
        sys.exit(1)


def download_smart_turn() -> None:
    onnx_dest = Path("models/smart_turn/smart-turn-v3.2-cpu.onnx")
    extractor_dest = Path("models/smart_turn/whisper-base")

    if not onnx_dest.exists():
        onnx_dest.parent.mkdir(parents=True, exist_ok=True)
        print("Downloading Smart Turn ONNX model …")
        try:
            from huggingface_hub import hf_hub_download
            hf_hub_download(
                repo_id=_SMART_TURN_HF_REPO,
                filename=_SMART_TURN_HF_FILE,
                local_dir=str(onnx_dest.parent),
            )
            print(f"  Saved to {onnx_dest} ({onnx_dest.stat().st_size // 1024} KB)")
        except Exception as err:
            print(f"  Failed: {err}", file=sys.stderr)
            sys.exit(1)
    else:
        print("Smart Turn ONNX model already present — skipping.")

    if not (extractor_dest / "preprocessor_config.json").exists():
        extractor_dest.mkdir(parents=True, exist_ok=True)
        print("Downloading Whisper-base feature extractor …")
        try:
            from transformers import WhisperFeatureExtractor
            extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-base")
            extractor.save_pretrained(str(extractor_dest))
            print(f"  Saved to {extractor_dest}")
        except Exception as err:
            print(f"  Failed: {err}", file=sys.stderr)
            sys.exit(1)
    else:
        print("Whisper-base extractor already present — skipping.")


def download_llm() -> None:
    if _LLM_DEST.exists():
        print(f"LLM model already present — skipping. ({_LLM_DEST.stat().st_size / 1_048_576:.0f} MB)")
        return
    _LLM_DEST.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading Llama 3.2 3B Instruct Q4_K_M GGUF (~2.0 GB) …")
    try:
        from huggingface_hub import hf_hub_download
        hf_hub_download(
            repo_id=_LLM_HF_REPO,
            filename=_LLM_HF_FILE,
            local_dir=str(_LLM_DEST.parent),
        )
        print(f"  Saved to {_LLM_DEST} ({_LLM_DEST.stat().st_size / 1_048_576:.0f} MB)")
    except Exception as err:
        print(f"  Failed: {err}", file=sys.stderr)
        print("  Install huggingface_hub: uv pip install huggingface_hub", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download NeuroTalk runtime models.")
    parser.add_argument("--skip-stt", action="store_true")
    parser.add_argument("--skip-vad", action="store_true")
    parser.add_argument("--skip-kokoro", action="store_true")
    parser.add_argument("--skip-smart-turn", action="store_true")
    parser.add_argument("--skip-llm", action="store_true")
    parser.add_argument("--only-llm", action="store_true", help="Download only the LLM GGUF model.")
    args = parser.parse_args()

    if args.only_llm:
        download_llm()
        print("\nLLM model ready.")
        return

    if not args.skip_stt:
        download_stt()
    if not args.skip_vad:
        download_vad()
    if not args.skip_kokoro:
        download_kokoro()
    if not args.skip_smart_turn:
        download_smart_turn()
    if not args.skip_llm:
        download_llm()

    print("\nAll models ready.")


if __name__ == "__main__":
    main()
