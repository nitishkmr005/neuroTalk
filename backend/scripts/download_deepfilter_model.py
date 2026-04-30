"""Download DeepFilterNet3 model weights to models/deepfilter/.

Run from the backend/ directory:

    python scripts/download_deepfilter_model.py

Requires the deepfilter dependency group:

    uv sync --group deepfilter
"""

import shutil
import sys
from pathlib import Path

_DEST = Path("models/deepfilter")


def _cache_dir() -> Path:
    """Return the platform-specific DeepFilterNet cache directory."""
    try:
        from platformdirs import user_cache_dir
        return Path(user_cache_dir("DeepFilterNet", appauthor=False)) / "DeepFilterNet3"
    except ImportError:
        import platform
        if platform.system() == "Darwin":
            return Path.home() / "Library" / "Caches" / "DeepFilterNet" / "DeepFilterNet3"
        return Path.home() / ".cache" / "DeepFilterNet" / "DeepFilterNet3"


def main() -> None:
    try:
        from df.enhance import init_df
    except ImportError:
        print(
            "deepfilternet is not installed. Run: uv sync --group deepfilter",
            file=sys.stderr,
        )
        sys.exit(1)

    _DEST.mkdir(parents=True, exist_ok=True)

    print("Downloading DeepFilterNet3 model (to system cache) …")
    try:
        _, df_state, _ = init_df()
    except Exception as err:
        print(f"Download failed: {err}", file=sys.stderr)
        sys.exit(1)

    cache = _cache_dir()
    if not cache.exists():
        print(f"Cache dir not found at {cache}", file=sys.stderr)
        sys.exit(1)

    print(f"Copying model from {cache} → {_DEST} …")
    shutil.copytree(cache, _DEST, dirs_exist_ok=True)
    print(f"Model ready at {_DEST}  (sr={df_state.sr()} Hz)")


if __name__ == "__main__":
    main()
