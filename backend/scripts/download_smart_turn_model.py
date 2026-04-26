"""Download the Smart Turn v3.2 ONNX model.

Run from the backend/ directory:

    python scripts/download_smart_turn_model.py
"""

import sys
import urllib.request
from pathlib import Path

_MODEL_URL = (
    "https://github.com/nitishkmr005/LucidAI/raw/main/backend/models/smart-turn-v3.2-cpu.onnx"
)
_DEST = Path("models/smart_turn/smart-turn-v3.2-cpu.onnx")


def main() -> None:
    _DEST.parent.mkdir(parents=True, exist_ok=True)
    if _DEST.exists():
        print(f"Model already present at {_DEST}")
        return

    print(f"Downloading Smart Turn model from {_MODEL_URL} …")
    try:
        urllib.request.urlretrieve(_MODEL_URL, _DEST)
        print(f"Saved to {_DEST} ({_DEST.stat().st_size // 1024} KB)")
    except Exception as err:
        print(f"Download failed: {err}", file=sys.stderr)
        print(
            "Please download the model manually and place it at backend/models/smart_turn/smart-turn-v3.2-cpu.onnx",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
