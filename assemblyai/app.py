from pathlib import Path
import sys

# Keep legacy launcher working when running from repository root without installation.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from voice_ai_explorer.ui import run_app


run_app()
