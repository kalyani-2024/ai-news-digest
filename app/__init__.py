"""Load configuration from the project-root .env before any submodule runs.

dotenv's default lookup walks upward from the importing module's directory,
so a stray .env inside app/ would shadow the real one at the project root.
Pinning the path here removes that ambiguity: this runs first for every
`app.*` import, and later load_dotenv() calls will not override what is set.
"""
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent

load_dotenv(PROJECT_ROOT / ".env")
