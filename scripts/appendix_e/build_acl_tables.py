from __future__ import annotations

import sys
from pathlib import Path

sys.dont_write_bytecode = True
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from build_appendix_e_tables import main


if __name__ == "__main__":
    print("[INFO] build_acl_tables.py is a compatibility wrapper; using build_appendix_e_tables.py.")
    raise SystemExit(main())
