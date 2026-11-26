"""Hardware/probe suites are explicitly opted in, never part of default CI."""
import os
from pathlib import Path

collect_ignore = ([] if os.environ.get("FLEXKV_RUN_GDR_TESTS") == "1" else
                  [path.name for path in Path(__file__).parent.glob("test_*.py")])
