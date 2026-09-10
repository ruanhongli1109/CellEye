"""让 pytest 能从仓库根目录 import src。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
