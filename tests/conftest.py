"""Test configuration.

The project uses a flat module layout (``utils.py``, ``config.py``, ...), so the
repository root is added to ``sys.path``. Only modules that do not import vLLM at
import time are tested here (``utils``, ``config`` and ``data_handler``).
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# The repository root contains an ``__init__.py`` that imports ``pipeline`` (and
# therefore vLLM). Because of that file pytest treats the root directory as a
# package and, when it sets the package up, imports ``__init__.py``. Skip that step
# so the tests do not require vLLM; the modules under test are imported directly.
pytest.Package.setup = lambda self: None
