"""CT entity generation and DRR utilities.

This file makes `CT_entities` a regular Python package so imports like
`from CT_entities.drr_triplet_pipeline import ...` work reliably.
"""

from __future__ import annotations

import os
import sys

# Many existing modules in this folder use legacy absolute imports like
# `from DRR_utils import ...` or `from Entity3D import Entity3D`.
# When `CT_entities` is imported as a package, those imports won't resolve
# unless this directory is also on `sys.path`.
_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
if _PKG_DIR not in sys.path:
	sys.path.insert(0, _PKG_DIR)
