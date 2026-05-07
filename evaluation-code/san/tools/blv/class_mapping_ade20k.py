#!/usr/bin/env python
"""Print the current ADE20K to BLV remapping table."""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from blv_pipeline.class_mapping_ade20k import DEFAULT_ADE20K_TO_BLV


def main() -> None:
    print(json.dumps(DEFAULT_ADE20K_TO_BLV, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()

