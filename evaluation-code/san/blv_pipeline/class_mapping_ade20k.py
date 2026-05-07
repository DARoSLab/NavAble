"""ADE20K to BLV class remapping for the zero-shot baseline."""

from typing import Dict

import numpy as np

from .constants import BLV_NAME_TO_ID, IGNORE_INDEX

# ADE20K does not have dedicated labels for most BLV categories, so this
# mapping is intentionally conservative. It is a usable baseline, but it
# should be revisited before reporting final paper numbers.
DEFAULT_ADE20K_TO_BLV: Dict[int, int] = {
    14: BLV_NAME_TO_ID['door'],           # door
    43: BLV_NAME_TO_ID['bus_stop_sign'],  # signboard, coarse proxy
    58: BLV_NAME_TO_ID['door'],           # screen door
    136: BLV_NAME_TO_ID['traffic_signal'],  # traffic light
}


def remap_semantic_mask(
    prediction: np.ndarray,
    mapping: Dict[int, int] = DEFAULT_ADE20K_TO_BLV,
    ignore_index: int = IGNORE_INDEX,
) -> np.ndarray:
    """Remap an integer semantic mask into the BLV label space."""

    output = np.full(prediction.shape, ignore_index, dtype=np.uint8)
    for source_id, target_id in mapping.items():
        output[prediction == source_id] = target_id
    return output

