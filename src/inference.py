
import numpy as np


class OverlapAccumulator:
    """
    Reconstruct a full-size prediction raster by averaging
    overlapping prediction windows.

    Reusable for segmentation and dense regression.
    """

    def __init__(self, height, width):
        self.height = int(height)
        self.width = int(width)

        self.total = np.zeros(
            (self.height, self.width),
            dtype=np.float32,
        )

        self.count = np.zeros(
            (self.height, self.width),
            dtype=np.uint32,
        )

    def add(self, patch, row_off, col_off):
        patch = np.asarray(patch, dtype=np.float32)

        if patch.ndim != 2:
            raise ValueError("Expected a 2D prediction patch")

        h, w = patch.shape

        r0 = int(row_off)
        c0 = int(col_off)
        r1 = r0 + h
        c1 = c0 + w

        if (
            r0 < 0 or c0 < 0
            or r1 > self.height
            or c1 > self.width
        ):
            raise ValueError("Patch exceeds raster boundaries")

        if not np.isfinite(patch).all():
            raise ValueError("Non-finite patch predictions")

        self.total[r0:r1, c0:c1] += patch
        self.count[r0:r1, c0:c1] += 1

    def finalize(self, valid_mask=None):
        if valid_mask is None:
            valid_mask = np.ones(
                (self.height, self.width),
                dtype=bool,
            )
        else:
            valid_mask = np.asarray(
                valid_mask,
                dtype=bool,
            )

        if valid_mask.shape != self.total.shape:
            raise ValueError("Validity mask shape mismatch")

        if np.any(valid_mask & (self.count == 0)):
            raise ValueError(
                "Some valid pixels have no predictions. "
                "Check the patch index and spatial coverage."
            )

        output = np.full(
            self.total.shape,
            np.nan,
            dtype=np.float32,
        )

        np.divide(
            self.total,
            self.count,
            out=output,
            where=self.count > 0,
        )

        output[~valid_mask] = np.nan

        return output
