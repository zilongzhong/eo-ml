# Fire figures

The updated Landsat exports follow native order:
`SR_B1, SR_B2, SR_B3, SR_B4, SR_B5, SR_B6, SR_B7, ST_B10`.
All three `Landsat89_2ndMIN_SR_MAX_ST_DN_2023_region*.tif` files were
inspected: they declare these exact band names, contain eight uint16 DN
bands, and align with their respective single-band binary NBAC labels.
RGB uses **SR_B4, SR_B3, SR_B2**: raster bands **4, 3, 2** (one-based),
or array indices **[3, 2, 1]** (zero-based). Display code resolves these
sensor bands from TIFF descriptions, accepting both `SR_B*` names and
native-order color names `Coastal, Blue, Green, Red, ...`.
Unnamed rasters fall back to the configured band order; ambiguous named
metadata raises an error. All dataset, evaluation, and TensorBoard
visualizations use this shared selector. Described legacy exports in
`Coastal, Red, Green, Blue, ...` order remain supported by their metadata.

Changing the source band order also changes model inputs. Training-only
normalization statistics and the patch index have been rebuilt for the
new exports. Preprocessing and dataset loading now validate the declared
band order before consuming the channels. Legacy checkpoints still
correspond to the previous data and channel positions, so training must
be rerun before generating predictions for the updated data. Evaluation
checks that checkpoint training source filenames match current statistics.
Existing checkpoints and evaluation products are preserved. The same
source check applies to `--plots-only`, so old predictions cannot be
redrawn beside RGB from these replacement exports as if they were a
matching evaluation.

`src/data/visualization.py` supplies the same RGB renderer to dataset,
evaluation, and TensorBoard figures. Raw DN is scaled once with
`DN * 0.0000275 - 0.2`; dataset and TensorBoard images are already scaled,
so their z-score normalization is undone instead. The renderer maps fixed
physical reflectance limits into display brightness with gamma correction:

```text
tone = clip((reflectance - min) / (max - min), 0, 1) ** (1 / gamma)
RGB = shadow_floor + (1 - shadow_floor) * tone  # valid pixels only
```

The defaults are `min: 0`, `max: 0.10`, `gamma: 2.2`, `shadow_floor: 0.08`,
shared by all RGB channels, patches, and regions. Gamma greater than 1
brightens positive reflectance. The shadow floor raises valid zero and
negative values to charcoal instead of pure black. It cannot recover
distinctions lost through clipping. Set `shadow_floor: 0` to disable this
display adjustment; the previous rendering also used `gamma: 1.8`. There is no
percentile stretch or per-scene brightness adjustment. Missing/non-finite
pixels stay black. Uniform patches retain their actual brightness.

Tune `visualization.rgb` in `configs/fire_segmentation.yaml` to change
brightness. To match GEE's `min`, `max`, and `gamma` settings, also disable
the extra shadow-floor adjustment with `shadow_floor: 0`.
The chosen max of 0.10 brightens these low-reflectance exports; it is a
display choice, not a change to the official calibration. The USGS
conversion is documented at
[Landsat Collection 2 Surface Reflectance](https://www.usgs.gov/landsat-missions/landsat-collection-2-surface-reflectance).

This changes display only; model inputs, normalization statistics,
checkpoints, predictions, and metrics retain their original channel order
and values.

The previous `MIN_SR` files contained many negative reflectance values.
For the updated `2ndMIN_SR` exports, the measured proportions of valid
pixels with all three RGB reflectances at/below zero are 0.71% in Region 1,
4.23% in Region 2, and 9.88% in Region 3 (previously 39.95% in Region 3).
The filenames describe second-minimum optical compositing; the export
script would be needed to verify the exact reducer and masking rules.
A minimum reducer selects the lowest observation independently
for each band across dates, making it sensitive to residual shadows and
low retrieval outliers. Negative retrievals can also occur over water.
The GEE export/masking code is needed to establish their exact causes here.
The display clips these values to its minimum tone and applies the shadow
floor; source values are unchanged. Adjusting the display cannot recover
a single-date photograph or ensure it matches a different composite in GEE.

## Burned-area probability

The U-Net returns one-channel logits from its final segmentation head.
Evaluation applies `torch.sigmoid(logits.float())` to each pixel, then
averages probabilities across overlapping patches to build the full raster.
The displayed map is this averaged probability, before the binary threshold
of 0.5. It is not a softmax map or a decoder feature map. A value such as
0.8 is a model score for the burned class, not a guarantee of calibrated
80% confidence.

Optical and thermal validity panels use blue for valid pixels and red for
missing pixels, with percentages and legends. A uniform blue panel marked
100% is expected when a selected patch has no missing data. These masks
describe DN fill validity, not cloud screening or reflectance quality.

From an ML container with project/data/output mounts, regenerate dataset figures:

```bash
docker exec -w /workspace eo-dev python -m src.data.check_fire_dataloader
```

When replacing the exports again, rebuild preprocessing before using
the Dataset (commands run from `/workspace` inside the container):

```bash
python -m src.data.compute_fire_stats
python -m src.data.build_fire_patch_index
python -m src.data.check_fire_dataloader
```

`python -m src.data.inspect_rasters` now uses the configured filenames and
prints the actual TIFF band descriptions rather than assumed labels.

Redraw evaluation figures from existing probability/prediction GeoTIFFs
and metrics, without GPU inference:

```bash
docker exec -w /workspace eo-dev python -m src.evaluate_fire \
  --plots-only --checkpoint /checkpoints/fire/fire_20261009_020921/best.pt
```

Both commands accept `--config`; the dataset command also accepts `--output`.
The previous figures are saved beside the updated files as
`dataset_samples_before_rgb_fix.png` and `region3_comparison_before_rgb_fix.png`.
The percentile-rendered versions are also preserved as
`dataset_samples_before_reflectance_display.png` and
`region3_comparison_before_reflectance_display.png`.
The rendering before shadow lifting is preserved in
`dataset_samples_before_shadow_lift.png` and `region3_comparison_before_shadow_lift.png`.
The statistics, patch index, patch summary, and dataset figure from the
previous exports are backed up beside their replacements with the
`_before_2ndmin` suffix.
