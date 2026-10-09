# Burned-area detection

The current configuration uses native-order Landsat 8/9 DN exports
(`SR_B1` through `SR_B7`, then `ST_B10`) and binary NBAC targets. Region 1
is used for training, Region 2 for validation/checkpoint selection, and
Region 3 is held out until final evaluation.

Run all tests from the host, using the existing image and the same mounts,
GPU, shared-memory size, and port mapping as the interactive Docker command:

```bash
bash /data/projects/eo-ml/scripts/run_docker.sh test
```

Retrain and evaluate everything with one host command:

```bash
bash /data/projects/eo-ml/scripts/run_docker.sh train
```

The training launcher rebuilds training-only normalization statistics and
the patch index, runs all pytest tests, generates the dataset figure,
trains a fresh model, then evaluates that run's `best.pt` on the configured
test regions. It stops immediately if any stage fails. Each run has a
unique name; existing checkpoints and evaluation folders are preserved.
The default is up to 30 epochs, with early stopping after eight epochs
without improved validation IoU. Change training settings in
`configs/fire_segmentation.yaml` before launching.

Inspect final results on the host:

```bash
python3 -m json.tool /data/outputs/fire/latest_run.json
```

This file points to the latest successfully completed pipeline. Each run
also saves `/data/outputs/fire/runs/<run>/summary.json`, `pipeline.log`,
`config.yaml`, and `pytest.xml`. A failed run records its stage and error
in its own summary and does not replace `latest_run.json`.

The evaluation directory is `/data/outputs/fire/evaluation/<run>/`, with:

- `metrics.csv` and `metrics.json`: held-out confusion counts and metrics.
- `region3/region3_probability.tif`: overlapping sigmoid probabilities averaged per pixel.
- `region3/region3_prediction.tif`: the averaged probability thresholded at 0.5.
- `region3/region3_comparison.png`: RGB, reference labels, predictions, and errors.

Best/latest checkpoints are in `/data/checkpoints/fire/<run>/`; training
history is in `/data/outputs/fire/tables/<run>_training_history.csv` and
TensorBoard logs are in `/data/outputs/fire/tensorboard/<run>/`.

If `eo-dev` is already running from your interactive Docker command, use
these commands from another host terminal:

```bash
docker exec -w /workspace eo-dev python -m pytest -q
docker exec -w /workspace eo-dev bash scripts/retrain_and_evaluate.sh
```

Or, inside that container:

```bash
cd /workspace
bash scripts/test.sh
bash scripts/retrain_and_evaluate.sh
```

Evaluate an explicitly selected checkpoint without retraining:

```bash
# Host, launching a new container:
bash /data/projects/eo-ml/scripts/run_docker.sh evaluate /checkpoints/fire/<run>/best.pt

# Inside an existing container:
bash scripts/evaluate.sh /checkpoints/fire/<run>/best.pt
```

Checkpoint paths use container paths (`/checkpoints`), not host paths.
Evaluation checks that the checkpoint and current data/statistics match.

Pipeline options can follow `train`, for example
`train --config configs/fire_segmentation.yaml --run-name fire_custom`.
`--skip-tests` skips the pytest stage; tests run by default. Test options
can follow `test`, for example `test --maxfail=1` or `test --cov=src`.

The host launcher defaults to container name `eo-dev`, image `eo-ml:latest`,
and data root `/data`. Override them with `EO_CONTAINER_NAME`, `EO_ML_IMAGE`,
and `EO_DATA_ROOT`. The launcher streams logs and exits on completion.
Container name `eo-dev` and host port 6006 must be available when launching
a new container; use `docker exec` when your interactive container is
already running. The scripts are mounted from the project, so rebuilding
the Docker image is unnecessary.

To view TensorBoard while an interactive `eo-dev` container is running:

```bash
docker exec -w /workspace eo-dev tensorboard \
  --logdir /outputs/fire/tensorboard --host 0.0.0.0 --port 6006
```

Open `http://127.0.0.1:6006` on the Docker host. The loopback port mapping
does not expose TensorBoard publicly.
