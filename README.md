# LeClim

LeClim adapts ClimateBench data for LeWorldModel experiments.

It converts ClimateBench NetCDF simulations into the HDF5 episode format used by `stable-worldmodel`, trains with the upstream LeWorldModel `train.py`, and provides optional tools for decoding latent representations back into gridded surface temperature (`tas`) for ClimateBench-style evaluation.

## What The Data Represents

ClimateBench contains simulated climate data from Earth system models. In this workflow, each training example represents one simulated year: a global surface-air-temperature field paired with greenhouse-gas forcing for that year.

The model is asked to learn how simulated global temperature patterns evolve as forcing changes.

## What Is Included

- `climatebench_lewm/`: converter, decoder, evaluator, and CLI entry points
- `config/`: minimal LeWorldModel Hydra configs for ClimateBench HDF5 data
- `scripts/`: small utility scripts for conversion, rollout diagnostics, and plotting
- `tests/`: schema, metric, import, and optional real-data integration tests
- `external/PIN_LEWM.md`: the upstream LeWorldModel commit used for reproducibility

Raw data, checkpoints, generated HDF5 files, and experiment outputs are intentionally excluded.

## Install

```bash
cd LeClim
python3 -m venv .venv
. .venv/bin/activate
pip install -e ".[dev]"
```

Optional extras:

```bash
pip install -e ".[decoder]"      # decoder/evaluation tools
pip install -e ".[climate-viz]"  # map plotting
```

## Convert ClimateBench To LeWM HDF5

Download and extract ClimateBench `train_val.tar.gz` from Zenodo, then point `CLIMATEBENCH_ROOT` at the extracted directory containing `inputs_*.nc` and `outputs_*.nc`.

```bash
export CLIMATEBENCH_ROOT=/path/to/train_val
export STABLEWM_HOME=/path/to/stablewm_cache

climatebench-to-lewm-hdf5 --verbose --overwrite
```

The converter writes `climatebench_train.h5` plus a matching metadata JSON.

## Train LeWorldModel

Use the upstream LeWorldModel repository pinned in `external/PIN_LEWM.md`. Copy or symlink the configs in `config/train/` into the upstream config tree, then run from that repository:

```bash
export STABLEWM_HOME=/path/to/stablewm_cache
export PYTHONPATH="$(pwd)"
python train.py --config-name=climatebench_lewm
```

For CPU-only runs, use conservative DataLoader overrides:

```bash
python train.py --config-name=climatebench_lewm \
  trainer.accelerator=cpu trainer.precision=32 \
  loader.batch_size=8 \
  loader.num_workers=0 loader.prefetch_factor=null \
  loader.persistent_workers=false loader.pin_memory=false
```

## Decode And Evaluate `tas`

LeWorldModel predicts latent states. It does not directly emit temperature maps. To score gridded `tas`, train a small decoder on top of a frozen LeWM encoder:

```bash
train-climatebench-decoder \
  --h5 "$STABLEWM_HOME/climatebench_train.h5" \
  --lewm-object /path/to/lewm_object.ckpt \
  --epochs 15 \
  --batch-size 8 \
  --out outputs/decoder_bundle
```

Then evaluate against ClimateBench NetCDF files:

```bash
evaluate-climatebench \
  --climatebench-root /path/to/test_zenodo \
  --experiment ssp245 \
  --year-min 2080 --year-max 2100 \
  --meta "$STABLEWM_HOME/climatebench_train.meta.json" \
  --lewm-object /path/to/lewm_object.ckpt \
  --decoder-bundle outputs/decoder_bundle \
  --out-json outputs/eval_metrics_test_ssp245_2080_2100.json
```

## Tests

```bash
pytest -q
```

The real-data roundtrip test runs only when `CLIMATEBENCH_ROOT` points to extracted ClimateBench NetCDF files.
