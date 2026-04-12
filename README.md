# Robot Learning Project

This repo contains the RL side of the project:

- `rl/` holds the actual SAC fine-tuning implementation.
- `train_rl.py` is the main entrypoint for SAC-DAPG / SAC-finetune runs.
- `evaluate_checkpoints.py` evaluates a robomimic checkpoint on its environment.
- `validate_sac_robosuite.py` is a standalone robosuite SAC sanity check.
- `clean_sac.py` and `sac_finetune.py` are earlier experiment variants that I kept around for reference.

The code depends on the sibling repos in this workspace:

- `../robomimic`
- `../mimicgen`
- `../robosuite-task-zoo`

## How it works

The main training path is:

1. Load a BC-RNN checkpoint with robomimic.
2. Extract the policy network and its encoder.
3. Build a replay buffer that stores short observation/action sequences.
4. Seed the buffer from demonstration data in the MimicGen dataset.
5. Run SAC updates with a BC regularizer on demo batches for DAPG-style finetuning.
6. Periodically evaluate the current policy in the original robosuite environment.

The lower-level logic is split like this:

- `rl/actor.py` handles observation encoding and per-step policy sampling.
- `rl/replay.py` handles sequence replay and demo loading from HDF5.
- `rl/sac.py` implements the twin-Q SAC update and the BC-regularized variant.
- `rl/evaluation.py` runs deterministic rollouts and reports success rate / return.
- `rl/trainer.py` wires everything together into the training loop.

## Helper Scripts

The `scripts/` directory contains thin wrappers that set the environment variables we keep needing in this workspace:

- `scripts/eval_bc_checkpoint.sh`
- `scripts/run_dapg.sh`
- `scripts/run_robosuite_smoke.sh`

## Running It

From this repo root:

```bash
scripts/eval_bc_checkpoint.sh <checkpoint-path>
scripts/run_dapg.sh <checkpoint-path>
scripts/run_robosuite_smoke.sh Lift
```

If you want to run the Python entrypoints directly, make sure `PYTHONPATH` includes this repo and the sibling repos, and set `NUMBA_DISABLE_JIT=1` in this workspace.

