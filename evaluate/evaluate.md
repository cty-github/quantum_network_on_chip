# Evaluation Workflow

本目录保留四个推荐入口，对应项目的完整实验流程：

1. 训练 package layout 的 RL 模型。
2. 使用训练好的模型完成 package layout design / scalability，并与其他 layout 策略比较。
3. 基于已设计好的 package 运行 program routing，对比 QuNoC routing 与 baseline MECH。
4. 运行 ablation 实验。

旧的 shared-RL、单独 RL 训练配置、notebook 和 monolithic MECH runner 仅作为历史参考，不再作为推荐入口。

## 1. Train RL layout model

```bash
python evaluate/train_layout_model.py --config evaluate/train_layout_model.yaml
```

Canonical files:

- `evaluate/train_layout_model.py`
- `evaluate/train_layout_model.yaml`
- `qunoc/layout/training.py`
- `qunoc/layout/env.py`
- `qunoc/layout/agent.py`
- `qunoc/layout/schema.py`
- `qunoc/layout/artifacts.py`

`train_layout_model.yaml` controls the training domain:

- `domain.package_shapes`: default square packages from `40x40` to `160x160`; append `[x, y]` entries for additional target shapes.
- `domain.fab_model`: default fabrication parameters.
- `domain.fab_model_variants`: optional one-factor variations. Each entry overrides `fab_model` and becomes one additional training case. Entries are not combined with each other.
- `domain.fab_models`: optional explicit full fabrication model list; if present, it takes precedence over `fab_model_variants`.

Training output is a model-only artifact under `evaluate/result/layout_models/<run_name>/` by default:

- `model.zip`
- `model_manifest.json`
- `run_manifest.json`
- `training.log`
- `training_domain_inference.csv`

`training_domain_inference.csv` records deterministic policy inference over the configured training task domain whenever the training script writes a checkpoint or final model. It stores `score`, `fide_avg`, `fide_std`, and `compute_ratio` for each training task.

During an unfinished training run, the same directory may also contain resumable checkpoint artifacts:

- `checkpoint.zip`
- `checkpoint_manifest.json`

With the default `resume.resume_from_checkpoint: true`, rerunning the same command resumes from `checkpoint.zip` when it exists. After training reaches `ppo.target_timesteps` and writes the final `model.zip`, transient checkpoint files are removed.

The training script must not write package pickles, figures, or scalability CSVs.

## 2. Layout scalability / package design

```bash
python evaluate/test_layout_scalability.py --config evaluate/test_layout_scalability.yaml
```

Canonical files:

- `evaluate/test_layout_scalability.py`
- `evaluate/test_layout_scalability.yaml`

The script designs packages with `template`, `heuristic`, `greedy`, and optionally `rl`.

RL is inference-only in this step:

1. Train first with `evaluate/train_layout_model.py`.
2. Set `defaults.rl_model_path` in `test_layout_scalability.yaml` to the generated `model.zip`.
3. Add `rl` to the desired `design_strategy_to_test` list.

If `design_strategy: rl` is used without `rl_model_path`, the script fails instead of silently training.

Outputs are written under `evaluate/result/layout_scalability/<experiment>/`:

- `output.csv` with layout metrics (`Score`, `Fidelity`, `FideStd`)
- `<case>/package.pkl`
- optional `<case>/package.png`

These package paths are the inputs for routing experiments.

## 3. Routing and MECH baseline

Run QuNoC routing on the packages produced by Step 2:

```bash
python evaluate/test_routing_scalability.py --config evaluate/test_routing_scalability.yaml
```

Run the MECH baseline sweeps:

```bash
python evaluate/run_mech_routing.py --experiment all
```

Or run one MECH axis:

```bash
python evaluate/run_mech_routing.py --experiment package_size
python evaluate/run_mech_routing.py --experiment package_shape
python evaluate/run_mech_routing.py --experiment geometry
python evaluate/run_mech_routing.py --experiment inter_fidelity
python evaluate/run_mech_routing.py --experiment chiplet_sparsity
```

Canonical files:

- `evaluate/test_routing_scalability.py`
- `evaluate/test_routing_scalability.yaml`
- `evaluate/run_mech_routing.py`
- `baseline/mech_utils.py`
- `baseline/MECH/`

QuNoC routing consumes exact `package.pkl` paths from Step 2. If a required package is missing, run the matching layout scalability experiment first.

QuNoC routing outputs are written under `evaluate/result/routing_scalability/<experiment>/`:

- `output.csv`
- `<case>/exp_result.json`
- `<case>/qunoc_circuit.pkl`

MECH outputs are written under `evaluate/result/mechbaseline/<experiment>/`:

- `output.csv`
- `<case>_mech.json`

The old per-axis MECH wrappers remain as compatibility shims, but the recommended entry is now `evaluate/run_mech_routing.py`.

## 4. Ablation

```bash
python evaluate/run_ablation.py
```

Canonical file:

- `evaluate/run_ablation.py`

Ablation is intentionally separate from routing scalability. It reuses package artifacts from layout scalability and writes resumable case outputs plus a merged CSV under `evaluate/result/ablation/package_size/output.csv` by default.

## Plotting

Use the scripted plotting entry instead of the old notebook workflow:

```bash
python evaluate/plot_result_figures.py
```

Figures are written under `evaluate/figure/` by default, at the same level as `evaluate/result/`. The old notebook is kept under `evaluate/backup/plot_results.ipynb` as historical reference only.

## Legacy files

These files are retained under `evaluate/backup/` for reference or compatibility, but are not recommended as primary workflow entrypoints:

- `evaluate/backup/shared_rl_layout.yaml`: legacy shared-RL config replaced by train-only + inference-only layout flow.
- `evaluate/backup/RL_only_layout.yaml`: legacy inline-RL layout config; migrate to `train_layout_model.yaml` and `test_layout_scalability.yaml`.
- `evaluate/backup/main_exp.py`: legacy overall summary script; prefer canonical routing/MECH CSVs and `plot_result_figures.py`.
- `evaluate/backup/plot_results.ipynb`: legacy plotting notebook.
- `evaluate/backup/run_mech_routing_all.py`: legacy monolithic MECH runner replaced by `evaluate/run_mech_routing.py --experiment all`.
- `evaluate/backup/run_mech_routing_package_size.py`, `evaluate/backup/run_mech_routing_package_shape.py`, `evaluate/backup/run_mech_routing_geometry.py`, `evaluate/backup/run_mech_routing_inter_fidelity.py`: compatibility wrappers for the consolidated MECH CLI.

## Generated files

Generated local runtime files should not be committed:

- `evaluate/result/`
- `evaluate/figure/`
- `demo/result/`
- `logs/`
- `evaluate/logs/`
- `*.log`
- `*.console.log`
- `__pycache__/`
- `.pytest_cache/`

The `tests/` directory is source and should remain separate from `qunoc/`.
