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

QuNoC routing consumes exact `package.pkl` paths from Step 2. If a required package is missing, run the matching layout scalability experiment first. `defaults.quantum_programs` defaults to `[vqe, qaoa, bv, s_ghz]` for every routing sweep, and an experiment can override it with `scalability.quantum_program_to_test`. `program_routing.s_ghz` configures the GHZ routing strategy but does not itself select that program. Missing GHZ cases require a normal run to compile them; `--statistics-only` cannot create missing circuits. Historical QFT directories are not deleted; the rebuilt CSV lists only the currently configured programs.

QuNoC routing outputs are written under `evaluate/result/routing_scalability/<experiment>/`; new CSV/JSON outputs use `QuantumProgram` / `quantum_program` for the routed program:

- `output.csv`
- `<case>/exp_result.json`
- `<case>/qunoc_circuit.pkl`

### EQST model selection / EQST 模型选择

Full staged QuNoC routing defaults to the original **legacy** linear timing and
raw-distance fidelity model, including when loading older package pickles.
Hamiltonian timing/fidelity is opt-in through `defaults.eqst` in the routing YAML
(or the same mapping under `experiments.<name>.defaults.eqst`):

```yaml
defaults:
  eqst:
    model: hamiltonian
    reference_time_ns: 125
    reference_distance: 5
    g_max: 0.03769911184307752  # 3*pi/250, rad/ns
    max_segment_steps: 10
    beam_width: 128
```

The model can also be selected from the CLI:

```bash
python evaluate/test_routing_scalability.py \
  --config evaluate/test_routing_scalability.yaml --eqst-model hamiltonian \
  --eqst-reference-time-ns 125 --eqst-reference-distance 5 \
  --eqst-g-max 0.03769911184307752 --eqst-max-segment-steps 10 --eqst-beam-width 128
```

Explicit CLI flags override experiment defaults, which override global defaults.
Unspecified calibration/planner settings retain the package's settings; older
pickles fall back to the values above. The model itself defaults to `legacy`
unless explicitly enabled. Each evaluation copies `package.op_config` before
applying overrides; shared package configuration is never mutated in place.

Hamiltonian estimates use actual staged Router batches for both duration and
fidelity. `reference_time_ns` and `reference_distance` calibrate the Hamiltonian
transfer timing; `g_max` caps coupling, `max_segment_steps` limits heuristic
planner stages (the fallback may exceed it), and `beam_width` bounds the ordered
batch search. Effective fidelity
distance is the sum of primitive transport displacements in the resynthesized
net-map plan, **not** duration converted to distance. It can differ from raw
Transport action distance in either direction. All times are in ns. Calibration
values must be positive and finite, and planner limits must be positive integers.

CSV retains the existing raw `QSTDist` column and appends:

- `QSTEffectiveDist`: model-plan primitive transport distance (equals raw distance for legacy).
- `QSTNegLogFidelity`: additive QST error, `-log(F_QST)`.
- `TotalNegLogFidelity`: additive physical-circuit error, available even when `TotalFide` underflows.
- `EQSTModel`: `legacy` or `hamiltonian`; never infer the model from `QSTDist`.

`GateFidelity` is computed by the Router from normalized additive cost, not by
rooting an underflowed `TotalFide`. JSON also records `eqst_config`, model/version,
`eqst_batch_count`, and a cache `signature`. Signatures include effective EQST
calibration/caps/planner settings, all operation timing, fabrication fidelities
(including `enhanced_fide`), program/routing options, and resolved package path,
file size and timestamps/identity. A separate `compilation_signature` records
package topology/chiplet geometry (excluding calibration), program/routing options
and qubit count; `compiled_circuit_identity` ties metadata to the saved pickle.
Changing only the EQST model/calibration, timing or fidelity invalidates statistics
but **reuses the saved compilation** when these compilation inputs still match.
`reuse_result: false` forces fresh statistics, not a fresh route; use `--recompile`
to deliberately compile again. The sweep CSV is rebuilt from selected cases on
every run, preventing duplicate rows or mixed old/new column schemas.

### Re-estimate saved circuits without routing / 只重新统计已有 circuit

```bash
python evaluate/test_routing_scalability.py \
  --config evaluate/test_routing_scalability.yaml \
  --eqst-model hamiltonian --statistics-only
```

`--statistics-only` never generates a program or calls routing compilation. It
loads each case's `qunoc_circuit.pkl` and `exp_result.json`, uses the cached logical
gate count and current package configuration, and rewrites only statistics
JSON/CSV. The circuit pickle is not overwritten. Missing files or mismatched
compilation metadata produce a clear error rather than silently rerouting. Select
only cases already compiled and keep their original program/routing/layout options
in the YAML. `defaults.statistics_only: true` is the equivalent config setting.

Historical result JSONs without compilation signatures may be migrated **only
with explicit `--statistics-only`** and matching available program, routing,
package-path, qubit-count and logical-count metadata. A warning explains that old
metadata has no topology fingerprint: ensure the package topology is unchanged.
The result records `legacy_compilation_metadata_trusted: true`. Older circuits
with ambiguous/incomplete action history may still be rejected by the estimator
with a diagnostic; statistics-only never compiles as a fallback.

For a single saved circuit outside a sweep, use the Router API directly (the
package and circuit must correspond; load only trusted pickle artifacts):

```python
import copy
import json
from pathlib import Path
from qunoc.element.package import Package
from qunoc.routing.circuit import Circuit
from qunoc.routing.router import Router

package = Package.from_pickle("path/to/package.pkl")
package.op_config = copy.deepcopy(package.op_config)
package.op_config.eqst_model = "hamiltonian"
circuit = Circuit.from_pickle("path/to/qunoc_circuit.pkl")
metadata = json.loads(Path("path/to/exp_result.json").read_text())
metrics = Router.reestimate_compiled_circuit(package, circuit, metadata["logic_gate_num"])
print(metrics["circuit_duration"], metrics["gate_fidelity"])
```

Archive the previous JSON/CSV if both model runs must be retained;
re-estimation updates the selected case's statistics. Artifact identity checks
require modern signed circuit files to remain at their recorded paths.

中文说明：默认保持 legacy 兼容；Hamiltonian 仅用于具备真实阶段的完整 QuNoC 路由。
原始 `QSTDist` 不变，新增有效距离与负对数误差用于正确比较时间和保真度。
修改模型、标定、规划限制或时序后，只要编译输入不变即可重用已保存电路，仅重新统计。
`--statistics-only` 保证不调用路由、不覆盖 circuit pickle；旧结果需元数据匹配并显示警告。
CSV 为当前 sweep 的快照。

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

Both supported ablation strategies remain **legacy baselines**:

- `qunoc_wo_opt`: dummy-circuit timing and raw-distance EQST fidelity; the existing
  three-physical-gates-per-SWAP convention is deliberately preserved.
- `mech_eqst`: the separate MECH EQST timing model and raw-distance fidelity,
  not the staged Hamiltonian model. `--mech-eqst-basic-time-ns` retains its meaning.

`--eqst-model legacy` is accepted (and is the default). Requesting
`--eqst-model hamiltonian` fails clearly rather than silently labeling dummy/MECH
results as Hamiltonian. Use the full routing scalability runner for that model.
Ablation CSVs append the same four EQST columns, label `EQSTModel=legacy`, and
preserve the distinct `Baseline` names. MECH's unavailable exact total physical
fidelity/negative-log cost remains blank. Older ablation CSVs migrate with legacy
labels; unavailable new metrics remain blank until recomputed. Per-case caches
use a new schema/signature including the explicit accounting policy and effective
package timing/fidelity settings.

中文说明：消融实验没有完整的分阶段 Hamiltonian 批次，因此仍作为独立 legacy 基线；
不以新增参数将 dummy 或 MECH 结果重新标注为 Hamiltonian。

Lightweight integration checks (synthetic cases, no long experiments):

```bash
python -m unittest evaluate.test_hamiltonian_evaluation
```

## Standalone Hamiltonian diagnostics

The reusable implementation lives in `qunoc.hamiltonian`; the active diagnostic,
plotting and CSV/JSON runner is `evaluate/run_hamiltonian_simulation.py`.
The former top-level `hamiltonian_simulation/` directory is preserved under
`backup/hamiltonian_simulation/` as historical material, not an active import path.
See [Hamiltonian usage](hamiltonian_simulation.md) for the APIs, calibration and limitations.

```bash
python evaluate/run_hamiltonian_simulation.py -n 20 --pairs 0:9,5:19,2:13 \
  --output-dir evaluate/result/hamiltonian_simulation/nonconflicting
python evaluate/run_hamiltonian_simulation.py -n 20 --pairs 0:9,9:10,5:19 \
  --output-dir evaluate/result/hamiltonian_simulation/conflicting
python -m unittest evaluate.test_chain_evolution
```

中文说明：数值核心、无矩阵估计已在 `qunoc`，独立诊断与测试在 `evaluate`；
旧资料完整归档，新的运行与测试不依赖归档目录。

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
