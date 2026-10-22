# Evaluation Workflow

本目录按以下五类任务组织实验流程，所有命令从项目根目录执行：

1. 训练 package layout 的 RL 模型。
2. 使用训练好的模型完成 package layout design / scalability，并与其他 layout 策略比较。
3. 基于已设计好的 package 运行 program routing，对比 QuNoC routing 与 baseline MECH。
4. 运行 ablation 实验。
5. 运行统一噪声 Hamiltonian 自检、单链扫描、参数优化与真实编译样本仿真。

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
    t1_ns: 141500
    gamma_phi_ns: null  # fixed six-site calibration; not refitted to planner time
    solver: auto
    trajectories: 2048
    min_trajectories: 128
    seed: 0
    atol: 0.002
    rtol: 0.02
    confidence: 0.95
    master_max_dim: 128
    coupling_policy: preserve-request
```

These noise controls are shared with the Hamiltonian core. The default reference fit
uses 125 ns, 0.992 and baseline T1=141500 ns regardless of changed planner timing or
actual T1. `--eqst-t1-ns` is in **ns**, whereas the standalone runner's `--t1-us`
is in microseconds. `--eqst-coupling-policy cap-limited-fast` changes the actual
primitive coupling/timing; it is not a post-hoc duration multiplier.

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
batch search. In `hamiltonian-lindblad-v2`, fidelity comes from the same calibrated
relaxation/dephasing dynamics as the standalone Hamiltonian experiments, not a
per-edge distance envelope. The reference is the six-site nearest-neighbor chain
at 125 ns and probability 0.992; baseline T1 is 141.5 microseconds and one effective
pure-dephasing rate is fitted once. Changed T1/coupling scenarios retain that baseline
fit. Effective distance remains a path diagnostic: the sum of primitive resident
transport displacements, **not** duration converted to distance. All times are ns.
Numerical or trajectory-uncertainty metadata accompanies noisy estimates; a batch
probability is not arbitrary-state/process fidelity or full-circuit decoherence.

CSV retains the existing raw `QSTDist` column and appends:

- `QSTEffectiveDist`: model-plan primitive transport distance (equals raw distance for legacy).
- `QSTNegLogFidelity`: additive QST error, `-log(F_QST)`.
- `TotalNegLogFidelity`: additive physical-circuit error, available even when `TotalFide` underflows.
- `EQSTModel`: `legacy` or `hamiltonian`; never infer the model from `QSTDist`.

`GateFidelity` is computed by the Router from normalized additive cost, not by
rooting an underflowed `TotalFide`. For Hamiltonian mode, each batch supplies its
noisy log-fidelity directly; `enhanced_fide` no longer determines this cost (it
still applies to legacy statistics). `QSTEffectiveDist` is diagnostic only.
JSON includes `eqst_uncertainty`, convergence/budget counts and conservative
log-confidence endpoints. Reusing a cached noisy estimate does not reduce its
uncertainty by the square root of the reuse count. A per-batch 95% interval is
**not** automatically a whole-circuit 95% interval; the reported union-bound
confidence can be low or zero for many distinct sampled batches. Do not label
unconverged results as precise. The model covers noise during each resynthesized
transfer batch, not full algorithmic density-matrix evolution or unmodeled extra
compute/inter-batch idle times. Signed infinite costs are exported as explicit
strings with a serialization-policy marker; NaN is rejected. JSON also records `eqst_config`, model/version,
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

The old per-axis MECH wrappers are archived under `backup/evaluate/`; the active entry is `evaluate/run_mech_routing.py`.

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

Hamiltonian regression scripts have been archived under `backup/evaluate/`; they are not active evaluation commands.

## 5. Hamiltonian 论文实验

唯一实验脚本为 `evaluate/run_hamiltonian_simulation.py`，核心计算由
`qunoc.hamiltonian` 提供；原辅助/回归脚本仅归档，不是当前运行入口。
本文已合并原 `hamiltonian_simulation.md`，以下是统一使用说明。

所有模式使用六站点最近邻参考链（125 ns、目标占据概率0.992）标定的
T1＋有效纯退相干动力学。基准T1为141.5 μs，退相干只拟合一次，
不随链长或实际T1扫描重新拟合。图的横轴统一为 `Time (ns)`。
下述概率是目标占据态概率，不是任意未知逻辑态的过程保真度。

### 5.1 自检

```bash
python evaluate/run_hamiltonian_simulation.py --self-test
```

检查参考标定、主方程与轨迹求解的一致性、耦合关断、参数传递和实验编排。
自检不是论文实验数据，也不会重跑routing编译。

### 5.2 含噪声单激发链长扫描

默认链长为 **6、8、10、…、80（共38个）**。以下命令等价：

```bash
python evaluate/run_hamiltonian_simulation.py --experiment single-chain

python evaluate/run_hamiltonian_simulation.py \
  --experiment single-chain \
  --lengths "$(seq -s, 6 2 80)"
```

这是开放边界最近邻PST链，不是一个端到端直接长程边。默认峰值耦合为6 MHz，
根据实际耦合确定传输时间；输出理想对照、含噪声曲线、逐链长数据和汇总图。

```bash
# 默认覆盖：本次指定的链长全部重新计算并覆盖对应文件
python evaluate/run_hamiltonian_simulation.py \
  --experiment single-chain --lengths 6,20,80 --existing overwrite

# 跳过：仅复用参数一致、结果完整且满足验证条件的链长
python evaluate/run_hamiltonian_simulation.py \
  --experiment single-chain --lengths "$(seq -s, 6 2 80)" --existing skip
```

固定输出到 `evaluate/result/hamiltonian_revision/single_chain/`，不创建编号目录。
`--existing skip` 不是“有同名目录就跳过”：噪声、耦合、采样、求解精度或模型版本不一致，
以及结果不完整时均重新计算。**每次汇总CSV和绘图只使用本次指定的 lengths**；
其他已有长度文件可留存，但不混入本次汇总。

### 5.3 最长链参数优化

```bash
python evaluate/run_hamiltonian_simulation.py --experiment optimize
```

固定输出到 `evaluate/result/hamiltonian_revision/optimize/`，重复运行覆盖旧结果。
扫描80站点的9组参数：T1为141.5/200/300 μs，最大边耦合为6/9/12 MHz，
基准退相干率保持不变。保存全部参数点及基准对照，不自动把最好点改成默认配置。
非基准点属于器件改进假设，并非已经实测的硬件能力。

### 5.4 真实 compiler 样本：六组，每组3例

| 分组 | 2激发 | 3激发 | 4激发 |
|---|---:|---:|---:|
| 无冲突：活动pair端点互不重叠 | 3 | 3 | 3 |
| 有冲突：存在共享活动端点 | 3 | 3 | 3 |

默认选择 `vqe,qaoa,bv,s_ghz` 的实际保存电路；显式 `--programs all` 可包括其他历史程序。
激发数按该Travel stage/die的全部驻留态计数，**不丢弃静止居民**。
默认至少两个实际移动请求；k=3/4可能包含静止居民，不能据此声称3/4个态都在净移动。
占据集合相同的标签交换不会被当作有信息量的占据态传输验证。

**扫描场景并选择18例：**

```bash
python evaluate/run_hamiltonian_simulation.py \
  --experiment compiler \
  --routing-root evaluate/result/routing_scalability \
  --min-excitations 2 --max-excitations 4 \
  --cases-per-class 3 --min-moving 2 \
  --max-dimension 5000 --max-intervals 100 --max-state-mb 512 \
  --scan-only
```

`--cases-per-class` 表示每个冲突类别、**每个激发数**各3例，不是每类混合取3例。
保留扫描中符合条件的真实场景及来源、精确筛选计数和缺额；缺少某组时报告不完整，
不用其他组或手工请求补位。优先不同完整位置映射和较长链，考虑计算成本及固定ID排序。
资源限制可能使4激发样本链长短于2激发样本，应披露筛选规则。

**复用同一路径清单进行仿真：**

```bash
python evaluate/run_hamiltonian_simulation.py \
  --experiment compiler \
  --compiler-manifest evaluate/result/hamiltonian_revision/compiler/selection_report.json \
  --min-excitations 2 --max-excitations 4 \
  --cases-per-class 3 --min-moving 2 \
  --num-steps 101 \
  --trajectories 32768 --min-trajectories 256 \
  --atol 0.001 --rtol 0.005 --seed 0
```

compiler扫描和仿真统一写入 `evaluate/result/hamiltonian_revision/compiler/`。
仿真先完整读取输入清单，再更新同目录的报告和各case文件，避免输入/输出同路径时丢失场景。
复用清单不重扫历史电路、不重新编译routing，扫描的场景与来源仍保留；
重复运行覆盖对应选中case结果，汇总只列本次结果。
`--scan-only` 不重新演化，也不改写既有case数值结果及 `case_summary.csv`；报告状态明确区分扫描与仿真，这时旧CSV不代表新扫描已经完成仿真。
不再选中的case目录不主动删除，引用本次结果应以报告/汇总清单为准。

需要扫描后立即仿真时，去掉扫描命令中的 `--scan-only` 即可。
原 `routing_scalability` 的JSON、CSV、package和circuit pickle始终只读。

### 5.5 固定目录、结果复用与精度

默认根目录为 `evaluate/result/hamiltonian_revision/`，可用 `--output-dir` 指定其他根目录：

```text
hamiltonian_revision/
├── compiler/
│   ├── selection_report.json       # 扫描场景、六组选择和本次状态
│   ├── case_summary.csv            # 本次仿真结果表
│   └── selected_cases/<case-id>/    # 来源、summary、曲线CSV与PNG/PDF
├── single_chain/
│   ├── summary.json
│   ├── case_summary.csv            # 只包含本次指定 lengths
│   ├── length_summary.png/.pdf
│   └── N_006/ ... N_080/           # 每长度单独覆盖或复用
└── optimize/
    ├── summary.json
    ├── all_points.csv
    └── ...                        # 参数点结果与优化图
```

这三个论文实验不再产生 `_run_002` 等新路径。历史编号目录不会自动删除或合并。
手工 `transfer` 模式保留原行为，不受本次固定目录调整影响。

数值输出区分 ideal 振幅和 noisy probability；后者有轨迹样本数、误差和置信区间时，
须检查 `precision_met`/最终状态。未达预算精度不代表验证完成，不能只选择较高随机值。
运行较慢时可在命令前加 `OPENBLAS_NUM_THREADS=1`，不改变物理参数。

**与系统重统计的区别：**本节compiler实验对选出的批次做演化曲线；
第3节 `test_routing_scalability.py --statistics-only` 更新整电路duration/fidelity统计。
二者使用相同核心噪声接口，但输出路径、样本范围和用途不同，不能互相替代。

## Plotting

Use the scripted plotting entry instead of the old notebook workflow:

```bash
python evaluate/plot_result_figures.py
```

Figures are written under `evaluate/figure/` by default, at the same level as `evaluate/result/`. The old notebook is kept under `backup/evaluate/plot_results.ipynb` as historical reference only.

## Legacy files

These files are retained under `backup/evaluate/` for reference or compatibility, but are not recommended as primary workflow entrypoints:

- `backup/evaluate/shared_rl_layout.yaml`: legacy shared-RL config replaced by train-only + inference-only layout flow.
- `backup/evaluate/RL_only_layout.yaml`: legacy inline-RL layout config; migrate to `train_layout_model.yaml` and `test_layout_scalability.yaml`.
- `backup/evaluate/main_exp.py`: legacy overall summary script; prefer canonical routing/MECH CSVs and `plot_result_figures.py`.
- `backup/evaluate/plot_results.ipynb`: legacy plotting notebook.
- `backup/evaluate/run_mech_routing_all.py`: legacy monolithic MECH runner replaced by `evaluate/run_mech_routing.py --experiment all`.
- `backup/evaluate/run_mech_routing_package_size.py`, `backup/evaluate/run_mech_routing_package_shape.py`, `backup/evaluate/run_mech_routing_geometry.py`, `backup/evaluate/run_mech_routing_inter_fidelity.py`: compatibility wrappers for the consolidated MECH CLI.

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
