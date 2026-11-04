# Evaluation Workflow

本目录按以下五类任务组织实验流程，所有命令从项目根目录执行：

1. 训练 package layout 的 RL 模型。
2. 使用训练好的模型完成 package layout design / scalability，并与其他 layout 策略比较。
3. 基于已设计好的 package 运行 program routing，对比 QuNoC routing 与 baseline MECH。
4. 运行 ablation 实验。
5. 运行统一噪声 Hamiltonian 单链扫描与真实编译样本仿真。

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

QuNoC routing consumes exact `package.pkl` paths from Step 2. If a required package is missing, run the matching layout scalability experiment first. `defaults.quantum_programs` defaults to `[vqe, qaoa, bv, s_ghz]` for every routing sweep, and an experiment can override it with `scalability.quantum_program_to_test`. `program_routing.s_ghz` configures the GHZ routing strategy but does not itself select that program. Missing GHZ cases require a normal run to compile them; `--statistics-only` cannot create missing circuits. The rebuilt CSV lists only the currently configured programs; directories for unselected programs are left untouched.

QuNoC routing outputs are written under `evaluate/result/routing_scalability/<experiment>/`; CSV/JSON outputs use `QuantumProgram` / `quantum_program` for the routed program:

- `output.csv`
- `<case>/exp_result.json`
- `<case>/qunoc_circuit.pkl`

### EQST model selection / EQST 模型选择

Full staged QuNoC routing defaults to the **`legacy`** linear timing and
raw-distance fidelity model, including package pickles without explicit EQST settings.
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
Unspecified calibration/planner settings retain the package's settings; pickles without those settings fall back to the values above. The model itself defaults to `legacy`
unless explicitly enabled. Each evaluation copies `package.op_config` before
applying overrides; shared package configuration is never mutated in place.

Hamiltonian estimates use actual staged Router batches for both duration and
fidelity. `reference_time_ns` and `reference_distance` calibrate the Hamiltonian
transfer timing; `g_max` caps coupling, `max_segment_steps` limits heuristic
planner stages (the fallback may exceed it), and `beam_width` bounds the ordered
batch search. In `hamiltonian-nn-lindblad-v3`, fidelity comes from the same calibrated
relaxation/dephasing dynamics as the standalone Hamiltonian experiments, not a
per-edge distance envelope. The reference is the six-site nearest-neighbor chain
at 125 ns and probability 0.992; baseline T1 is 141.5 microseconds and one effective
pure-dephasing rate is fitted once. Changed T1/coupling scenarios retain that baseline
fit. Effective distance remains a path diagnostic: the sum of primitive resident
transport displacements, **not** duration converted to distance. All times are ns.
Numerical or trajectory-uncertainty metadata accompanies noisy estimates; a batch
probability is not arbitrary-state/process fidelity or full-circuit decoherence.

CSV reports raw distance in `QSTDist` and includes:

- `QSTEffectiveDist`: model-plan primitive transport distance (equals raw distance for legacy).
- `QSTNegLogFidelity`: additive QST error, `-log(F_QST)`.
- `TotalNegLogFidelity`: additive physical-circuit error, available even when `TotalFide` underflows.
- `EQSTModel`: `legacy` or `hamiltonian`; never infer the model from `QSTDist`.

`GateFidelity` is computed by the Router from normalized additive cost, not by
rooting an underflowed `TotalFide`. For Hamiltonian mode, each batch supplies its
noisy log-fidelity directly; `enhanced_fide` does not determine this cost (it
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
every run, preventing duplicate rows or mixed column schemas.

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

Result JSONs without compilation signatures are accepted **only
with explicit `--statistics-only`** and matching available program, routing,
package-path, qubit-count and logical-count metadata. A warning explains that this
metadata has no topology fingerprint: ensure the package topology is unchanged.
The result records `legacy_compilation_metadata_trusted: true`. Circuits
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

Save separate copies of JSON/CSV if both model runs must be retained;
re-estimation updates the selected case's statistics. Artifact identity checks
require signed circuit files to remain at their recorded paths.

中文说明：默认保持 legacy 兼容；Hamiltonian 仅用于具备真实阶段的完整 QuNoC 路由。
`QSTDist` 记录原始距离，有效距离与负对数误差用于正确比较时间和保真度。
修改模型、标定、规划限制或时序后，只要编译输入不变即可重用已保存电路，仅重新统计。
`--statistics-only` 保证不调用路由、不覆盖 circuit pickle；缺少编译签名的结果需元数据匹配并显示警告。
CSV 为当前 sweep 的快照。

MECH outputs are written under `evaluate/result/baseline_mech/<experiment>/`:

- `output.csv`
- `<case>_mech.json`

The MECH entry point is `evaluate/run_mech_routing.py`.

## 4. Ablation

```bash
python evaluate/run_ablation.py
```

Canonical file:

- `evaluate/run_ablation.py`

Ablation is intentionally separate from routing scalability. It reuses package artifacts from layout scalability and writes resumable case outputs plus a merged CSV under `evaluate/result/ablation/package_size/output.csv` by default.

Both supported ablation strategies use the **`legacy` model**:

- `qunoc_wo_opt`: dummy-circuit timing and raw-distance EQST fidelity; the accounting convention
  is three physical gates per SWAP.
- `mech_eqst`: the separate MECH EQST timing model and raw-distance fidelity,
  not the staged Hamiltonian model. `--mech-eqst-basic-time-ns` retains its meaning.

`--eqst-model legacy` is accepted (and is the default). Requesting
`--eqst-model hamiltonian` fails clearly rather than silently labeling dummy/MECH
results as Hamiltonian. Use the full routing scalability runner for that model.
Ablation CSVs include the same four EQST columns, label `EQSTModel=legacy`, and
preserve the distinct `Baseline` names. MECH's unavailable exact total physical
fidelity/negative-log cost is blank. CSVs without EQST columns are interpreted as
`legacy`; unavailable metrics remain blank until recomputed. Per-case caches
use a schema/signature including the explicit accounting policy and effective
package timing/fidelity settings.

中文说明：消融实验没有完整的分阶段 Hamiltonian 批次，因此作为独立 legacy 基线；
不能通过参数将 dummy 或 MECH 结果重新标注为 Hamiltonian。

Nearest-neighbor Hamiltonian regressions are run with `python -m unittest evaluate.test_hamiltonian_strict_nn -v`.

## 5. Hamiltonian 论文实验

统一入口为 `evaluate/run_hamiltonian_simulation.py`，仅提供两个论文实验：

- `maximum-chain-length`：单激发最近邻链扫描，默认 `N=6,8,...,80`。
- `real-case-study`：实际保存编译映射中的六案例，上排3个 nonconflicting，下排3个 conflicting，分别覆盖2、3、4个居民。

不指定实验时运行两个实验。所有默认路径相对于代码位置解析，不依赖启动时的工作目录。

真实案例默认读取已保存的选择 manifest：
`evaluate/result/hamiltonian_revision/compiler/selection_report.json`。
该文件是必需的只读输入，可用 `--manifest <path>` 指定其他已保存的选择 manifest；
输出记录来源路径与 SHA-256，以便复现。输入映射按当前最近邻及居民约束校验，不能仅凭文件路径推断有效性。
传输芯粒仅支持一维链上的相邻站点耦合，不存在非局域直接边例外。

### 5.1 运行实验

```bash
# 默认运行两个实验 / Run both experiments
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python evaluate/run_hamiltonian_simulation.py

# 只运行链长扫描 / Maximum chain length only
OPENBLAS_NUM_THREADS=1 python evaluate/run_hamiltonian_simulation.py \
  --experiment maximum-chain-length

# 只运行真实案例 / Real case study only
OPENBLAS_NUM_THREADS=1 python evaluate/run_hamiltonian_simulation.py \
  --experiment real-case-study
```

两项实验默认统一使用 `g_max=None`、六站点请求时间62.5 ns、参考距离5条边：
`tau=62.5*sqrt((L-1)/5)` ns，`g/(2*pi)=4*sqrt(5/(L-1))` MHz。
六站点所需峰值是12 MHz，80站点约40.252 MHz；这是随区间变化的耦合需求，不是统一12 MHz上限。
T1为141.5 us，有效纯退相干率由六站点125 ns、0.992参考标定，不随请求时间重新拟合。

实际案例按最近邻区间重叠及全部居民约束分类，在数值模拟前确定，不能按输出保真度筛选。
Nonconflicting 使用兼容的互斥区间并行演化；conflicting 在案例实验中使用从左到右固定目标位置的扫描算法，
通过核心接口的显式 PST 分段求解。该案例算法不等于全局 routing 估计器的贪心/beam 规划器。
居民数包含静止居民；所有概率均为目标占据概率，不是任意逻辑态的过程保真度。

### 5.2 数据与图片分离

```text
evaluate/
├── result/hamiltonian_simulation/
│   ├── maximum_chain_length/       # JSON/CSV、各链长曲线数据
│   └── real_case_study/            # 来源、六案例映射、JSON/CSV曲线数据
└── figure/hamiltonian_simulation/
    ├── maximum_chain_length.png/.pdf/.svg
    └── real_case_study.png/.pdf/.svg
```

重复运行覆盖所选实验的输出，不创建编号目录。`result` 仅保存数据；PNG/PDF/SVG仅写入 `figure`。
如需保留对照，请在重跑前备份当前结果。数据文件中的状态与置信区间必须一并检查；
未达到采样精度不能算作验证完成。单个所选实验重跑不应覆盖另一实验的数据。

### 5.3 仅重新绘图

绘图实现统一集中在 **`evaluate/plot_result_figures.py`**，可调整图大小、字体、配色、marker和导出样式。
无参数时绘制系统级实验结果；Hamiltonian 图使用下列选项。

```bash
# 读取已保存的数据，重新绘制两图，不运行 Hamiltonian 演化
python evaluate/plot_result_figures.py --experiment hamiltonian

python evaluate/plot_result_figures.py --experiment maximum-chain-length
python evaluate/plot_result_figures.py --experiment real-case-study
```

**与系统重统计的区别：**以上两个实验分别验证单链与选定批次的占据演化；
`evaluate/test_routing_scalability.py --statistics-only` 更新整电路统计。
二者范围不同，不能用六案例结果替代全量路由评估。

## Plotting

Use the unified plotting entry:

```bash
python evaluate/plot_result_figures.py
```

Figures are written under `evaluate/figure/` by default, at the same level as `evaluate/result/`.

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
