# 一维量子比特链：距离校准的 XY / PST 传输

输入源与目标位置，自动选择直接 XY 或连续子链 PST。核心已迁入 `qunoc.hamiltonian`，提供**无矩阵时间/距离估计**和**完整占据态数值演化**；`evaluate/run_hamiltonian_simulation.py` 提供命令行、绘图和 CSV/JSON 输出；旧目录已整体归档到 `backup/hamiltonian_simulation/`，其中历史脚本不再作为运行入口。核心不导入 routing、Qiskit 或 matplotlib；仅完整数值演化构造固定激发数硬核 XY 子空间的稀疏矩阵指数。

## 公共 Python 接口

从仓库根目录调用；Python ≥ 3.10，核心需要 `numpy scipy`，绘图另需 `matplotlib`。

### 无矩阵有序估计：`qunoc.hamiltonian`

```python
from qunoc.hamiltonian import estimate_transfer

estimate = estimate_transfer([0, 9, 5], [9, 10, 19], chain_length=20)
print(estimate["transfer_time_ns"], estimate["effective_distance_edges"])
# 520.1381155644951 34
print(estimate["log_fidelity"])
# -0.054618767541397...

# 占据集合相同但逻辑标签互换，不是零成本。
swap = estimate_transfer([0, 1], [1, 0])
# transfer_time_ns ≈ 55.9016994375, effective_distance_edges = 2
```

签名为：

```python
estimate_transfer(sources, targets, chain_length=None, *,
                  reference_time_ns=125.0, reference_distance=5.0,
                  reference_fidelity=0.992, g_max=3*pi/250,
                  max_segment_steps=10, beam_width=128, ordered=True)
```

- 默认 `ordered=True`，始终按输入顺序配对，贪心、束搜索和最终校验均检查每个标签的目标。必须传入**该阶段开始时全部建模居民**，包括静止的 `i:i`，否则未知居民的位移不会计费。静止居民被镜像移动时必须恢复，并累计实际位移；未被扰动则零成本。
- 返回新字典，含 `transfer_time_ns`、`effective_distance_edges`、直接由距离算出的 `log_fidelity` / `neglog_fidelity`、`model_version` 及不可变 `plan`。不会先求极小概率再取对数；`log_fidelity` 是距离包络成本，不是任意量子态过程保真度。routing 的概率标定另外以其 fabrication 配置为准。
- `plan_transfer(SimulationConfig(...))` 与 `run_simulation(...)` 共用同一冻结 `TransferPlan`。计划只保存不可变配置、位置、基本操作及时序，不生成 `C(N,k)` 基底、稀疏矩阵或长度为链长的耦合数组；PST 峰值用 `sqrt(floor(L/2)*ceil(L/2))` 闭式。`plan.to_route_details()` 生成独立可修改的审计字典。
- 进程内 LRU 以不可变请求内容、链长、校准和规划参数为键，`maxsize=256`，不按 Circuit 身份缓存，不泄漏可变缓存字典。采样密度不影响规划缓存。
- 启发式有硬工作量上限：最多 32 步、128 束宽、20,000 次候选展开，较大链/高占据绕开束搜索。有序请求即使预算耗尽，也用确定性奇偶相邻镜像置换兜底；预算是启发式预算，不限制最终合法兜底阶段数。跨度大于 512 站点时，以整段及内部镜像合成端点交换，避免枚举巨大空隙，最多 `2k` 段；过程中移动的内部居民照常计费。两种兜底都不需要 Fock 空间，不保证最优成本。
- `ordered=False` 保留旧占据目标语义；`SimulationConfig` 和下述兼容模拟函数也默认占据目标。完整占据概率即使接近 1，也不能证明有序量子信息传输或相位校正。

### 兼容的完整模拟：两字段返回

统一通过 `qunoc.hamiltonian` 导入；完整模拟的两字段返回和占据默认语义不变，旧目录导入/命令行路径已退役。

```python
from qunoc.hamiltonian import simulate_transfer

simulate_transfer([0, 5, 2], [9, 19, 13], chain_length=20)
# {'transfer_time_ns': 209.16500663351889, 'fidelity': 0.946846047602...}

simulate_transfer([0, 9, 5], [9, 10, 19], chain_length=20)
# {'transfer_time_ns': 520.1381155644951, 'fidelity': 0.946846047602...}
```

- `sources`、`targets` 为等长非空整数位置序列，按输入顺序配对，不自动重新匹配。源各自唯一、目标各自唯一、位置不得越界；不接受浮点位置或布尔值。
- `chain_length` 可省略，推断为最大位置加一；若要使用端点范围外的站点进行 PST 规划，应显式提供链长。
- 可选关键字：`reference_time_ns=125.0`、`reference_distance=5`、`reference_fidelity=0.992`、`g_max=3π/250`、`ideal=False`。`g_max=None` 移除耦合上限；`ideal=True` 仅关闭距离衰减。
- 返回值只有 `transfer_time_ns` 和 `fidelity` 两个浮点数；返回**最终值而非峰值**，不写任何文件。接口内部只采样恒定哈密顿量区间端点，末态矩阵指数仍完整计算。
- evaluate runner 的 `__all__` 仍只导出 `simulate_transfer`。下述 CLI 用于诊断与结果绘图；核心另提供 `SimulationConfig`、`TransferPlan`、`plan_transfer`、`estimate_transfer`、`run_simulation` 和 `MODEL_VERSION`。

**这里的 fidelity 只是指定目标占据态的概率，不保证带标签激发逐一到达，也不是任意叠加态或完整量子过程保真度。** 默认两例的联合占据概率约为 **94.6846%**，不能声称达到 99% 或任何未给出的保真度门槛。

## 校准原则与物理单位

用户指定参考点为 **六站点、单激发、跨越 5 条边、125 ns、概率 0.992**。这是一点校准的设计原则，不足以唯一识别误差机制或复现实验。

```text
d_ref = 5,  t_ref = 125 ns,  F_ref = 0.992
请求用时 τ_req(d) = t_ref √(d/d_ref),  d = |right − left|
参考尺度 g_ref = π/(2t_ref) = 0.012566370614359173 rad/ns
默认最大边耦合 J_max = 3g_ref = 0.03769911184307752 rad/ns
J_max/(2π) = 6 MHz
```

**耦合上限优先，不强制严格满足平方根用时。** 默认上限取自六站点工程化剖面的峰值，作为建模边界，不宣称它就是器件的最大可实现耦合。采用 `ℏ=1`，所有时间为 **ns**，所有耦合为 **rad/ns**；实验若给 `J/(2π)` 的 MHz 值，应乘 `2π×10⁻³` 转为 rad/ns。

```text
H = Σ_(u,v) J_uv (σ⁺_u σ⁻_v + σ⁻_u σ⁺_v)
  = Σ_(u,v) (J_uv/2) (X_u X_v + Y_u Y_v)    [实耦合]
```

每个基本操作分别设置耦合与完成时间：

| 操作 | 单位耦合剖面 | 实际尺度与时间 |
| --- | --- | --- |
| 直接 XY 边 `(s,t)` | `M=1`，`J=g` | `g=min[π/(2τ_req), J_max/M]`，`τ=π/(2g)` |
| PST 区间 `[l,r]`，`L=r−l+1` | `J_j=g√[j(L−j)]`，`M=max_j √[j(L−j)]` | 同上，每段独立限幅 |

不限幅时 `g=π/(2τ_req)`。同阶段不同操作可以有不同完成时间：完成的边或子链立即关闭，其他操作继续演化；阶段用时为其中最大值。下一阶段在前一阶段全部完成后开始。整个态连续相干演化，**不重置、不测量中间态**。

文献背景：`backup/hamiltonian_simulation/` 内 *Enhanced quantum state transfer by circumventing quantum chaotic behavior* 主文第 3 页 Fig. 2 描述六站点约 125 ns、保真度 “above 0.99”，使用 `J/(2π)=2 MHz` 的工程化尺度；补充材料第 4 页 Fig. 4 给出接近 `[4.47,5.66,6,5.66,4.47] MHz` 的耦合幅值。**0.992 是用户指定校准值，不是此处逐字引用的文献精确值。** 代码使用正耦合；对这里实初态与实阶段哈密顿量同时整体反号只共轭演化，不改变占据概率。

## 多激发距离误差模型

理想目标概率来自完整相干演化，距离校准另外施加唯象包络：

```text
F_ideal(t) = |⟨T|ψ_ideal(t)⟩|²
α = −ln(F_ref)/d_ref = 0.0016064343394528533 / edge
F_calibrated(t) = exp[−α L(t)] F_ideal(t)
```

总距离暴露 `L(t)` 按实际路线计算，不直接把原请求距离求和：

- 直接操作的距离成本为 `|target−source|`。
- PST 操作的成本为 `Σ |l+r−2p|`，求和覆盖**该逻辑阶段开始时**区间内的全部占据 `p`。因此内部激发、绕路和中间被移动的静止请求都会按实际镜像位移计入；区间中心激发成本为零。
- 每个操作的距离成本从其开始到实际完成时线性累积，完成后保持不变，所有操作相加得到 `L(t)`。这是连续曲线的插值约定，不是微观粒子轨迹测量。
- 真正没有移动的激发不产生距离成本；没有额外等待时间或总激发数寿命惩罚。降低耦合只延长时间，不改变这条路线的最终距离衰减。真实器件往往还有随时间增长的误差，这里没有包含。

**该包络不是局域振幅阻尼的精确解，不是 Lindblad 主方程模拟，也不代表测得的 T1/T2。** 它不包含额外退相干、失谐、串扰、噪声、开关脉冲、硬件路由开销或任意态相位校正。`ideal=True` / `--ideal` 或 `reference_fidelity=1` 可关闭包络而保留时间和耦合设置。

## 两个已有示例

以下命令从仓库根目录执行，在 `evaluate/result/hamiltonian_simulation/` 生成两例的 PNG / CSV / JSON；图片使用 `Agg` 后端。历史结果保留在 `backup/hamiltonian_simulation/results/`，不会覆盖。

```bash
PYTHONDONTWRITEBYTECODE=1 python3 evaluate/run_hamiltonian_simulation.py \
  -n 20 --pairs '0:9,5:19,2:13' --num-steps 401 \
  --output-dir evaluate/result/hamiltonian_simulation/nonconflicting

PYTHONDONTWRITEBYTECODE=1 python3 evaluate/run_hamiltonian_simulation.py \
  -n 20 --pairs '0:9,9:10,5:19' --num-steps 401 \
  --output-dir evaluate/result/hamiltonian_simulation/conflicting
```

| 默认结果 | 无冲突 `direct-xy` | 冲突 `pst-segments` |
| --- | ---: | ---: |
| 逻辑阶段数 | 1 | 2 |
| 恒定哈密顿量区间数 | 3 | 3 |
| 完成时间（ns） | **209.165006634** | **520.138115564** |
| 最终理想占据概率 | ≈1 | ≈1 |
| 最终距离校准占据概率 | **0.946846047602** | **0.946846047602** |
| 总传输距离成本 | 34 | 34 |
| 默认输出点数 | 1201 | 1201 |

### 无端点冲突

`{0,2,5} → {9,13,19}`，三条直接边分别跨越 9、14、11 条边，完成时间分别为 **167.705098312、209.165006634、185.404962177 ns**。它们同时启动，依次关闭，联合理想概率为

```text
F_ideal(t) = Π_i sin²[(π/2) min(t/τ_i,1)]
最终 F_calibrated = 0.992^[(9+14+11)/5]
```

只判断端点是否共享；几何区间重叠不构成直接边冲突。**非相邻直接边需要额外硬件连接能力，不等于只用最近邻链实现这些传输。**

### 存在端点冲突

站点 9 被多个 pair 使用，整个任务转为 PST，而不是把无冲突部分单独作直接传输：

```text
阶段 1：镜像 [5,19]，请求 209.165007 ns，限幅后 311.804782 ns
        5→19 位移14，9→15 位移6；距离成本20
阶段 2：并行镜像 [0,9]、[10,15]
        [0,9] 请求167.705098 ns，限幅后208.333333 ns；成本9
        [10,15] 用时125 ns，先完成后关闭；成本5

{0,5,9} → {0,15,19} → {9,10,19}
总时间 311.804782231 + 208.333333333 = 520.138115564 ns
总距离 20 + 9 + 5 = 34，最终 F_calibrated = 0.992^(34/5)
```

第一阶段目标概率为零；第二阶段令 `u=t−311.804782231`，理想曲线为 `sin¹⁸[(π/2)min(u/208.333333333,1)] × sin¹⁰[(π/2)min(u/125,1)]`。移除上限后冲突任务时间为 **376.870104946 ns**，但在此距离包络模型中最终概率不变。

standalone 的占据规划器先长位移/长段优先贪心，再有限束搜索；不保证最少阶段或最短时间，小规模占据搜索失败仍明确报错，失败不意味着物理上无解。较大任务使用多项式置换兜底；有序估计请求无论大小，启发式失败后均精确兜底。`--segments '5:19;0:9,10:15'` 可指定冲突路线；分号分阶段，逗号分并行区间。同阶段区间不得共享站点，段内所有占据均镜像，最终占据集合必须匹配。无冲突任务不能用手动 PST 覆盖自动直接路线。

`i:i` 是静止占据，不创建自耦合。**仅 standalone/ordered=False**：全静止或源目标占据集合已经相同且自动路线无需操作时，返回 `0 ns, fidelity=1`；这不证明标签交换已完成。默认 `estimate_transfer(..., ordered=True)` 对非恒等标签排列会返回正时间/距离。

## CLI、采样和输出

| 参数 | 默认 / 必填 | 含义 |
| --- | --- | --- |
| `-n`, `--chain-length` | 必填 | 正整数站点数，编号 `0…N−1` |
| `--pairs` | 必填 | 非空有序 `source:target` 列表 |
| `--reference-time-ns` | `125` | 有限正参考时间，ns |
| `--reference-distance` | `5` | 有限正参考距离，边数 |
| `--reference-fidelity` | `0.992` | 参考概率，必须在 `(0,1]` |
| `--g-max` | `0.03769911184307752` | 有限正最大边耦合，rad/ns |
| `--no-coupling-cap` | 关闭 | 移除上限，与 `--g-max` 互斥 |
| `--ideal` | 关闭 | 关闭距离衰减，不改变理想演化/时间 |
| `--segments` | 自动 | 仅冲突任务可用的手动 PST 阶段 |
| `--max-segment-steps` / `--beam-width` | `10` / `128` | 自动规划阶段预算与束宽 |
| `--num-steps` | `401` | **每个恒定哈密顿量区间**的采样点数，至少 2 |
| `--output-dir` | 不写文件 | 写 PNG / CSV / JSON，同名文件覆盖 |

旧归一化 `--g-base` 和旧模式参数不再提供。`--num-steps` 不是积分步数；使用 `expm_multiply`，改变采样密度不改变矩阵指数的定义或完成时间。每个关断事件划分新的恒定哈密顿量区间，`I` 个区间产生 `1+I×(num_steps−1)` 点。边界只出现一次；保留最终时刻。峰值是**各自采样网格上的最大值**，可能早于完成时刻，不是连续时间最优化；浮点舍入可能使理想概率略高于 1，原始数据不裁剪。

标准输出始终为 JSON 摘要；指定目录时另写：

- `summary.json`：`final_time` 单位 ns；保留 `peak_time/peak_fidelity/final_fidelity` 的**理想**含义，另有 `calibrated_peak_time/calibrated_peak_fidelity/calibrated_final_fidelity`。元数据列出参考来源、公式、单位和模型限制；`route_details` 区分逻辑阶段、恒定哈密顿量区间以及每个操作的请求/实际时间、实际耦合、是否限幅和距离成本。
- `fidelity.csv`：`time_ns`、`conditional_ideal_target_amp_real/imag`、`conditional_ideal_source_amp_real/imag`、`ideal_target_fidelity`、`distance_exposure_edges`、`distance_survival_factor`、`calibrated_target_fidelity`。振幅是归一化理想成功分支的振幅，**目标振幅模平方等于理想概率，不等于校准概率**；它们不是带损耗完整态向量或各站点边缘概率。
- `fidelity.png`：横轴 **Time (ns)**，理想虚线与距离校准实线、图例和完成值标注；固定浅色、已验证双系列配色，CSV 提供可检查的表格版本。

## 测试与规模限制

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover \
  -s evaluate -p 'test_chain_evolution.py' -v

# 也可直接运行已移动的测试脚本。
PYTHONDONTWRITEBYTECODE=1 python3 evaluate/test_chain_evolution.py -v
```

测试已从 `hamiltonian_simulation/test_chain_evolution.py` 移至 `evaluate/test_chain_evolution.py`。43 项测试保留原 32 项：公共接口、真正六站点最近邻参考链、两例完整解析曲线、独立关断、限幅与不限幅、内部激发/绕路/镜像中心距离、静止与零时间、手动路线、相干性、对易性、采样事件、输出字段和非法输入；新增估计/数值一致、逐对互换/循环/全占据、静止居民恢复、相邻兜底与不变量、缓存隔离/有界淘汰、概率下溢及独立进程 CLI/import 检查。核心数值测试与 wrapper 输出/绘图测试分别导入，mock 作用域对应实际定义域。每个新增/修改函数都有中英文功能、输入、输出 docstring。

**仅完整数值演化**的基底维数为 `C(N,k)`，`N=20,k=3` 时为 1140；显式基底和约 `num_steps×C(N,k)` 的采样态存储仍限制规模。大并发请使用 `estimate_transfer`：测试在禁止基底/矩阵/耦合剖面构造的条件下验证 `N=80,k=40`、80 站点全占据以及十亿站点稀疏链。`C(80,40)>10²³`，估计不分配此空间。原始三份 PDF、历史计划和两组历史结果保持不改；迁移没有改变两组示例的数值结果，因此没有重生成图片。
