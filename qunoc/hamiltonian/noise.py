"""Uniform T1 plus local pure dephasing / 均匀 T1 与局域纯退相干。

Only the target fixed-k occupation is propagated: losses cannot return to that
sector, so exp(-k*t/T1) is exact for this observable, not a full density matrix.
仅传播目标固定激发扇区；零温损失不能返回，故目标概率可严格乘弛豫因子。
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass
from functools import lru_cache
from itertools import combinations
from typing import TYPE_CHECKING, Sequence

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import eigh
from scipy.optimize import brentq
from scipy.stats import poisson

if TYPE_CHECKING:
    from .chain_evolution import TransferPlan

NOISE_MODEL_VERSION = "hamiltonian-nn-lindblad-v3"
BASELINE_T1_NS = 141500.0
MAX_MASTER_DIM = 128
MAX_SLATER_SITES = 512
MAX_SLATER_INTERVALS = 1024


@dataclass(frozen=True)
class NoiseConfig:
    """Purpose / 功能: Freeze physical and solver controls / 冻结物理和求解参数。

    Inputs / 输入: Times in ns, gamma in ns^-1, probability tolerances / 纳秒时间、退相干率及概率容差。
    Outputs / 输出: Hashable validated controls; None gamma fits baseline only / 可哈希配置，缺省 gamma 仅拟合基准。
    """
    t1_ns: float = BASELINE_T1_NS
    gamma_phi_ns: float | None = None
    calibration_t1_ns: float = BASELINE_T1_NS
    calibration_time_ns: float = 125.0
    calibration_fidelity: float = 0.992
    solver: str = "auto"
    trajectories: int = 2048
    min_trajectories: int = 128
    seed: int = 0
    atol: float = 0.002
    rtol: float = 0.02
    confidence: float = 0.95
    master_max_dim: int = MAX_MASTER_DIM

    def __post_init__(self) -> None:
        """Purpose / 功能: Reject invalid controls without fitting / 不拟合即校验控制。

        Inputs / 输入: Dataclass fields / 数据类字段。
        Outputs / 输出: None or explicit ValueError / 无返回或显式异常。
        """
        for name in ("t1_ns", "calibration_t1_ns"):
            value = getattr(self, name)
            if isinstance(value, bool) or math.isnan(value) or value <= 0:
                raise ValueError(f"{name} must be positive (positive infinity disables relaxation)")
        if not math.isfinite(self.calibration_time_ns) or self.calibration_time_ns <= 0:
            raise ValueError("calibration_time_ns must be finite and positive")
        if not math.isfinite(self.calibration_fidelity) or not 0 < self.calibration_fidelity <= 1:
            raise ValueError("calibration_fidelity must be finite and in (0,1]")
        if self.gamma_phi_ns is not None and (
                not math.isfinite(self.gamma_phi_ns) or self.gamma_phi_ns < 0):
            raise ValueError("gamma_phi_ns must be finite and nonnegative, or None")
        if self.solver not in ("auto", "master", "trajectories"):
            raise ValueError("solver must be auto, master, or trajectories")
        for name, minimum in (("trajectories", 2), ("min_trajectories", 2), ("seed", 0),
                              ("master_max_dim", 1)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if self.min_trajectories > self.trajectories:
            raise ValueError("min_trajectories must not exceed trajectories")
        if self.master_max_dim > MAX_MASTER_DIM:
            raise ValueError(f"master_max_dim exceeds the hard safety limit {MAX_MASTER_DIM}")
        if not math.isfinite(self.atol) or self.atol <= 0 or not math.isfinite(self.rtol) or self.rtol < 0:
            raise ValueError("atol must be positive and rtol nonnegative, both finite")
        if not math.isfinite(self.confidence) or not 0 < self.confidence < 1:
            raise ValueError("confidence must be strictly between zero and one")


def _reference_probability(gamma: float, time_ns: float, t1_ns: float) -> float:
    """Purpose / 功能: Solve the true six-site NN reference / 求真实六站点最近邻参考解。

    Inputs / 输入: Pure dephasing rate, PST time, reference T1 / 退相干率、PST 时间及参考 T1。
    Outputs / 输出: Endpoint occupation including relaxation / 含弛豫的末端占据概率。
    """
    from scipy.sparse import diags
    couplings = math.pi / (2 * time_ns) * np.sqrt(np.arange(1, 6) * np.arange(5, 0, -1))
    hamiltonian = diags((couplings, couplings), (-1, 1), shape=(6, 6), format="csr")
    damping = 2 * gamma * (1 - np.eye(6))
    rho = np.zeros((6, 6), dtype=complex)
    rho[0, 0] = 1
    values, _ = _density_interval(hamiltonian, damping, rho, time_ns, (time_ns,), 5)
    return float(values[0] * math.exp(-time_ns / t1_ns))


@lru_cache(maxsize=32)
def _fit_reference(time_ns: float, fidelity: float, calibration_t1_ns: float) -> tuple[float, float]:
    """Purpose / 功能: Fit once against an immutable reference / 对固定参考缓存一次拟合。

    Inputs / 输入: Reference time/F and calibration T1, never actual T1 / 参考时间、概率及标定 T1，非实际 T1。
    Outputs / 输出: Fitted gamma and achieved probability / 拟合率与实际参考概率。
    """
    if not math.isfinite(time_ns) or time_ns <= 0 or not math.isfinite(fidelity) or not 0 < fidelity <= 1:
        raise ValueError("reference time must be positive and fidelity in (0,1]")
    ceiling = math.exp(-time_ns / calibration_t1_ns)
    if fidelity > ceiling + 2e-12:
        raise ValueError("reference fidelity exceeds the uniform-T1-only upper bound")
    if abs(fidelity - ceiling) <= 2e-12:
        return 0.0, ceiling
    upper = 1.0 / time_ns
    while _reference_probability(upper, time_ns, calibration_t1_ns) > fidelity:
        upper *= 2
        if upper * time_ns > 128:
            raise ValueError("reference target is outside the supported dephasing-fit bracket")
    gamma = brentq(lambda value: _reference_probability(value, time_ns, calibration_t1_ns) - fidelity,
                   0.0, upper, xtol=1e-15, rtol=1e-12)
    return gamma, _reference_probability(gamma, time_ns, calibration_t1_ns)


def calibration_reference(noise: NoiseConfig | None = None, reference_time_ns: float | None = None,
                          reference_fidelity: float | None = None) -> dict[str, object]:
    """Purpose / 功能: Describe the fixed true-NN calibration / 返回固定真实最近邻标定。

    Inputs / 输入: Noise options and six-site reference time/F / 噪声配置及六站点参考时间与概率。
    Outputs / 输出: JSON-safe fit/provenance; actual T1 never changes fitted gamma / 可序列化标定，实际 T1 不影响拟合。
    """
    noise = NoiseConfig() if noise is None else noise
    reference_time_ns = noise.calibration_time_ns if reference_time_ns is None else reference_time_ns
    reference_fidelity = noise.calibration_fidelity if reference_fidelity is None else reference_fidelity
    gamma, achieved = _fit_reference(float(reference_time_ns), float(reference_fidelity),
                                    float(noise.calibration_t1_ns))
    return {"sites": 6, "topology": "open-nearest-neighbor-pst", "gamma_phi_ns": gamma,
            "tphi_ns": None if gamma == 0 else 1 / gamma,
            "t1_ns": None if math.isinf(noise.calibration_t1_ns) else noise.calibration_t1_ns,
            "calibration_t1_ns": None if math.isinf(noise.calibration_t1_ns) else noise.calibration_t1_ns,
            "reference_time_ns": reference_time_ns, "reference_fidelity": reference_fidelity,
            "achieved_fidelity": achieved, "source": "Supplementary Table 1, p18; effective fitted pure dephasing",
            "actual_t1_used_for_fit": False}


def _density_interval(hamiltonian, damping: np.ndarray, rho: np.ndarray, duration: float,
                      sample_times: Sequence[float], target_index: int) -> tuple[np.ndarray, np.ndarray]:
    """Purpose / 功能: Integrate a density matrix, not a dense Liouvillian / 积分密度矩阵而非稠密超算符。

    Inputs / 输入: Real sparse H, damping, initial rho, local times, target / 实稀疏 H、阻尼、初态、局部时间及目标。
    Outputs / 输出: Target probabilities and final rho / 目标概率与最终密度矩阵。
    """
    dim = rho.shape[0]

    def rhs(_time: float, flat: np.ndarray) -> np.ndarray:
        """Purpose / 功能: Apply Lindblad RHS / 应用主方程右端。

        Inputs / 输入: Time and flattened rho / 时间与展平密度矩阵。
        Outputs / 输出: Flattened derivative / 展平导数。
        """
        matrix = flat.reshape(dim, dim)
        return (-1j * (hamiltonian @ matrix - (hamiltonian @ matrix.T).T)
                - damping * matrix).ravel()

    samples = np.asarray(sample_times, dtype=float)
    evaluation = np.unique(np.append(samples, duration))
    if len(evaluation) * dim * dim * np.dtype(complex).itemsize > 256 * 1024**2:
        raise ValueError("sampled master density history exceeds 256 MiB; reduce sampling density")
    solution = solve_ivp(rhs, (0.0, duration), rho.ravel(), method="DOP853",
                         t_eval=evaluation, rtol=2e-10, atol=2e-12)
    if not solution.success:
        raise RuntimeError(f"Lindblad integration failed: {solution.message}")
    probabilities = solution.y[target_index * dim + target_index].real
    if probabilities.min(initial=0) < -2e-8 or probabilities.max(initial=1) > 1 + 2e-8:
        raise RuntimeError("density integration violated probability bounds")
    return np.clip(probabilities[np.searchsorted(evaluation, samples)], 0, 1), solution.y[:, -1].reshape(dim, dim)


def _interval_edges(plan: TransferPlan) -> tuple[tuple[tuple[int, int, float], ...], ...]:
    """Purpose / 功能: Reuse exactly the planned switching schedule / 精确复用计划开关时序。

    Inputs / 输入: Frozen transfer plan / 冻结传输方案。
    Outputs / 输出: Real XY edges for each constant-H interval / 各恒定 H 区间实耦合边。
    """
    result = []
    for active in plan.interval_primitive_indices:
        edges = []
        for index in active:
            primitive = plan.primitives[index]
            left, right = primitive.sites
            if primitive.kind == "direct-xy":
                if abs(left - right) != 1:
                    raise ValueError("strict NN noise requires adjacent direct edges")
                edges.append((left, right, primitive.effective_g_rad_per_ns))
            elif primitive.kind == "pst-segment":
                length = right - left + 1
                edges.extend((left + j - 1, left + j,
                              primitive.effective_g_rad_per_ns * math.sqrt(j * (length - j)))
                             for j in range(1, length))
            else:
                raise ValueError(f"unsupported noise primitive: {primitive.kind}")
        result.append(tuple(edges))
    return tuple(result)


def _master_probabilities(plan: TransferPlan, times: np.ndarray, gamma: float) -> np.ndarray:
    """Purpose / 功能: Propagate small fixed-k dephasing density matrices / 传播小规模固定 k 退相干密度矩阵。

    Inputs / 输入: Schedule, sample times and gamma / 时序、采样时刻与退相干率。
    Outputs / 输出: Dephasing-only target probabilities / 仅纯退相干目标概率。
    """
    from .chain_evolution import build_edge_hamiltonian
    n, k = plan.config.chain_length, len(plan.config.pairs)
    basis = list(combinations(range(n), k))
    index = {state: i for i, state in enumerate(basis)}
    dim = len(basis)
    occupations = np.zeros((dim, n), dtype=np.int8)
    for row, state in enumerate(basis):
        occupations[row, list(state)] = 1
    # 每个占据不同的站点贡献 gamma；单粒子非对角元衰减率为 2gamma。
    # Each differing occupation contributes gamma: single-particle coherences decay at 2gamma.
    damping = gamma * (2 * k - 2 * (occupations.astype(float) @ occupations.T))
    rho = np.zeros((dim, dim), dtype=complex)
    rho[index[tuple(sorted(plan.config.sources))], index[tuple(sorted(plan.config.sources))]] = 1
    target = index[tuple(sorted(plan.config.targets))]
    output = np.full(len(times), float(tuple(sorted(plan.config.sources)) == tuple(sorted(plan.config.targets))))
    start = 0.0
    for edges, duration in zip(_interval_edges(plan), plan.interval_durations, strict=True):
        selected = np.flatnonzero((times > start) & (times <= start + duration + 1e-9))
        local_times = np.clip(times[selected] - start, 0.0, duration)
        hamiltonian = build_edge_hamiltonian(basis, index, edges)
        output[selected], rho = _density_interval(hamiltonian, damping, rho, duration, local_times, target)
        start += duration
    return output


def _direct_log_probabilities(plan: TransferPlan, times: np.ndarray, gamma: float) -> np.ndarray:
    """Purpose / 功能: Solve independent physical spin pairs exactly / 精确求独立物理自旋对。

    Inputs / 输入: Disjoint direct-edge plan, times and gamma / 不相交直接边方案、时间与退相干率。
    Outputs / 输出: Product dephasing log probabilities, no JW long-edge approximation / 退相干乘积对数概率，不误用长边费米子近似。
    """
    logs = np.zeros(len(times))
    for primitive in plan.primitives:
        if (primitive.kind != "direct-xy" or primitive.start_time_ns != 0
                or abs(primitive.sites[0] - primitive.sites[1]) != 1):
            raise ValueError("independent-pair backend requires simultaneous disjoint direct edges")
        t = np.minimum(times, primitive.duration_ns)
        coupling = primitive.effective_g_rad_per_ns
        discriminant = 4 * coupling * coupling - gamma * gamma
        if abs(discriminant) <= 1e-14 * max(gamma * gamma, coupling * coupling):
            z = np.exp(-gamma * t) * (1 + gamma * t)
        elif discriminant > 0:
            frequency = math.sqrt(discriminant)
            z = np.exp(-gamma * t) * (np.cos(frequency * t) + gamma / frequency * np.sin(frequency * t))
        else:
            frequency = math.sqrt(-discriminant)
            # 分开指数避免 cosh 溢出。 Split exponentials avoid overflow in the overdamped regime.
            z = 0.5 * ((1 + gamma / frequency) * np.exp((-gamma + frequency) * t)
                       + (1 - gamma / frequency) * np.exp((-gamma - frequency) * t))
        probability = np.clip((1 - z) / 2, 0, 1)
        with np.errstate(divide="ignore"):
            logs += np.log(probability)
    return logs


def _slater_system(plan: TransferPlan) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    """Purpose / 功能: Diagonalize only legal nearest-neighbor one-particle Hamiltonians / 仅对合法最近邻单粒子 H 对角化。

    Inputs / 输入: Nearest-neighbor open-chain plan / 开放最近邻链方案。
    Outputs / 输出: Eigenvalues/vectors per interval; explicit resource/topology errors / 各区间本征系，资源或拓扑超限时报错。
    """
    n = plan.config.chain_length
    if n > MAX_SLATER_SITES or len(plan.interval_durations) > MAX_SLATER_INTERVALS:
        raise ValueError("Slater backend exceeds supported site/interval limits (512/1024)")
    if len(plan.interval_durations) * n * n * 8 > 256 * 1024**2:
        raise ValueError("Slater eigensystem storage exceeds 256 MiB safety limit")
    systems = []
    for edges in _interval_edges(plan):
        matrix = np.zeros((n, n))
        for left, right, coupling in edges:
            if abs(left - right) != 1:
                raise ValueError("Slater trajectories require open nearest-neighbor spin XY, not long edges")
            matrix[left, right] += coupling
            matrix[right, left] += coupling
        systems.append(eigh(matrix, check_finite=False))
    return tuple(systems)


def _slater_path(plan: TransferPlan, times: np.ndarray, systems,
                  jump_times: np.ndarray, jump_sites: np.ndarray) -> np.ndarray:
    """Purpose / 功能: Evolve orthonormal orbitals with local Z jumps / 以局域 Z 跃迁传播正交轨道。

    Inputs / 输入: Plan, observation grid, spectra and phase-jump events / 方案、观测网格、本征系及相位跃迁事件。
    Outputs / 输出: Target log determinant probabilities; observations never alter evolution / 目标行列式对数概率，观测不改变演化。
    """
    n, k = plan.config.chain_length, len(plan.config.pairs)
    orbitals = np.eye(n, dtype=complex)[:, sorted(plan.config.sources)]
    target = sorted(plan.config.targets)
    output = np.full(len(times), -math.inf)
    same = sorted(plan.config.sources) == target
    output[times == 0] = 0.0 if same else -math.inf
    start, jump_index = 0.0, 0
    for (eigenvalues, eigenvectors), duration in zip(systems, plan.interval_durations, strict=True):
        end, current = start + duration, start
        spectral = eigenvectors.T @ orbitals
        selected = np.flatnonzero((times > start) & (times <= end + 1e-9))
        observation_index = 0
        while True:
            event_time = float(jump_times[jump_index]) if jump_index < len(jump_times) else math.inf
            stop = min(event_time, end)
            while observation_index < len(selected) and times[selected[observation_index]] <= stop + 1e-10:
                row = selected[observation_index]
                evolved_target = eigenvectors[target] @ (np.exp(-1j * eigenvalues * (times[row] - current))[:, None] * spectral)
                _, logabs = np.linalg.slogdet(evolved_target)
                output[row] = min(0.0, float(2 * logabs))
                observation_index += 1
            spectral = np.exp(-1j * eigenvalues * (stop - current))[:, None] * spectral
            current = stop
            if event_time >= end:
                orbitals = eigenvectors @ spectral
                break
            orbitals = eigenvectors @ spectral
            orbitals[jump_sites[jump_index], :] *= -1
            spectral = eigenvectors.T @ orbitals
            jump_index += 1
        start = end
    return output


def _trajectory_probabilities(plan: TransferPlan, times: np.ndarray, gamma: float,
                              noise: NoiseConfig, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """Purpose / 功能: Stratified bounded-MC Slater dephasing / 分层有界 MC Slater 退相干。

    Inputs / 输入: NN plan, times, rate, budget and stable seed / 最近邻方案、时间、退相干率、预算与稳定种子。
    Outputs / 输出: Log mean, SE, log CI bounds, honest adaptive diagnostics / 对数均值、标准误、对数区间及诚实精度诊断。
    """
    systems = _slater_system(plan)
    nojump = _slater_path(plan, times, systems, np.array([]), np.array([], dtype=int))
    mu = plan.config.chain_length * gamma * plan.transfer_time_ns / 2
    if mu == 0:
        return nojump, np.zeros(len(times)), nojump.copy(), nojump.copy(), {"trajectories": 0, "precision_met": True, "budget_exhausted": False}
    if mu > 10000:
        raise ValueError("expected phase-jump count exceeds supported 10000 per trajectory")
    weight = -math.expm1(-mu)
    log_weight = math.log(weight)
    rng = np.random.default_rng(seed)
    mean, m2 = np.zeros(len(times)), np.zeros(len(times))
    logsum = np.full(len(times), -math.inf)
    checkpoints = list(range(noise.min_trajectories, noise.trajectories + 1, noise.min_trajectories))
    if not checkpoints or checkpoints[-1] != noise.trajectories:
        checkpoints.append(noise.trajectories)
    # 每个观测点为边际区间；跨自适应检查点作联合界，不声称整曲线同时覆盖。
    # Marginal per-time intervals; allocate delta to both tails and all checkpoints.
    # 经验 Bernstein 单尾使用 ln(2/delta)，双尾及全部检查点因此采用 ln(4M/delta)。
    # Empirical Bernstein uses ln(2/delta) per tail, hence ln(4M/delta) here.
    concentration = math.log(4 * len(checkpoints) / (1 - noise.confidence))
    checkpoint_set = set(checkpoints)
    loss = len(plan.config.pairs) * times / noise.t1_ns
    precision_met = False
    for count in range(1, noise.trajectories + 1):
        if mu < 30:
            quantile = min(np.nextafter(1.0, 0.0), math.exp(-mu) + weight * rng.random())
            events = max(1, int(poisson.ppf(quantile, mu)))
        else:
            events = int(rng.poisson(mu))
            while events == 0:
                events = int(rng.poisson(mu))
        jump_times = np.sort(rng.uniform(0, plan.transfer_time_ns, events))
        jump_sites = rng.integers(0, plan.config.chain_length, events)
        sample_log = _slater_path(plan, times, systems, jump_times, jump_sites)
        sample = np.exp(sample_log)
        delta = sample - mean
        mean += delta / count
        m2 += delta * (sample - mean)
        logsum = np.logaddexp(logsum, sample_log)
        if count not in checkpoint_set:
            continue
        variance = np.maximum(0.0, m2 / (count - 1))
        radius = np.minimum(1.0, np.sqrt(2 * variance * concentration / count)
                            + 7 * concentration / (3 * (count - 1)))
        log_mean = np.logaddexp(-mu + nojump, log_weight + logsum - math.log(count))
        with np.errstate(divide="ignore"):
            lower = np.logaddexp(-mu + nojump, log_weight + np.log(np.maximum(0.0, mean - radius)))
            upper = np.logaddexp(-mu + nojump, log_weight + np.log(np.minimum(1.0, mean + radius)))
        # 完成指标决定停止；观测网格不会改变样本预算或最终结果。
        # Completion determines stopping, independent of the requested observation grid.
        width = (math.exp(upper[-1] - loss[-1]) - math.exp(lower[-1] - loss[-1])) / 2
        precision_met = width <= noise.atol + noise.rtol * math.exp(log_mean[-1] - loss[-1])
        if precision_met:
            break
    standard_error = weight * np.sqrt(variance / count)
    return log_mean, standard_error, lower, upper, {
        "trajectories": count, "precision_met": bool(precision_met), "budget_exhausted": not precision_met,
        "confidence_method": "no-jump-stratified empirical Bernstein; union over adaptive checkpoints",
        "curve_interval_semantics": "marginal per time, not simultaneous whole-curve coverage",
        "no_jump_probability": math.exp(-mu), "phase_jump_rate_ns": plan.config.chain_length * gamma / 2,
    }


def _safe_log(values: np.ndarray) -> np.ndarray:
    """Purpose / 功能: Preserve zero as negative infinity / 保留零概率为负无穷。

    Inputs / 输入: Nonnegative probabilities / 非负概率。
    Outputs / 输出: Natural logarithms without epsilon clipping / 不作 epsilon 截断的自然对数。
    """
    with np.errstate(divide="ignore"):
        return np.log(values)


def evaluate_plan_noise(plan: TransferPlan, times: Sequence[float], noise: NoiseConfig | None = None) -> dict[str, object]:
    """Purpose / 功能: Evaluate shared dynamical noise on the actual schedule / 在实际时序上求共享动力学噪声。

    Inputs / 输入: Frozen plan, increasing ns samples ending at completion, optional noise / 冻结方案、递增且含完成时刻的采样、可选噪声。
    Outputs / 输出: JSON-safe probability/error/log curves and metadata / 可序列化概率、误差、对数曲线及元数据。
    """
    noise = plan.config.noise if noise is None else noise
    grid = tuple(float(time) for time in times)
    if not grid or any(not math.isfinite(time) or time < 0 for time in grid):
        raise ValueError("noise sampling times must be finite and nonnegative")
    if any(right <= left for left, right in zip(grid, grid[1:])):
        raise ValueError("noise sampling times must be strictly increasing")
    if not math.isclose(grid[-1], plan.transfer_time_ns, rel_tol=1e-11, abs_tol=1e-9):
        raise ValueError("noise samples must end at the planned completion time")
    # 返回新容器，不暴露缓存中的可变字典。 Return fresh containers, never cached mutable dictionaries.
    import copy
    return copy.deepcopy(_cached_evaluate(plan, grid, noise, NOISE_MODEL_VERSION))


@lru_cache(maxsize=128)
def _cached_evaluate(plan: TransferPlan, grid: tuple[float, ...], noise: NoiseConfig, version: str) -> dict[str, object]:
    """Purpose / 功能: Cache content-addressed physical results / 缓存按内容寻址的物理结果。

    Inputs / 输入: Immutable schedule, grid, all noise controls and version / 不可变时序、网格、全部噪声控制及版本。
    Outputs / 输出: Internal cached result, copied by the public API / 内部缓存结果，公共接口复制后返回。
    """
    times = np.asarray(grid)
    if plan.config.ideal:
        reference = {"status": "not fitted: explicit ideal control", "sites": 6,
                     "reference_time_ns": noise.calibration_time_ns,
                     "reference_fidelity": noise.calibration_fidelity,
                     "calibration_t1_ns": None if math.isinf(noise.calibration_t1_ns) else noise.calibration_t1_ns,
                     "actual_t1_used_for_fit": False}
        gamma, t1_ns = 0.0, math.inf
    else:
        if noise.gamma_phi_ns is None:
            reference = calibration_reference(noise)
            gamma = reference["gamma_phi_ns"]
            reference["gamma_source"] = "deterministic fixed-calibration-T1 reference fit"
        else:
            gamma = noise.gamma_phi_ns
            reference = {"status": "explicit gamma supplied; no reference fit performed",
                         "sites": 6, "topology": "open-nearest-neighbor-pst",
                         "reference_time_ns": noise.calibration_time_ns,
                         "reference_fidelity": noise.calibration_fidelity,
                         "calibration_t1_ns": None if math.isinf(noise.calibration_t1_ns) else noise.calibration_t1_ns,
                         "gamma_phi_ns": gamma, "gamma_source": "explicit NoiseConfig.gamma_phi_ns",
                         "actual_t1_used_for_fit": False}
        t1_ns = noise.t1_ns
    n, k = plan.config.chain_length, len(plan.config.pairs)
    loss = k * times / t1_ns
    seed_key = (version, n, plan.config.pairs, plan.primitives, plan.interval_durations,
                plan.interval_primitive_indices, gamma, t1_ns, noise.seed)
    seed = int.from_bytes(hashlib.sha256(repr(seed_key).encode()).digest()[:8], "big")
    diagnostics = {"trajectories": 0, "precision_met": True, "budget_exhausted": False}
    errors = np.zeros(len(times))
    if not plan.primitives or k == n:
        log_dephasing = np.zeros(len(times))
        solver = "exact-invariant-occupation"
    elif plan.route == "direct-xy" and noise.solver != "master":
        log_dephasing = _direct_log_probabilities(plan, times, gamma)
        solver = "exact-independent-pairs"
    else:
        # 仅用组合数判定安全阈值，不为大扇区构造 C(N,k) 状态。
        # Use the integer dimension only as a guard; never construct a large Fock sector.
        dim = math.comb(n, k)
        if noise.solver == "master" or (noise.solver == "auto" and dim <= noise.master_max_dim):
            if dim > noise.master_max_dim:
                raise ValueError(f"master sector dimension {dim} exceeds limit {noise.master_max_dim}")
            log_dephasing = _safe_log(_master_probabilities(plan, times, gamma))
            solver = "fixed-k-master"
        else:
            log_dephasing, errors, lower, upper, diagnostics = _trajectory_probabilities(plan, times, gamma, noise, seed)
            solver = "slater-phase-jump-trajectories" if gamma else "slater-unitary"
    if "lower" not in locals():
        lower, upper = log_dephasing.copy(), log_dephasing.copy()
    logs, lower, upper = log_dephasing - loss, lower - loss, upper - loss
    errors = errors * np.exp(-loss)
    probabilities, lo, hi = np.exp(logs), np.exp(lower), np.exp(upper)
    uncertainty = {**diagnostics, "standard_error": float(errors[-1]), "ci_lower": float(lo[-1]),
                   "ci_upper": float(hi[-1]), "log_ci_lower": float(lower[-1]), "log_ci_upper": float(upper[-1]),
                   "confidence": noise.confidence, "seed": seed, "requested_seed": int(noise.seed),
                   "scope": "completion target occupation", "numerical_rtol": 2e-10 if solver == "fixed-k-master" else None}
    parameters = {**asdict(noise), "resolved_gamma_phi_ns": gamma,
                  "actual_t1_ns": None if math.isinf(t1_ns) else t1_ns,
                  "tphi_ns": None if gamma == 0 else 1 / gamma,
                  "relaxation_disabled": math.isinf(t1_ns), "pure_dephasing_disabled": gamma == 0}
    for name in ("t1_ns", "calibration_t1_ns"):
        if math.isinf(parameters[name]):
            parameters[name] = None
    metadata = {"model_version": version, "topology": "open-nearest-neighbor-xy", "parameters": parameters, "reference_fit": reference,
                "solver": solver, "uncertainty": uncertainty, "excitation_count": k,
                "converged": bool(diagnostics["precision_met"]), "confidence": noise.confidence,
                "sampling_uncertainty_only": True,
                "numerical_error_note": "Deterministic zero-width intervals denote no sampling error, not a rigorous integrator error bound.",
                "relaxation_neglog_fidelity": float(loss[-1]), "dephasing_log_fidelity": float(log_dephasing[-1]),
                "noise_scope": "all residents throughout actual batch schedule, including shutoff waits; excludes extra compute/inter-batch waits",
                "observable": "target occupation probability, not process fidelity",
                "dissipators": "sqrt(1/T1)*sigma_minus; sqrt(2*gamma_phi)*n_i"}
    return {"fidelities": tuple(map(float, probabilities)), "standard_errors": tuple(map(float, errors)),
            "ci_lower": tuple(map(float, lo)), "ci_upper": tuple(map(float, hi)),
            "log_fidelities": tuple(map(float, logs)), "metadata": metadata}


def noise_self_test() -> dict[str, object]:
    """Purpose / 功能: Run small independent physical regression checks / 执行小规模独立物理回归检查。

    Inputs / 输入: None; bounded deterministic fixtures and seeded MC / 无输入，有限确定性案例与固定种子 MC。
    Outputs / 输出: JSON-safe check names/calibration; raises on failure / 可序列化检查名与标定，失败即异常。
    """
    from dataclasses import replace
    from scipy.linalg import expm
    from scipy.sparse import csr_matrix
    from .chain_evolution import (SimulationConfig, build_edge_hamiltonian,
                                 build_single_chain_plan, estimate_transfer, plan_transfer,
                                 run_simulation, simulate_transfer)
    checks = []
    reference = calibration_reference()
    assert abs(reference["gamma_phi_ns"] - 5.6444919904e-5) < 2e-12
    assert abs(reference["achieved_fidelity"] - 0.992) < 2e-10
    assert calibration_reference(NoiseConfig(t1_ns=300000))["gamma_phi_ns"] == reference["gamma_phi_ns"]
    changed_timing = estimate_transfer([0], [5], 6, reference_time_ns=250, reference_fidelity=0.95)
    assert changed_timing["noise"]["parameters"]["resolved_gamma_phi_ns"] == reference["gamma_phi_ns"]
    checks.append("true-NN six-site calibration and frozen gamma under changed actual T1 or planner timing")
    try:
        calibration_reference(reference_fidelity=1.0)
    except ValueError:
        pass
    else:
        raise AssertionError("unreachable reference was accepted")
    checks.append("unreachable reference rejected")
    n, k = 4, 2
    config = SimulationConfig(n, ((0, 1), (1, 3)), ordered=True, num_steps=9)
    plan = plan_transfer(config)
    grid = np.linspace(0, plan.transfer_time_ns, 9)
    baseline = evaluate_plan_noise(plan, grid)
    zero = NoiseConfig(t1_ns=math.inf, gamma_phi_ns=0)
    ideal_result = run_simulation(replace(config, noise=zero))
    assert np.allclose(ideal_result[0].noisy_fidelities, ideal_result[-1], atol=2e-8)
    t1_only = run_simulation(replace(config, noise=NoiseConfig(gamma_phi_ns=0)))
    assert np.allclose(t1_only[0].noisy_fidelities,
                       t1_only[-1] * np.exp(-k * t1_only[1] / BASELINE_T1_NS), atol=2e-8)
    checks.append("zero noise restores ideal curves and T1-only gives exact all-resident loss")

    # 全 Hilbert 空间显式 sigma_minus Lindblad，独立验证扇区简化与系数。
    # Independent full-Hilbert sigma-minus Lindblad checks sector reduction and factors.
    dimension = 2**n
    source_mask = sum(1 << site for site in config.sources)
    target_mask = sum(1 << site for site in config.targets)
    rho = np.zeros((dimension, dimension), dtype=complex)
    rho[source_mask, source_mask] = 1
    lowers = []
    numbers = []
    for site in range(n):
        operator = np.zeros((dimension, dimension))
        for mask in range(dimension):
            if mask & (1 << site):
                operator[mask ^ (1 << site), mask] = 1
        lowers.append(operator)
        numbers.append(operator.T @ operator)
    gamma = reference["gamma_phi_ns"]
    jumps = [operator / math.sqrt(BASELINE_T1_NS) for operator in lowers]
    jumps += [math.sqrt(2 * gamma) * operator for operator in numbers]
    rates = [operator.conj().T @ operator for operator in jumps]
    for edges, duration in zip(_interval_edges(plan), plan.interval_durations, strict=True):
        hamiltonian = sum((coupling * (lowers[left].T @ lowers[right] + lowers[right].T @ lowers[left])
                           for left, right, coupling in edges), np.zeros((dimension, dimension)))

        def full_rhs(_time: float, flat: np.ndarray) -> np.ndarray:
            """Purpose / 功能: Independent full-Fock Lindblad derivative / 独立完整 Fock 主方程导数。

            Inputs / 输入: Time and full density matrix / 时间与完整密度矩阵。
            Outputs / 输出: Hamiltonian plus loss/dephasing derivative / 含哈密顿、损失与退相干导数。
            """
            matrix = flat.reshape(dimension, dimension)
            derivative = -1j * (hamiltonian @ matrix - matrix @ hamiltonian)
            for jump, rate in zip(jumps, rates, strict=True):
                derivative += jump @ matrix @ jump.conj().T - 0.5 * (rate @ matrix + matrix @ rate)
            return derivative.ravel()

        solution = solve_ivp(full_rhs, (0, duration), rho.ravel(), method="DOP853", rtol=3e-10, atol=2e-12)
        assert solution.success
        rho = solution.y[:, -1].reshape(dimension, dimension)
    assert abs(rho[target_mask, target_mask].real - baseline["fidelities"][-1]) < 2e-8
    assert abs(np.trace(rho).real - 1) < 2e-10
    checks.append("full-Hilbert Lindblad agrees with fixed-k plus exact uniform relaxation")

    rho_coherence = np.full((2, 2), 0.5, dtype=complex)
    _, damped = _density_interval(csr_matrix((2, 2)), gamma * (1 - np.eye(2)),
                                  rho_coherence, 100.0, (100.0,), 0)
    assert abs(damped[0, 1] - 0.5 * math.exp(-gamma * 100)) < 1e-12
    _, damped = _density_interval(csr_matrix((2, 2)), 2 * gamma * (1 - np.eye(2)),
                                  rho_coherence, 100.0, (100.0,), 0)
    assert abs(damped[0, 1] - 0.5 * math.exp(-2 * gamma * 100)) < 1e-12
    checks.append("single-site gamma and single-excitation 2gamma coherence damping")

    # 同一相位跃迁记录逐事件对照 Fock，不将长边错误视作自由费米子。
    # Compare identical phase-jump records event-by-event against the spin Fock sector.
    systems = _slater_system(plan)
    jump_times = np.asarray([0.11, 0.44, 0.73]) * plan.transfer_time_ns
    jump_sites = np.array([0, 2, 1])
    slater_log = _slater_path(plan, grid, systems, jump_times, jump_sites)
    basis = list(combinations(range(n), k))
    index = {state: row for row, state in enumerate(basis)}
    state = np.zeros(len(basis), dtype=complex)
    state[index[tuple(sorted(config.sources))]] = 1
    start = 0.0
    jump_index = 0
    fock = np.zeros(len(grid))
    fock[0] = abs(state[index[tuple(sorted(config.targets))]])**2
    for edges, duration in zip(_interval_edges(plan), plan.interval_durations, strict=True):
        hamiltonian = build_edge_hamiltonian(basis, index, edges).toarray()
        end, current = start + duration, start
        observations = [float(t) for t in grid if start < t <= end + 1e-9]
        events = [float(t) for t in jump_times if start < t < end]
        for stop in sorted(set(observations + events + [end])):
            state = expm(-1j * hamiltonian * (stop - current)) @ state
            current = stop
            if jump_index < len(jump_times) and abs(stop - jump_times[jump_index]) < 1e-9:
                site = jump_sites[jump_index]
                state *= np.array([-1 if site in occupation else 1 for occupation in basis])
                jump_index += 1
            matches = np.flatnonzero(np.isclose(grid, stop, atol=1e-9, rtol=0))
            if len(matches):
                fock[matches] = abs(state[index[tuple(sorted(config.targets))]])**2
        start = end
    assert np.allclose(np.exp(slater_log), fock, atol=2e-10)
    checks.append("Slater NN evolution and phase jumps match explicit multi-stage Fock paths")

    monte_carlo = NoiseConfig(solver="trajectories", trajectories=1024, min_trajectories=128,
                             seed=21, atol=1e-5, rtol=0)
    sampled = evaluate_plan_noise(plan, grid, monte_carlo)
    uncertainty = sampled["metadata"]["uncertainty"]
    assert uncertainty["ci_lower"] <= baseline["fidelities"][-1] <= uncertainty["ci_upper"]
    assert uncertainty["ci_lower"] < uncertainty["ci_upper"]
    assert uncertainty["budget_exhausted"] and not uncertainty["precision_met"]
    repeated = evaluate_plan_noise(plan, grid, monte_carlo)
    _cached_evaluate.cache_clear()
    assert repeated == evaluate_plan_noise(plan, grid, monte_carlo)
    completion = evaluate_plan_noise(plan, [plan.transfer_time_ns], monte_carlo)
    assert completion["fidelities"][-1] == sampled["fidelities"][-1]
    alternate = evaluate_plan_noise(plan, [plan.transfer_time_ns], replace(monte_carlo, seed=22))
    assert alternate["metadata"]["uncertainty"]["seed"] != uncertainty["seed"]
    assert alternate["ci_lower"][-1] <= baseline["fidelities"][-1] <= alternate["ci_upper"][-1]
    checks.append("bounded 95% MC intervals, unmet-budget flag, deterministic seeds, grid-independent completion")

    direct = plan_transfer(SimulationConfig(5, ((0, 1), (2, 3), (4, 4))))
    direct_grid = np.linspace(0, direct.transfer_time_ns, 11)
    direct_exact = evaluate_plan_noise(direct, direct_grid)
    direct_master = evaluate_plan_noise(direct, direct_grid, NoiseConfig(solver="master"))
    assert np.allclose(direct_exact["fidelities"], direct_master["fidelities"], atol=2e-8)
    direct_loss = evaluate_plan_noise(direct, [direct.transfer_time_ns], NoiseConfig(gamma_phi_ns=0))
    assert abs(direct_loss["log_fidelities"][-1] + 3 * direct.transfer_time_ns / BASELINE_T1_NS) < 1e-12
    checks.append("independent adjacent spin pairs match master; idle and already-finished residents keep losing")

    full = plan_transfer(SimulationConfig(3, ((0, 2), (1, 1), (2, 0)), ordered=True))
    full_noise = evaluate_plan_noise(full, [full.transfer_time_ns])
    assert full_noise["log_fidelities"][-1] == -3 * full.transfer_time_ns / BASELINE_T1_NS
    underflow = evaluate_plan_noise(full, [full.transfer_time_ns], NoiseConfig(t1_ns=0.001))
    assert underflow["fidelities"][-1] == 0 and math.isfinite(underflow["log_fidelities"][-1])
    assert math.isfinite(underflow["metadata"]["uncertainty"]["log_ci_lower"])
    checks.append("full occupancy invariant and underflow-safe relaxation/log confidence bounds")
    simulated = simulate_transfer([0, 1], [1, 3], 4)
    estimated = estimate_transfer([0, 1], [1, 3], 4)
    assert simulated["fidelity"] == estimated["fidelity"]
    fast = estimate_transfer([0], [2], 3, coupling_policy="cap-limited-fast")
    slow = estimate_transfer([0], [2], 3)
    assert fast["transfer_time_ns"] < slow["transfer_time_ns"]
    checks.append("public API completion agreement and cap-limited-fast changes actual timing")
    longest = build_single_chain_plan(80)
    longest_result = evaluate_plan_noise(longest, [longest.transfer_time_ns])
    assert abs(longest_result["fidelities"][-1] - 0.847857231) < 2e-8
    checks.append("N80 fixed-peak baseline uses deterministic master without large Liouvillian")
    return {"passed": True, "check_count": len(checks), "checks": checks,
            "calibration": reference, "n80_fidelity": longest_result["fidelities"][-1],
            "n80_time_ns": longest.transfer_time_ns, "mc_uncertainty": uncertainty}
