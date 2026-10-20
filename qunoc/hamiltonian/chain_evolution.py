#!/usr/bin/env python3
"""Automatic XY routing / 自动 XY 路由，固定激发数占据态模拟。

Disjoint moving endpoints use simultaneous direct XY edges. Any shared endpoint
sends the entire task through contiguous PST mirrors (greedy + beam heuristic).
Time is in ns, couplings in rad/ns. A user-provided N=6, k=1, 125 ns, F=0.992
anchor sets sqrt(distance) transfer times and exponential distance attenuation.
This phenomenological design rule is not an experimental reproduction.
Ideal and calibrated target probabilities are NOT labeled or process fidelities.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace
from functools import lru_cache
from itertools import combinations
from typing import Sequence

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import expm_multiply

__all__ = ["MODEL_VERSION", "SimulationConfig", "TransferPlan", "plan_transfer", "estimate_transfer",
           "run_simulation", "simulate_transfer"]

Pair = tuple[int, int]
Stage = tuple[Pair, ...]
Stages = tuple[Stage, ...]

# 用户指定的单激发校准点；不是逐字文献数值。 User-prescribed, not a literal paper quote.
REFERENCE_SITES = 6
REFERENCE_DISTANCE = REFERENCE_SITES - 1
REFERENCE_TIME_NS = 125.0
REFERENCE_FIDELITY = 0.992
REFERENCE_G_RAD_PER_NS = math.pi / (2.0 * REFERENCE_TIME_NS)
# 六站点参考剖面的峰值，不是宣称设备硬件极限。 Six-site peak used as a modeling cap.
DEFAULT_G_MAX_RAD_PER_NS = 3.0 * REFERENCE_G_RAD_PER_NS
MODEL_VERSION = "hamiltonian-net-v1"
PLAN_CACHE_MAXSIZE = 256
# 限制启发式组合爆炸；大任务直接使用多项式兜底。 Bound combinatorial search work.
MAX_HEURISTIC_STEPS = 32
MAX_BEAM_EXPANSIONS = 20000
MAX_BEAM_WIDTH = 128
MAX_BEAM_OCCUPANTS = 8
MAX_BEAM_CHAIN_LENGTH = 64
MAX_ADJACENT_FALLBACK_SPAN = 512


@dataclass(frozen=True)
class SimulationConfig:
    chain_length: int
    pairs: tuple[Pair, ...]
    g_max: float | None = DEFAULT_G_MAX_RAD_PER_NS
    num_steps: int = 401
    segments: Stages | None = None
    max_segment_steps: int = 10
    beam_width: int = 128
    reference_time_ns: float = REFERENCE_TIME_NS
    reference_distance: float = REFERENCE_DISTANCE
    reference_fidelity: float = REFERENCE_FIDELITY
    ideal: bool = False
    # 独立数值接口保留占据目标；估计接口默认要求标签逐一到达。 Legacy occupation default.
    ordered: bool = False

    @property
    def attenuation_per_edge(self) -> float:
        """功能 / Purpose: Fit the distance attenuation coefficient / 拟合距离衰减系数。

        输入 / Inputs: Reference probability, distance and ideal flag / 参考概率、距离与无损标记。
        输出 / Outputs: alpha=-ln(F_ref)/d_ref per edge, or zero / 每边衰减系数或零。
        """
        return 0.0 if self.ideal else -math.log(self.reference_fidelity) / self.reference_distance

    @property
    def active_pairs(self) -> tuple[Pair, ...]:
        """功能 / Purpose: Keep only moving requests / 提取非静止请求。

        输入 / Inputs: self.pairs, including identity pairs / 含 i:i 的请求。
        输出 / Outputs: Non-self pair tuple / 不含自耦合的 pair 元组。
        """
        # i:i 是静止占据，保留在初/末态但不产生自耦合。 Identity pairs need no XY edge.
        return tuple((source, target) for source, target in self.pairs if source != target)

    @property
    def sources(self) -> tuple[int, ...]:
        """功能 / Purpose: Derive source occupations / 推导源占据。

        输入 / Inputs: Ordered self.pairs / 有序请求对。
        输出 / Outputs: Source tuple in request order / 按请求顺序的源站点元组。
        """
        return tuple(source for source, _ in self.pairs)

    @property
    def targets(self) -> tuple[int, ...]:
        """功能 / Purpose: Derive target occupations / 推导目标占据。

        输入 / Inputs: Ordered self.pairs / 有序请求对。
        输出 / Outputs: Target tuple in request order / 按请求顺序的目标站点元组。
        """
        return tuple(target for _, target in self.pairs)


@dataclass(frozen=True)
class TransferPrimitive:
    """Immutable calibrated mirror/edge / 不可变校准镜像或直接边。

    输入 / Inputs: Stage, endpoints, costs and independent timing / 阶段、端点、成本及独立时序。
    输出 / Outputs: Scalar-only primitive record, no matrix / 仅标量的操作记录，无矩阵。
    """

    stage_index: int
    kind: str
    sites: Pair
    span_distance: int
    transport_distance: int
    requested_duration_ns: float
    duration_ns: float
    start_time_ns: float
    end_time_ns: float
    effective_g_rad_per_ns: float
    max_coupling_rad_per_ns: float
    cap_limited: bool


@dataclass(frozen=True)
class TransferPlan:
    """Immutable matrix-free schedule / 不可变无矩阵调度。

    输入 / Inputs: Validated configuration and calibrated route records / 合法配置及校准路线记录。
    输出 / Outputs: Reusable tuple-only plan; no basis or mutable dictionaries / 可复用元组方案，无基底或可变字典。
    """

    config: SimulationConfig
    route: str
    planner: str
    direct_pairs: tuple[Pair, ...]
    segment_stages: Stages
    primitives: tuple[TransferPrimitive, ...]
    stage_durations: tuple[float, ...]
    stage_requested_durations: tuple[float, ...]
    interval_durations: tuple[float, ...]
    interval_primitive_indices: tuple[tuple[int, ...], ...]
    labeled_path: tuple[tuple[int, ...], ...]
    transfer_time_ns: float
    effective_distance_edges: int

    def to_route_details(self) -> dict[str, object]:
        """功能 / Purpose: Make independent legacy-compatible audit metadata / 生成独立兼容审计元数据。

        输入 / Inputs: This immutable plan / 当前不可变方案。
        输出 / Outputs: Fresh nested dictionaries/lists, safe for caller mutation / 可安全修改的新嵌套字典和列表。
        """
        return {
            "model_version": MODEL_VERSION, "ordered": self.config.ordered,
            "planner": self.planner, "pairs_are_disjoint": self.route == "direct-xy",
            "direct_pairs": self.direct_pairs, "segment_stages": self.segment_stages,
            "manual_segments": self.config.segments is not None,
            "stage_count": len(self.stage_durations),
            "stage_requested_durations": list(self.stage_requested_durations),
            "stage_durations": list(self.stage_durations),
            "stage_end_times": list(_cumulative_times(self.stage_durations)),
            "hamiltonian_interval_count": len(self.interval_durations),
            "interval_durations": list(self.interval_durations),
            "interval_end_times": list(_cumulative_times(self.interval_durations)),
            "primitives": [asdict(primitive) for primitive in self.primitives],
            "total_transport_distance": self.effective_distance_edges,
            "occupation_path": [tuple(sorted(positions)) for positions in self.labeled_path],
            "labeled_path": list(self.labeled_path),
        }


def _cumulative_times(durations: Sequence[float]) -> tuple[float, ...]:
    """功能 / Purpose: Sum schedule boundaries without arrays / 无数组累加调度边界。

    输入 / Inputs: Ordered durations / 有序时长。
    输出 / Outputs: Cumulative end times / 累计结束时刻元组。
    """
    elapsed = 0.0
    ends = []
    for duration in durations:
        elapsed += duration
        ends.append(elapsed)
    return tuple(ends)


@dataclass(frozen=True)
class SimulationResult:
    config: SimulationConfig
    route: str
    hilbert_dim: int
    sample_count: int
    peak_time: float
    peak_fidelity: float
    final_time: float
    final_fidelity: float
    calibrated_peak_time: float
    calibrated_peak_fidelity: float
    calibrated_final_fidelity: float
    route_details: dict[str, object]


def are_disjoint_edges(edges: Sequence[Pair]) -> bool:
    """功能 / Purpose: Check endpoint conflicts / 检查端点冲突。

    输入 / Inputs: Sequence of active pairs / 活动 pair 序列。
    输出 / Outputs: True iff all endpoints are distinct / 端点完全不重复时返回真。
    """
    sites = [site for edge in edges for site in edge]
    return len(sites) == len(set(sites))


def validate_config(config: SimulationConfig) -> None:
    """功能 / Purpose: Validate requests and controls / 校验请求与控制参数。

    输入 / Inputs: SimulationConfig / 模拟配置。
    输出 / Outputs: None; invalid input raises ValueError / 无返回值；非法输入时报错。
    """
    if isinstance(config.chain_length, bool) or not isinstance(config.chain_length, (int, np.integer)) or config.chain_length < 1:
        raise ValueError("--chain-length must be a positive integer")
    if not config.pairs:
        raise ValueError("--pairs must contain at least one source:target pair")
    for name, positions in (("sources", config.sources), ("targets", config.targets)):
        if any(isinstance(site, bool) or not isinstance(site, (int, np.integer)) for site in positions):
            raise ValueError(f"{name} must contain integer site indices")
        if len(set(positions)) != len(positions):
            raise ValueError(f"{name} must be unique")
        if any(site < 0 or site >= config.chain_length for site in positions):
            raise ValueError(f"{name} out of range [0, {config.chain_length - 1}]")
    for name, value in (("--reference-time-ns", config.reference_time_ns),
                        ("--reference-distance", config.reference_distance)):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if config.g_max is not None and (not math.isfinite(config.g_max) or config.g_max <= 0):
        raise ValueError("--g-max must be finite and positive")
    if not math.isfinite(config.reference_fidelity) or not 0 < config.reference_fidelity <= 1:
        raise ValueError("--reference-fidelity must be finite and in (0, 1]")
    for name, value, minimum in (("--num-steps", config.num_steps, 2),
                                 ("--max-segment-steps", config.max_segment_steps, 1),
                                 ("--beam-width", config.beam_width, 1)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
            raise ValueError(f"{name} must be an integer at least {minimum}")
    if not isinstance(config.ordered, bool):
        raise ValueError("ordered must be a boolean")
    if config.segments is not None:
        if are_disjoint_edges(config.active_pairs):
            raise ValueError("--segments is only used for conflicting pairs; disjoint pairs use direct XY")
        validate_segment_stages(config.chain_length, config.segments)


def build_fixed_excitation_basis(
    chain_length: int, excitation_count: int
) -> tuple[list[tuple[int, ...]], dict[tuple[int, ...], int]]:
    """功能 / Purpose: Enumerate fixed-k occupations / 枚举固定激发数占据。

    输入 / Inputs: Chain length N and excitation count k / 链长与激发数。
    输出 / Outputs: Basis list and state-to-index map, size C(N,k) / 基底与索引字典。
    """
    basis = list(combinations(range(chain_length), excitation_count))
    return basis, {state: i for i, state in enumerate(basis)}


def build_edge_hamiltonian(
    basis: Sequence[tuple[int, ...]],
    basis_index: dict[tuple[int, ...], int],
    edges: Sequence[tuple[int, int, complex]],
) -> csr_matrix:
    """功能 / Purpose: Build hard-core XY hopping / 构造硬核 XY 跃迁。

    输入 / Inputs: Basis, index, directed (u,v,J) edges / 占据基底、索引、含耦合的边。
    输出 / Outputs: Hermitian CSR matrix; conjugate reverse hops / 厄米稀疏矩阵，反向耦合共轭。
    """
    rows: list[int] = []
    cols: list[int] = []
    data: list[complex] = []
    # 不允许双占据；复耦合使用共轭反向跃迁。 No double occupancy; Hermitian reverse hop.
    for col, state in enumerate(basis):
        occupied = set(state)
        for left, right, coupling in edges:
            if left in occupied and right not in occupied:
                moved = tuple(sorted((occupied - {left}) | {right}))
                rows.append(basis_index[moved])
                cols.append(col)
                data.append(coupling)
            if right in occupied and left not in occupied:
                moved = tuple(sorted((occupied - {right}) | {left}))
                rows.append(basis_index[moved])
                cols.append(col)
                data.append(np.conjugate(coupling))
    dim = len(basis)
    return coo_matrix((data, (rows, cols)), shape=(dim, dim), dtype=np.complex128).tocsr()


def engineered_pst_couplings(length: int, g_base: float = 1.0) -> np.ndarray:
    """功能 / Purpose: Build the engineered PST profile / 构造工程化 PST 剖面。

    输入 / Inputs: Segment length L and scale g_base / 段长与耦合尺度。
    输出 / Outputs: Array J_j=g_base*sqrt(j*(L-j)), j=1..L-1 / 最近邻耦合数组。
    """
    js = np.arange(1, length, dtype=float)
    return g_base * np.sqrt(js * (length - js))


def validate_segment_stages(chain_length: int, stages: Stages) -> None:
    """功能 / Purpose: Validate disjoint contiguous intervals / 校验连续区间与阶段。

    输入 / Inputs: Chain length and nonempty manual stages / 链长与非空手动方案。
    输出 / Outputs: None; invalid/overlapping intervals raise ValueError / 无返回；非法或重叠时报错。
    """
    if not stages:
        raise ValueError("manual --segments must contain at least one nonempty stage")
    for stage_idx, stage in enumerate(stages, start=1):
        if not stage:
            raise ValueError(f"PST stage {stage_idx} is empty")
        previous_right = -1
        for left, right in sorted(stage):
            if any(isinstance(site, bool) or not isinstance(site, (int, np.integer)) for site in (left, right)):
                raise ValueError(f"PST stage {stage_idx} endpoints must be integer site indices")
            if not 0 <= left < right < chain_length:
                raise ValueError(f"PST stage {stage_idx} has invalid segment ({left}, {right})")
            if left <= previous_right:
                raise ValueError(f"PST stage {stage_idx} has overlapping segments")
            previous_right = right


def apply_segment_stage_labeled(positions: Sequence[int], stage: Stage) -> tuple[int, ...]:
    """功能 / Purpose: Mirror all contained positions / 镜像段内全部位置。

    输入 / Inputs: Ordered positions and disjoint closed intervals / 有序位置与不相交闭区间。
    输出 / Outputs: Mirrored positions retaining input order / 保留输入顺序的镜像位置。
    """
    return tuple(next((left + right - pos for left, right in stage if left <= pos <= right), pos)
                 for pos in positions)


def apply_segment_stages(positions: Sequence[int], stages: Stages) -> tuple[int, ...]:
    """功能 / Purpose: Apply mirrors sequentially / 顺序施加镜像。

    输入 / Inputs: Initial positions and stage sequence / 初始位置与阶段序列。
    输出 / Outputs: Sorted final occupation tuple / 排序后的最终占据元组。
    """
    current = tuple(positions)
    for stage in stages:
        current = apply_segment_stage_labeled(current, stage)
    return tuple(sorted(current))


def candidate_segments_for_mapping(pos: int, target: int, chain_length: int) -> list[Pair]:
    """功能 / Purpose: Find long-first mirror candidates / 搜索长段优先候选。

    输入 / Inputs: Current position, target and chain length / 当前位置、目标与链长。
    输出 / Outputs: Intervals mapping pos to target; empty for a hold / 完成该映射的区间列表，静止时为空。
    """
    if pos == target:
        return []
    total = pos + target
    # 直接推导合法端点范围，避免扫描整条链。 Derive valid endpoints, rather than scanning N.
    first = max(0, total - chain_length + 1)
    last = min(pos, target, (total - 1) // 2)
    return [(left, total - left) for left in range(first, last + 1)]


def segment_target_displacement(segment: Pair, state: Sequence[int], targets: Sequence[int]) -> int:
    """功能 / Purpose: Score displacement solved by an interval / 评价区间解决的位移。

    输入 / Inputs: Interval, ordered current positions and target hints / 区间、有序位置及目标提示。
    输出 / Outputs: Largest exactly solved displacement, or zero / 最大精确解决位移，缺省零。
    """
    left, right = segment
    return max((abs(pos - target) for pos, target in zip(state, targets, strict=True)
                if left <= pos <= right and left + right - pos == target), default=0)


def labeled_stage_candidates(state: Sequence[int], targets: Sequence[int], chain_length: int) -> list[Stage]:
    """功能 / Purpose: Build bounded beam-search candidates / 构造有限束搜索候选。

    输入 / Inputs: Ordered positions, paired targets, small chain length / 有序位置、逐对目标和小链长度。
    输出 / Outputs: Ranked single-/two-interval stages / 排序后含一或两个区间的阶段列表。
    """
    segments = {segment for pos, target in zip(state, targets, strict=True)
                for segment in candidate_segments_for_mapping(pos, target, chain_length)[:8]}
    ranked = sorted(segments, key=lambda seg: (-segment_target_displacement(seg, state, targets),
                                              -(seg[1] - seg[0] + 1), seg))[:32]
    stages = [(segment,) for segment in ranked]
    for first, second in combinations(ranked, 2):
        if first[1] < second[0] or second[1] < first[0]:
            stages.append(tuple(sorted((first, second))))
    return sorted(stages, key=lambda stage: (
        -sum(segment_target_displacement(segment, state, targets) for segment in stage),
        -len(stage), -sum(right - left + 1 for left, right in stage), stage))


def mapping_matches(state: Sequence[int], targets: Sequence[int], ordered: bool) -> bool:
    """功能 / Purpose: Check the requested goal semantics / 检查请求目标语义。

    输入 / Inputs: Current label positions, targets, ordered flag / 当前标签位置、目标及有序标志。
    输出 / Outputs: Exact paired equality, or legacy occupation equality / 逐对相等或旧占据集合相等。
    """
    return tuple(state) == tuple(targets) if ordered else tuple(sorted(state)) == tuple(sorted(targets))


def labeled_segment_score(
    state: Sequence[int], targets: Sequence[int], ordered: bool = False,
) -> tuple[int, int, int]:
    """功能 / Purpose: Rank paired or occupation mismatches / 评价逐对或占据偏差。

    输入 / Inputs: Ordered current positions, goals and goal semantics / 有序当前位置、目标及目标语义。
    输出 / Outputs: Ordered-first mismatch/L1 score, or legacy hint score / 标签优先偏差分数或旧提示分数。
    """
    occupation = len(set(state) ^ set(targets))
    mismatches = sum(pos != target for pos, target in zip(state, targets, strict=True))
    distance = sum(abs(pos - target) for pos, target in zip(state, targets, strict=True))
    return (mismatches, distance, occupation) if ordered else (occupation, mismatches, distance)


def greedy_pst_segment_stages(
    sources: Sequence[int], targets: Sequence[int], chain_length: int, max_steps: int,
    ordered: bool = False,
) -> Stages | None:
    """功能 / Purpose: Try long-mirror greedy routing / 尝试长镜像贪心路由。

    输入 / Inputs: Sources, paired goals, chain length, budget and goal semantics / 源、逐对目标、链长、预算及目标语义。
    输出 / Outputs: Goal-valid route or None; holds remain paired constraints / 满足目标的路线或 None，静止仍为逐对约束。
    """
    current = tuple(sources)
    stages: list[Stage] = []
    seen = {current}
    for _ in range(max_steps):
        if mapping_matches(current, targets, ordered):
            return tuple(stages)
        unresolved = sorted(((pos, target) for pos, target in zip(current, targets, strict=True)
                             if pos != target), key=lambda pair: (-abs(pair[0] - pair[1]), pair))
        stage: list[Pair] = []
        for pos, target in unresolved:
            for left, right in candidate_segments_for_mapping(pos, target, chain_length):
                if all(right < a or b < left for a, b in stage):
                    stage.append((left, right))
                    break
        if not stage:
            return None
        stage_tuple = tuple(sorted(stage))
        current = apply_segment_stage_labeled(current, stage_tuple)
        if current in seen:
            return None
        seen.add(current)
        stages.append(stage_tuple)
    return tuple(stages) if mapping_matches(current, targets, ordered) else None


def adjacent_permutation_stages(sources: Sequence[int], targets: Sequence[int]) -> Stages:
    """功能 / Purpose: Complete any paired permutation with adjacent mirrors / 用相邻镜像完成任意逐对排列。

    输入 / Inputs: Unique valid sources and paired targets in a bounded window / 有界窗口内的唯一源及逐对目标。
    输出 / Outputs: Deterministic disjoint odd/even swap stages; anonymous holes are free / 确定性奇偶不相交交换阶段，匿名空槽不计成本。
    """
    low, high = min((*sources, *targets)), max((*sources, *targets))
    length = high - low + 1
    target_set = set(targets)
    holes = iter(site - low for site in range(low, high + 1) if site not in target_set)
    destination = dict(zip(sources, targets, strict=True))
    ranks = [destination[site] - low if site in destination else next(holes)
             for site in range(low, high + 1)]
    stages = []
    # 奇偶交换排序最多需窗口长度轮，不构造激发基底。 At most window-size sorting rounds.
    for turn in range(length):
        stage = []
        for left in range(turn % 2, length - 1, 2):
            if ranks[left] > ranks[left + 1]:
                ranks[left], ranks[left + 1] = ranks[left + 1], ranks[left]
                stage.append((left + low, left + low + 1))
        if stage:
            stages.append(tuple(stage))
    if ranks != list(range(length)):
        raise RuntimeError("adjacent permutation fallback failed its sorting invariant")
    return tuple(stages)


def sparse_permutation_stages(sources: Sequence[int], targets: Sequence[int]) -> Stages:
    """功能 / Purpose: Avoid enumerating huge empty gaps / 避免枚举巨大空白间隔。

    输入 / Inputs: Unique valid paired positions in an arbitrarily long chain / 任意长链的唯一逐对位置。
    输出 / Outputs: O(k) endpoint transpositions, each realized by one/two mirrors / O(k) 端点对换，每次由一或两个镜像实现。
    """
    current = tuple(sources)
    stages = []
    for index, target in enumerate(targets):
        source = current[index]
        if source == target:
            continue
        left, right = sorted((source, target))
        # 整段及内部镜像合成仅交换端点；中间居民先移动再恢复，距离会全计入。
        # Whole + interior reversal swaps only endpoints; spectators move and their cost is retained.
        transposition = [((left, right),)]
        if right - left > 2:
            transposition.append(((left + 1, right - 1),))
        for stage in transposition:
            current = apply_segment_stage_labeled(current, stage)
            stages.append(stage)
    if current != tuple(targets):
        raise RuntimeError("sparse permutation fallback failed its labeled invariant")
    return tuple(stages)


def _fallback_stages(sources: Sequence[int], targets: Sequence[int], ordered: bool) -> tuple[Stages, str]:
    """功能 / Purpose: Select a bounded exact permutation fallback / 选择有界精确排列兜底。

    输入 / Inputs: Valid positions and paired/occupation goal semantics / 合法位置及逐对或占据目标语义。
    输出 / Outputs: Stages and fallback strategy name / 阶段元组与兜底策略名称。
    """
    if not ordered:
        sources, targets = tuple(sorted(sources)), tuple(sorted(targets))
    span = max((*sources, *targets)) - min((*sources, *targets)) + 1
    if span <= MAX_ADJACENT_FALLBACK_SPAN:
        return adjacent_permutation_stages(sources, targets), "adjacent-fallback"
    return sparse_permutation_stages(sources, targets), "sparse-mirror-fallback"


def _auto_pst_plan(
    sources: Sequence[int], targets: Sequence[int], chain_length: int,
    max_steps: int, beam_width: int, ordered: bool,
) -> tuple[Stages, str]:
    """功能 / Purpose: Bound search and guarantee valid ordered mappings / 有界搜索并保证合法有序映射。

    输入 / Inputs: Valid paired positions, N, heuristic budgets and semantics / 合法逐对位置、链长、启发式预算及语义。
    输出 / Outputs: Route and strategy; small legacy occupation failures remain explicit / 路线及策略，小规模旧占据搜索失败仍显式报错。
    """
    if max_steps < 1 or beam_width < 1:
        raise ValueError("planner budgets must be positive")
    start = tuple(sources)
    if mapping_matches(start, targets, ordered):
        return (), "identity"
    # 大链/高占据跳过组合搜索；成本只与站点窗口或建模居民数量有关。
    # Large chains/high occupancy bypass combinatorial search, never build C(N,k) states.
    if chain_length > MAX_ADJACENT_FALLBACK_SPAN or len(start) > 64:
        return _fallback_stages(sources, targets, ordered)
    steps = min(max_steps, MAX_HEURISTIC_STEPS)
    greedy = greedy_pst_segment_stages(sources, targets, chain_length, steps, ordered)
    if greedy is not None:
        return greedy, "greedy"
    if len(start) > MAX_BEAM_OCCUPANTS or chain_length > MAX_BEAM_CHAIN_LENGTH:
        return _fallback_stages(sources, targets, ordered)
    beam: list[tuple[tuple[int, ...], Stages]] = [(start, ())]
    visited = {start}
    expansions = 0
    for _ in range(steps):
        candidates: dict[tuple[int, ...], Stages] = {}
        for state, path in beam:
            for stage in labeled_stage_candidates(state, targets, chain_length):
                expansions += 1
                if expansions > MAX_BEAM_EXPANSIONS:
                    if ordered:
                        return _fallback_stages(sources, targets, ordered)
                    raise ValueError("heuristic could not find a PST mirror route within its work bound")
                next_state = apply_segment_stage_labeled(state, stage)
                if next_state in visited:
                    continue
                next_path = path + (stage,)
                if mapping_matches(next_state, targets, ordered):
                    return next_path, "beam"
                candidates.setdefault(next_state, next_path)
        if not candidates:
            break
        beam = sorted(candidates.items(), key=lambda item: (
            labeled_segment_score(item[0], targets, ordered),
            -sum(right - left + 1 for stage in item[1] for left, right in stage), item[0]))[:min(beam_width, MAX_BEAM_WIDTH)]
        visited.update(state for state, _ in beam)
    if ordered:
        return _fallback_stages(sources, targets, ordered)
    raise ValueError("heuristic could not find a PST mirror route; use --segments or increase "
                     "--max-segment-steps/--beam-width (failure does not imply impossibility)")


def auto_pst_segment_stages(
    sources: Sequence[int], targets: Sequence[int], chain_length: int,
    max_steps: int = 10, beam_width: int = 128, ordered: bool = False,
) -> Stages:
    """功能 / Purpose: Plan paired or legacy occupation PST routes / 规划逐对或旧占据 PST 路线。

    输入 / Inputs: Sources, targets, N, heuristic budgets and ordered flag / 源、目标、链长、启发式预算及有序标志。
    输出 / Outputs: Valid stages, including exact fallback for ordered goals / 合法阶段，含有序目标的精确兜底。
    """
    return _auto_pst_plan(sources, targets, chain_length, max_steps, beam_width, ordered)[0]


def stage_scale(max_profile: float, g_base: float, g_max: float | None) -> tuple[float, float]:
    """功能 / Purpose: Choose a safe common coupling/time / 选择限幅后的共同耦合和时间。

    输入 / Inputs: Maximum unit profile M, g_base, optional g_max / 单位剖面最大值和耦合参数。
    输出 / Outputs: (g=min(g_base,g_max/M), tau=pi/(2g)); uncapped g=g_base / 有效尺度与时长。
    """
    # 同一阶段统一 g；限幅只减小、不增大耦合。 Common g synchronizes all parallel segments.
    g = g_base if g_max is None else min(g_base, g_max / max_profile)
    if g <= 0 or not math.isfinite(g * max_profile):
        raise ValueError("coupling scale is outside floating-point range")
    duration = (math.pi / 2.0) / g
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("stage duration is outside floating-point range")
    return g, duration


def build_pst_segment_stage_hamiltonian(
    basis: Sequence[tuple[int, ...]], basis_index: dict[tuple[int, ...], int],
    stage: Stage, g_base: float, g_max: float | None,
) -> tuple[csr_matrix, float, float, float]:
    """功能 / Purpose: Synchronize parallel PST segments / 构造同步并行 PST 阶段。

    输入 / Inputs: Basis/index, validated intervals, g_base, optional cap / 基底、合法区间与耦合参数。
    输出 / Outputs: (CSR Hamiltonian, duration, shared g, max edge J) / 哈密顿量、时长、共同尺度和峰值耦合。
    """
    # M=max(all profiles), g=min(g_base,g_max/M), tau=pi/(2g)：短段同步降速，不独立限幅。
    # One shared scale slows shorter segments too, so every mirror ends at the same tau.
    profiles = [engineered_pst_couplings(right - left + 1) for left, right in stage]
    max_profile = max(float(profile.max()) for profile in profiles)
    g, duration = stage_scale(max_profile, g_base, g_max)
    edges = [(left + j, left + j + 1, complex(g * profile[j]))
             for (left, _), profile in zip(stage, profiles, strict=True) for j in range(len(profile))]
    return build_edge_hamiltonian(basis, basis_index, edges), duration, g, g * max_profile


def simulate_schedule_time_grid(
    basis_dim: int, source_index: int, target_index: int,
    hamiltonians: Sequence[csr_matrix], durations: Sequence[float], num_steps: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """功能 / Purpose: Propagate coherent states interval by interval / 逐恒定哈密顿量区间相干演化。

    输入 / Inputs: Basis/source/target indices, matrices, durations, samples per interval / 基底索引、矩阵、时长及每区间采样数。
    输出 / Outputs: (Ns times, ideal target/source amplitudes, ideal probabilities) / 纳秒时间、理想振幅和概率数组。
    """
    state = np.zeros(basis_dim, dtype=np.complex128)
    state[source_index] = 1.0
    times = [0.0]
    target_amps = [complex(state[target_index])]
    source_amps = [complex(state[source_index])]
    elapsed = 0.0
    for hamiltonian, duration in zip(hamiltonians, durations, strict=True):
        end = elapsed + duration
        if not math.isfinite(end) or end <= elapsed:
            raise ValueError("total time is outside floating-point range")
        local_times = np.linspace(elapsed, end, num_steps)
        if np.any(np.diff(local_times) <= 0):
            raise ValueError("sampling interval is too small for floating-point time resolution")
        # 累积复振幅，不重置/测量中间态。 Carry coherent states across stages, without measurement.
        states = expm_multiply((-1j * duration) * hamiltonian, state,
                               start=0.0, stop=1.0, num=num_steps, traceA=0.0)
        # 丢弃每段重复的起点，总点数 1+I*(num_steps-1)。 Sampling, not integration steps.
        times.extend(local_times[1:])
        target_amps.extend(states[1:, target_index])
        source_amps.extend(states[1:, source_index])
        state = states[-1].copy()
        elapsed = end
    target_arr = np.asarray(target_amps, dtype=np.complex128)
    return (np.asarray(times), target_arr, np.asarray(source_amps, dtype=np.complex128),
            np.abs(target_arr) ** 2)


def distance_exposure(times: np.ndarray, primitives: Sequence[dict[str, object]]) -> np.ndarray:
    """功能 / Purpose: Accumulate traveled-distance budgets / 累积传输距离预算。

    输入 / Inputs: Nanosecond times and scheduled primitive metadata / 纳秒时间及基本操作调度元数据。
    输出 / Outputs: L(t), each primitive's distance growing linearly until shutoff / 各操作线性累积至完成的总距离。
    """
    exposure = np.zeros_like(times, dtype=float)
    for primitive in primitives:
        fraction = np.clip((times - primitive["start_time_ns"]) / primitive["duration_ns"], 0.0, 1.0)
        exposure += primitive["transport_distance"] * fraction
    return exposure


def distance_survival(config: SimulationConfig, exposure: np.ndarray) -> np.ndarray:
    """功能 / Purpose: Evaluate the phenomenological distance envelope / 计算唯象距离衰减包络。

    输入 / Inputs: Validated config and accumulated nonnegative distance / 合法配置及累计非负距离。
    输出 / Outputs: exp(-alpha*L), with no independent idle-time penalty / 无额外静止时间惩罚的包络数组。
    """
    return np.exp(-config.attenuation_per_edge * exposure)


def calibrated_target_fidelities(
    config: SimulationConfig, exposure: np.ndarray, ideal_fidelities: np.ndarray,
) -> np.ndarray:
    """功能 / Purpose: Apply the distance-error design envelope / 应用距离误差设计包络。

    输入 / Inputs: Validated config, distance exposure and ideal probabilities / 合法配置、距离暴露及理想概率。
    输出 / Outputs: exp(-alpha*L(t))*F_ideal(t); not a Lindblad solution / 校准概率数组；并非 Lindblad 解。
    """
    return distance_survival(config, exposure) * ideal_fidelities


def plan_transfer(config: SimulationConfig) -> TransferPlan:
    """功能 / Purpose: Share an immutable matrix-free transfer schedule / 共享不可变无矩阵传输方案。

    输入 / Inputs: SimulationConfig; ordered=False preserves occupation behavior / 模拟配置，ordered=False 保留占据行为。
    输出 / Outputs: Content-cached frozen plan without basis/matrix/profile arrays / 按内容缓存的冻结方案，不生成基底、矩阵或剖面数组。
    Sampling density is irrelevant to planning and canonicalized to two / 采样密度不影响规划，缓存统一为二。
    """
    validate_config(config)
    # 即使调用者传入列表也复制冻结；不把可变对象放入缓存。 Freeze user-owned sequences before caching.
    canonical = replace(
        config, chain_length=int(config.chain_length),
        pairs=tuple((int(source), int(target)) for source, target in config.pairs),
        segments=None if config.segments is None else tuple(
            tuple((int(left), int(right)) for left, right in stage) for stage in config.segments),
        num_steps=2, max_segment_steps=int(config.max_segment_steps), beam_width=int(config.beam_width),
        reference_time_ns=float(config.reference_time_ns), reference_distance=float(config.reference_distance),
        reference_fidelity=float(config.reference_fidelity),
        g_max=None if config.g_max is None else float(config.g_max), ideal=bool(config.ideal),
    )
    return _cached_transfer_plan(canonical)


@lru_cache(maxsize=PLAN_CACHE_MAXSIZE)
def _cached_transfer_plan(config: SimulationConfig) -> TransferPlan:
    """功能 / Purpose: Build/cache calibrated schedules using positions only / 仅用位置构造并缓存校准调度。

    输入 / Inputs: Canonical immutable validated config / 已校验的规范化不可变配置。
    输出 / Outputs: Immutable primitives, independent shutoffs and resident displacement / 不可变操作、独立关断及居民位移。
    """
    disjoint = are_disjoint_edges(config.active_pairs)
    stages: Stages = ()
    planner = "direct" if config.active_pairs else "identity"
    if not disjoint:
        if config.segments is not None:
            stages, planner = config.segments, "manual"
        else:
            stages, planner = _auto_pst_plan(
                config.sources, config.targets, config.chain_length,
                config.max_segment_steps, config.beam_width, config.ordered)
        if stages:
            validate_segment_stages(config.chain_length, stages)

    stage_durations, stage_requested_durations, interval_durations = [], [], []
    interval_indices = []
    primitives: list[TransferPrimitive] = []
    labeled_path = [config.sources]
    logical_stages = (config.active_pairs,) if disjoint and config.active_pairs else stages
    elapsed = 0.0
    for stage_index, stage in enumerate(logical_stages):
        first_primitive = len(primitives)
        for left, right in stage:
            span = abs(right - left)
            requested = config.reference_time_ns * math.sqrt(span / config.reference_distance)
            if not math.isfinite(requested) or requested <= 0:
                raise ValueError("requested duration is outside floating-point range")
            g_requested = (math.pi / 2) / requested
            length = span + 1
            # max sqrt(j*(L-j)) has a closed form; no O(L) coupling profile for estimation.
            # 最大耦合剖面用闭式，估计不创建 O(L) 数组。
            max_profile = 1.0 if disjoint else math.sqrt((length // 2) * ((length + 1) // 2))
            g, duration = stage_scale(max_profile, g_requested, config.g_max)
            cost = span if disjoint else sum(
                abs(left + right - 2 * site) for site in labeled_path[-1] if left <= site <= right)
            end_time = elapsed + duration
            if not math.isfinite(end_time) or end_time <= elapsed:
                raise ValueError("total time is outside floating-point range")
            primitives.append(TransferPrimitive(
                stage_index=stage_index, kind="direct-xy" if disjoint else "pst-segment",
                sites=(left, right), span_distance=span, transport_distance=cost,
                requested_duration_ns=requested, duration_ns=duration,
                start_time_ns=elapsed, end_time_ns=end_time,
                effective_g_rad_per_ns=g, max_coupling_rad_per_ns=g * max_profile,
                cap_limited=g < g_requested,
            ))
        scheduled = primitives[first_primitive:]
        previous = 0.0
        for end in sorted({primitive.duration_ns for primitive in scheduled}):
            interval_indices.append(tuple(index for index in range(first_primitive, len(primitives))
                                          if primitives[index].duration_ns > previous))
            interval_durations.append(end - previous)
            previous = end
        stage_durations.append(previous)
        stage_requested_durations.append(max(primitive.requested_duration_ns for primitive in scheduled))
        elapsed += previous
        labeled_path.append(config.targets if disjoint else
                            apply_segment_stage_labeled(labeled_path[-1], stage))
    if not mapping_matches(labeled_path[-1], config.targets, config.ordered):
        goal_type = "paired targets" if config.ordered else "target occupation"
        raise ValueError(f"PST segments do not map source positions to {goal_type}: "
                         f"{labeled_path[-1]} != {config.targets}")
    return TransferPlan(
        config=config, route="direct-xy" if disjoint else "pst-segments", planner=planner,
        direct_pairs=config.active_pairs if disjoint else (), segment_stages=stages,
        primitives=tuple(primitives), stage_durations=tuple(stage_durations),
        stage_requested_durations=tuple(stage_requested_durations),
        interval_durations=tuple(interval_durations), interval_primitive_indices=tuple(interval_indices),
        labeled_path=tuple(labeled_path), transfer_time_ns=float(elapsed),
        effective_distance_edges=sum(primitive.transport_distance for primitive in primitives),
    )


def _request_config(
    sources: Sequence[int], targets: Sequence[int], chain_length: int | None, **controls: object,
) -> SimulationConfig:
    """功能 / Purpose: Validate public lists before pairing / 在配对前校验公共输入列表。

    输入 / Inputs: Source/target sequences, optional N and configuration overrides / 源目标序列、可选链长及配置覆盖。
    输出 / Outputs: Config owning immutable integer pairs; malformed lists raise ValueError / 拥有不可变整数对的配置，非法列表时报错。
    """
    sources, targets = tuple(sources), tuple(targets)
    if not sources or len(sources) != len(targets):
        raise ValueError("sources and targets must be nonempty and have equal lengths")
    if any(isinstance(site, bool) or not isinstance(site, (int, np.integer)) for site in sources + targets):
        raise ValueError("sources and targets must contain integer site indices")
    if chain_length is None:
        chain_length = int(max(sources + targets)) + 1
    return SimulationConfig(chain_length=chain_length, pairs=tuple(zip(sources, targets, strict=True)),
                            num_steps=2, **controls)


def estimate_transfer(
    sources: Sequence[int], targets: Sequence[int], chain_length: int | None = None, *,
    reference_time_ns: float = REFERENCE_TIME_NS, reference_distance: float = float(REFERENCE_DISTANCE),
    reference_fidelity: float = REFERENCE_FIDELITY, g_max: float | None = DEFAULT_G_MAX_RAD_PER_NS,
    max_segment_steps: int = 10, beam_width: int = 128, ordered: bool = True,
) -> dict[str, object]:
    """功能 / Purpose: Estimate paired transfer cost without Fock-space evolution / 不演化 Fock 空间地估计逐对传输成本。

    输入 / Inputs: All modeled residents including i:i holds, N, calibration/budgets / 全部建模居民含静止 i:i、链长、校准与预算。
    输出 / Outputs: Fresh dict with ns time, edge distance, stable log_fidelity and immutable plan / 含纳秒时间、边距离、稳定对数保真度及不可变方案的新字典。
    Ordered goals constrain positions, not unknown-state/process fidelity / 有序目标约束位置，不验证未知量子态或过程保真度。
    """
    config = _request_config(
        sources, targets, chain_length, reference_time_ns=reference_time_ns,
        reference_distance=reference_distance, reference_fidelity=reference_fidelity,
        g_max=g_max, max_segment_steps=max_segment_steps, beam_width=beam_width, ordered=ordered,
    )
    plan = plan_transfer(config)
    # 直接累计对数误差，不先计算会下溢的概率。 Never take log of an underflowed probability.
    log_fidelity = (-plan.config.attenuation_per_edge * plan.effective_distance_edges
                    if plan.effective_distance_edges else 0.0)
    return {
        "transfer_time_ns": plan.transfer_time_ns,
        "effective_distance_edges": plan.effective_distance_edges,
        "log_fidelity": log_fidelity, "neglog_fidelity": -log_fidelity,
        "model_version": MODEL_VERSION, "plan": plan,
    }


def run_simulation(config: SimulationConfig) -> tuple[SimulationResult, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """功能 / Purpose: Evolve the same immutable plan used by estimation / 演化与估计共用的不可变方案。

    输入 / Inputs: SimulationConfig, occupation-only unless ordered=True / 模拟配置，默认占据目标，可指定逐对目标。
    输出 / Outputs: Result, ns times, ideal amplitudes/probabilities / 摘要、纳秒时间、理想振幅及概率。
    Numerical probability is occupation-only even for ordered plans / 即使方案有序，数值概率仍仅验证占据。
    """
    plan = plan_transfer(config)
    basis, index = build_fixed_excitation_basis(config.chain_length, len(config.pairs))
    matrices = []
    for primitive in plan.primitives:
        left, right = primitive.sites
        g = primitive.effective_g_rad_per_ns
        if primitive.kind == "direct-xy":
            edges = ((left, right, complex(g)),)
        else:
            profile = engineered_pst_couplings(right - left + 1, g)
            edges = tuple((left + j, left + j + 1, complex(coupling))
                          for j, coupling in enumerate(profile))
        matrices.append(build_edge_hamiltonian(basis, index, edges))
    hamiltonians = [sum((matrices[index] for index in active),
                        csr_matrix((len(basis), len(basis)), dtype=complex))
                    for active in plan.interval_primitive_indices]
    times, target_amps, source_amps, fidelities = simulate_schedule_time_grid(
        len(basis), index[tuple(sorted(config.sources))], index[tuple(sorted(config.targets))],
        hamiltonians, plan.interval_durations, config.num_steps)
    details = plan.to_route_details()
    exposure = distance_exposure(times, details["primitives"])
    calibrated = calibrated_target_fidelities(config, exposure, fidelities)
    peak, calibrated_peak = int(np.argmax(fidelities)), int(np.argmax(calibrated))
    result = SimulationResult(
        config=config, route=plan.route, hilbert_dim=len(basis),
        sample_count=len(times), peak_time=float(times[peak]), peak_fidelity=float(fidelities[peak]),
        final_time=float(times[-1]), final_fidelity=float(fidelities[-1]),
        calibrated_peak_time=float(times[calibrated_peak]),
        calibrated_peak_fidelity=float(calibrated[calibrated_peak]),
        calibrated_final_fidelity=float(calibrated[-1]), route_details=details,
    )
    return result, times, target_amps, source_amps, fidelities


def simulate_transfer(
    sources: Sequence[int], targets: Sequence[int], chain_length: int | None = None, *,
    reference_time_ns: float = REFERENCE_TIME_NS, reference_distance: float = REFERENCE_DISTANCE,
    reference_fidelity: float = REFERENCE_FIDELITY,
    g_max: float | None = DEFAULT_G_MAX_RAD_PER_NS, ideal: bool = False,
) -> dict[str, float]:
    """功能 / Purpose: Public source/target-to-time/fidelity interface / 公共源目标到时间保真度接口。

    输入 / Inputs: Equal nonempty integer site lists, optional N and calibration controls / 等长非空整数位置列表、可选链长及校准参数。
    输出 / Outputs: {'transfer_time_ns': completion time, 'fidelity': calibrated completion probability} / 完成时间与校准最终占据概率。
    Pairs follow input order; no rematching or process-fidelity claim / 按输入顺序配对，不重匹配且不保证过程保真度。
    """
    config = _request_config(
        sources, targets, chain_length, reference_time_ns=reference_time_ns,
        reference_distance=reference_distance, reference_fidelity=reference_fidelity,
        g_max=g_max, ideal=ideal,
    )
    result = run_simulation(config)[0]
    return {"transfer_time_ns": result.final_time, "fidelity": result.calibrated_final_fidelity}


