"""Stable additive error accounting / 稳定的加性误差统计。"""

import math


def probability_cost(probability: float, count: float = 1.0) -> float:
    """功能 / Purpose: Convert repeated success probability to additive cost / 转换为加性误差。

    输入 / Inputs: Probability in [0,1], nonnegative finite count / 合法概率与非负次数或距离。
    输出 / Outputs: -count*log(probability); zero for no operations / 负对数成本；零次操作为零。
    """
    if not math.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError("Fidelity must be finite and in [0, 1].")
    if not math.isfinite(count) or count < 0:
        raise ValueError("Operation count or effective distance must be finite and nonnegative.")
    # 避免 0 * inf 产生 NaN；没有操作不产生误差。 No operations means no error.
    if count == 0 or probability == 1:
        return 0.0
    if probability == 0:
        return math.inf
    return -math.log(probability) * count


def aggregate_batch_uncertainty(batches: list[dict]) -> dict:
    """功能 / Purpose: Sum log-confidence endpoints without independence assumptions / 不假设独立性地累加对数置信端点。

    Inputs / 输入: Noisy batch estimates with cache identity and core uncertainty / 含缓存标识与核心不确定度的批次。
    Outputs / 输出: Conservative cost interval, union-bound confidence and sampling counts / 保守成本区间、联合界置信水平与采样计数。
    Repeated cached estimates contribute their full interval each time, never sqrt(count).
    重复缓存估计每次贡献完整区间，不按次数平方根缩小误差；只保证核心单批次区间所声明的统计含义。
    """
    lower_costs, upper_costs = [], []
    unique = {}
    mc_batches = unconverged = exhausted = 0
    solvers = {}
    for batch in batches:
        noise = batch["noise"]
        uncertainty = noise["uncertainty"]
        log_lower = float(uncertainty["log_ci_lower"])
        log_upper = float(uncertainty["log_ci_upper"])
        if math.isnan(log_lower) or math.isnan(log_upper) or not log_lower <= log_upper <= 0:
            raise ValueError("Hamiltonian batch must provide ordered nonpositive log confidence bounds")
        lower_costs.append(-log_upper)
        upper_costs.append(-log_lower)
        sampled = uncertainty["trajectories"] > 0
        mc_batches += int(sampled)
        unconverged += int(not uncertainty["precision_met"])
        exhausted += int(uncertainty["budget_exhausted"])
        solver = noise["solver"]
        solvers[solver] = solvers.get(solver, 0) + 1
        unique.setdefault(batch["noise_estimate_key"], uncertainty)
    sampled_unique = [item for item in unique.values() if item["trajectories"] > 0]
    confidence = max(0.0, 1.0 - math.fsum(1.0 - item["confidence"] for item in sampled_unique))
    return {
        "method": "summed_log_endpoints_union_bound_no_independence",
        "neglog_fidelity_interval": [math.fsum(lower_costs), math.fsum(upper_costs)],
        "simultaneous_confidence_lower_bound": confidence,
        "batch_count": len(batches), "unique_estimate_count": len(unique),
        "reused_estimate_count": len(batches) - len(unique),
        "mc_batch_count": mc_batches, "unique_mc_estimate_count": len(sampled_unique),
        "unique_mc_trajectories": sum(item["trajectories"] for item in sampled_unique),
        "unconverged_batch_count": unconverged, "budget_exhausted_batch_count": exhausted,
        "precision_met": unconverged == 0,
        "solver_batch_counts": solvers,
        "scope": "batch_occupation_probability_only_not_full_circuit_decoherence",
        "interval_scope": "sampling_only_not_a_rigorous_numerical_error_bound",
    }


def normalized_gate_fidelity(
    intra_count: float,
    inter_count: float,
    qst_cost: float,
    logic_gate_count: int,
    intra_fidelity: float,
    inter_fidelity: float,
) -> float:
    """功能 / Purpose: Exponentiate only after logical-gate normalization / 归一化后计算逻辑门概率。

    输入 / Inputs: Physical gate counts, additive QST cost, logical count, gate fidelities / 门数、QST成本和标定。
    输出 / Outputs: Mean logical-gate fidelity; raises for an empty program / 平均逻辑门保真度；空程序报错。
    """
    if logic_gate_count <= 0:
        raise ValueError("Cannot normalize fidelity for a circuit without logic gates.")
    if math.isnan(qst_cost) or qst_cost < 0:
        raise ValueError("QST negative-log fidelity must be nonnegative.")
    # 不从全电路概率开根号：该概率可能早已下溢。 Never take a root of an underflowed product.
    average_cost = math.fsum((
        probability_cost(intra_fidelity, intra_count / logic_gate_count),
        probability_cost(inter_fidelity, inter_count / logic_gate_count),
        qst_cost / logic_gate_count,
    ))
    return math.exp(-average_cost)
