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
