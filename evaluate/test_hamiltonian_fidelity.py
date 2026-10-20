"""Stable router fidelity tests / 路由保真度稳定性测试（不写文件）。"""

import math
from pathlib import Path
import pickle
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import networkx as nx

from qunoc.element.benchmark import Control, Gate, Program
from qunoc.element.chiplet import FabricateModel, OperationConfig
from qunoc.element.package import Package
from qunoc.routing.circuit import Transport
from qunoc.routing.fidelity import normalized_gate_fidelity, probability_cost
from qunoc.routing.router import Router


def make_router(actions, *, model="hamiltonian", effective_distance=34, logic_gates=2):
    """功能 / Purpose: Build isolated accounting fixture / 构造隔离的统计测试对象。

    输入 / Inputs: Actions, mode, modeled distance and logical gate count / 动作、模型、距离与门数。
    输出 / Outputs: Router with minimal package/circuit stubs / 带最小封装和电路桩的 Router。
    """
    router = Router.__new__(Router)
    op_config = OperationConfig(eqst_model=model)
    batches = [{"effective_distance_edges": effective_distance, "transfer_time_ns": 520.0}]
    circuit = SimpleNamespace(stages=[SimpleNamespace(circuit_actions=actions)], op_config=op_config,
                              estimate_eqst_batches=Mock(return_value=batches),
                              estimate_duration=Mock(return_value=520.0))
    fab = SimpleNamespace(enhanced_fide=0.992 ** 0.2, inter_fide=0.983,
                          max_length=80, get_intra_fide=lambda _: 0.99)
    router.circuit = circuit
    router.package = SimpleNamespace(fab_model=fab, get_coupler_fide=lambda *_: 0.99)
    router.package_graph = nx.Graph()
    graph = nx.DiGraph()
    graph.add_nodes_from(range(-1, logic_gates))
    router.program = SimpleNamespace(get_gate_depend_graph=lambda: graph)
    return router


class StableFidelityTests(unittest.TestCase):
    def test_probability_boundaries(self):
        """功能 / Purpose: Check safe 0/1 probabilities / 检查概率边界。

        输入 / Inputs: Boundary probabilities and counts / 边界概率及次数。
        输出 / Outputs: None; asserts zero, infinity and validation / 无；断言边界及校验结果。
        """
        self.assertEqual(probability_cost(0, 0), 0)
        self.assertEqual(probability_cost(1, 1e8), 0)
        self.assertEqual(probability_cost(0, 1), math.inf)
        for p, count in ((-0.1, 1), (1.1, 1), (float("nan"), 1), (0.9, -1)):
            with self.subTest(p=p, count=count), self.assertRaises(ValueError):
                probability_cost(p, count)

    def test_model_distance_counted_once_and_raw_distance_preserved(self):
        """功能 / Purpose: Separate geometric and modeled distances / 区分原始与模型距离。

        输入 / Inputs: Three conflicting transfers, raw24/model34 / 三条冲突请求。
        输出 / Outputs: None; asserts one batch charge and five-tuple compatibility / 无；断言单次批次计费。
        """
        actions = [Transport(i, (s, 0), (t, 0)) for i, (s, t) in enumerate(((0, 9), (9, 10), (5, 19)))]
        router = make_router(actions)
        details = router.statistic_route_details()
        self.assertEqual(details["qst_dist"], 24)
        self.assertEqual(details["qst_effective_dist"], 34)
        router.circuit.estimate_eqst_batches.assert_called_once_with(router.package)
        expected_cost = probability_cost(router.package.fab_model.enhanced_fide, 34)
        self.assertAlmostEqual(details["qst_neglog_fidelity"], expected_cost)
        self.assertAlmostEqual(router.estimate_qunoc_logic_gate_fide(details), math.exp(-expected_cost / 2))
        self.assertEqual(router.statistic_route_results()[:4], (0, 0, 24, 0))
        router.estimate_qunoc_circuit_duration(details["eqst_batches"])
        router.circuit.estimate_duration.assert_called_with(package=router.package, eqst_batches=details["eqst_batches"])

    def test_legacy_uses_raw_distance_and_does_not_call_batches(self):
        """功能 / Purpose: Preserve legacy EQST accounting / 保留旧统计口径。

        输入 / Inputs: Legacy fixture / 旧模型测试对象。
        输出 / Outputs: None; asserts raw-distance probability / 无；断言原距离概率。
        """
        router = make_router([Transport(0, (0, 0), (9, 0))], model="legacy")
        details = router.statistic_route_details()
        self.assertEqual(details["qst_effective_dist"], 9)
        self.assertAlmostEqual(details["total_fide"], router.package.fab_model.enhanced_fide ** 9)
        router.circuit.estimate_eqst_batches.assert_not_called()
        del router.circuit.op_config.eqst_model
        self.assertEqual(router.statistic_route_details()["eqst_model"], "legacy")

    def test_underflow_does_not_destroy_logical_fidelity(self):
        """功能 / Purpose: Normalize before exponential underflow / 下溢前归一化。

        输入 / Inputs: Huge modeled distance and many logical gates / 大距离与逻辑门数。
        输出 / Outputs: None; asserts total0 but finite mean probability / 无；总概率零但均值仍有效。
        """
        router = make_router([], effective_distance=1_000_000, logic_gates=10000)
        details = router.statistic_route_details()
        self.assertEqual(details["total_fide"], 0)
        self.assertTrue(math.isfinite(details["total_neglog_fidelity"]))
        result = router.estimate_qunoc_logic_gate_fide(details)
        self.assertGreater(result, 0.8)
        self.assertLess(result, 1)
        # 逻辑门路径只允许最终一次exp，不生成全电路概率。 Only one final exponential on this path.
        real_exp = math.exp
        with patch("qunoc.routing.fidelity.math.exp", wraps=real_exp) as exp:
            self.assertAlmostEqual(router.estimate_qunoc_logic_gate_fide(), result)
            self.assertEqual(exp.call_count, 1)

    def test_custom_and_perfect_fabrication_calibration(self):
        """功能 / Purpose: Respect package calibration / 保持封装标定有效。

        输入 / Inputs: Custom/perfect/zero probabilities / 自定义与边界概率。
        输出 / Outputs: None; asserts stable calibrated means / 无；断言稳定均值。
        """
        router = make_router([], effective_distance=34)
        router.package.fab_model.enhanced_fide = 0.98
        self.assertAlmostEqual(router.estimate_qunoc_logic_gate_fide(), 0.98 ** 17)
        router.package.fab_model.enhanced_fide = 1
        self.assertEqual(router.estimate_qunoc_logic_gate_fide(), 1)
        router.package.fab_model.enhanced_fide = 0
        self.assertEqual(router.estimate_qunoc_logic_gate_fide(), 0)
        self.assertEqual(normalized_gate_fidelity(0, 0, 0, 1, 0, 0), 1)

    def test_logical_denominator_and_empty_program(self):
        """功能 / Purpose: Preserve logical-gate count normalization / 保持逻辑门分母。

        输入 / Inputs: Known gate counts and empty-program fixture / 已知门数及空程序。
        输出 / Outputs: None; asserts formula and explicit error / 无；断言公式与报错。
        """
        cost = probability_cost(0.99, 30)
        expected = 0.98 ** 2 * 0.97 ** 3 * 0.99 ** 5
        self.assertAlmostEqual(normalized_gate_fidelity(12, 18, cost, 6, 0.98, 0.97), expected)
        router = make_router([], logic_gates=0)
        with self.assertRaisesRegex(ValueError, "without logic gates"):
            router.estimate_qunoc_logic_gate_fide()


class RealRoutingSmokeTests(unittest.TestCase):
    def test_both_schedulers_on_small_real_package(self):
        """功能 / Purpose: Exercise actual compile-to-statistics pipeline / 验证真实编译到统计路径。

        输入 / Inputs: 12x12 package and five logical gates, both schedulers / 小型封装、五个门、两种调度。
        输出 / Outputs: None; assert replay, batching and stable metrics / 无；验证重放、批次及稳定指标。
        """
        for strategy in ("immediate", "concurrent"):
            with self.subTest(strategy=strategy):
                package = Package(
                    package_size=(12, 12),
                    fab_model=FabricateModel(max_capacity=16, max_length=12),
                    op_config=OperationConfig(eqst_model="hamiltonian"),
                    design_strategy="template", record_design_history=False,
                )
                Gate._counter = 0
                program = Program("smoke", 8, [Control(a, b) for a, b in
                    ((0, 7), (1, 6), (2, 5), (0, 6), (1, 7))])
                router = Router(package, program, route_strategy=strategy)
                with patch.object(Router, "draw_qubit_mapping_block", return_value=None), \
                     patch.object(Router, "draw_qubit_mapping_action", return_value=None):
                    router.route_program_on_package()
                details = router.statistic_route_details()
                self.assertGreater(details["qst_dist"], 0)
                self.assertGreater(details["eqst_batch_count"], 0)
                self.assertTrue(math.isfinite(router.estimate_qunoc_circuit_duration(details["eqst_batches"])))
                probability = router.estimate_qunoc_logic_gate_fide(details)
                self.assertGreater(probability, 0)
                self.assertLessEqual(probability, 1)
                expected = probability_cost(package.fab_model.enhanced_fide, details["qst_effective_dist"])
                self.assertAlmostEqual(details["qst_neglog_fidelity"], expected)

    def test_pickle_restatistics_does_not_compile_or_mutate_circuit(self):
        """功能 / Purpose: Restat a saved Circuit without rerouting / 无需重新路由地统计保存电路。

        输入 / Inputs: Pickle roundtrip of a legacy-compiled small circuit / 旧模型编译电路的内存序列化往返。
        输出 / Outputs: None; verify new metrics and unchanged source artifact / 无；验证新指标和原对象不变。
        """
        package = Package(package_size=(12, 12),
                          fab_model=FabricateModel(max_capacity=16, max_length=12),
                          op_config=OperationConfig(), design_strategy="template", record_design_history=False)
        Gate._counter = 0
        program = Program("smoke", 8, [Control(0, 7), Control(1, 6)])
        router = Router(package, program)
        with patch.object(Router, "draw_qubit_mapping_block", return_value=None), \
             patch.object(Router, "draw_qubit_mapping_action", return_value=None):
            circuit = router.route_program_on_package()
        saved = pickle.loads(pickle.dumps(circuit))
        before = pickle.dumps(saved)
        package.op_config = OperationConfig(eqst_model="hamiltonian")
        with patch.object(Router, "route_program_on_package", side_effect=AssertionError("must not compile")), \
             patch.object(Program, "__init__", side_effect=AssertionError("must not rebuild program")):
            result = Router.reestimate_compiled_circuit(package, saved, 2)
        self.assertEqual(pickle.dumps(saved), before)
        self.assertEqual(saved.op_config.eqst_model, "legacy")
        self.assertEqual(result["eqst_model"], "hamiltonian")
        self.assertGreater(result["circuit_duration"], 0)
        self.assertTrue(0 < result["gate_fidelity"] <= 1)
        for count in (0, -1, True, 1.5):
            with self.subTest(count=count), self.assertRaises(ValueError):
                Router.reestimate_compiled_circuit(package, saved, count)


if __name__ == "__main__":
    unittest.main()
