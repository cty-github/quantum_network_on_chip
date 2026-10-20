"""Synthetic evaluation integration tests / 不运行长实验的评估集成测试。

Run with: python -m unittest evaluate.test_hamiltonian_evaluation
"""

import argparse
import copy
import csv
import json
import math
import pickle
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import networkx as nx

from evaluate import run_ablation as ablation
from evaluate import test_routing_scalability as scalability
from qunoc.element.chiplet import FabricateModel, OperationConfig, TransferChiplet
from qunoc.routing.circuit import Board, Circuit, SwapGate, Transport
from qunoc.routing.router import Router


def _package(op_config=None):
    """Return a small package substitute with real operation/fabrication settings.

    输入可选操作配置；输出小型 package 替身，使用真实标定类，避免布局实验。
    """
    return SimpleNamespace(
        op_config=op_config if op_config is not None else OperationConfig(),
        fab_model=FabricateModel(),
        package_size=(40, 40),
        package_graph=nx.path_graph([(index, 0) for index in range(11)]),
        get_coupler_fide=lambda *_: 0.98,
    )


def _program():
    """Return a ten-gate program substitute; no input or benchmark generation.

    无输入；输出十逻辑门程序替身，避免生成大型 benchmark。
    """
    graph = nx.DiGraph()
    graph.add_nodes_from(range(-1, 10))
    return SimpleNamespace(get_gate_depend_graph=lambda: graph)


def _staged_router(package, program):
    """Build real Router/Circuit metrics for a detour with a ten-edge net transfer.

    输入 package 和程序；输出含往返绕行的十边净传输统计对象，不调用完整路由搜索。
    """
    die = TransferChiplet(intra_fide=0.99, axis=0, pos=0, start=0, end=10)
    package.get_node_chiplet = lambda _node: die
    initial = {0: (0, 0)}
    physical = {(x, 0): (0 if x == 0 else -1) for x in range(11)}
    circuit = Circuit(1, (11, 1), initial, physical, None, package.op_config)
    circuit.extend_stage(True)
    stage = circuit.extend_stage(False)
    stage.circuit_actions.extend([
        Transport(0, (0, 0), (5, 0)),
        Transport(0, (5, 0), (0, 0)),
        Transport(0, (0, 0), (10, 0)),
    ])
    circuit.curr_l2p_map[0] = (10, 0)
    circuit.curr_p2l_map[(0, 0)] = -1
    circuit.curr_p2l_map[(10, 0)] = 0
    router = Router.__new__(Router)
    router.package = package
    router.package_graph = nx.Graph()
    router.program = program
    router.circuit = circuit
    router.route_program_on_package = Mock(return_value=circuit)
    return router


class EQSTConfigurationTests(unittest.TestCase):
    def test_all_routing_sweeps_inherit_default_programs(self):
        """功能 / Purpose: Keep every sweep on the same four programs / 保证各实验继承同一程序列表。

        输入 / Inputs: Built-in defaults and checked-in YAML / 内置默认配置和项目 YAML。
        输出 / Outputs: None; assert all five effective lists / 无；断言五类实验的实际列表。
        """
        expected = ["vqe", "qaoa", "bv", "s_ghz"]
        for path in (None, Path(__file__).with_name("test_routing_scalability.yaml")):
            config = scalability.load_experiment_config(path)
            self.assertEqual(config.defaults["quantum_programs"], expected)
            for name, spec in config.experiments.items():
                with self.subTest(path=path, experiment=name):
                    effective = scalability._deep_update(config.defaults, spec.defaults)
                    programs = spec.scalability.get("quantum_program_to_test", effective["quantum_programs"])
                    self.assertEqual(programs, expected)

    def test_program_override_is_local_to_experiment(self):
        """功能 / Purpose: Preserve explicit per-experiment selection / 保留实验局部覆盖。

        输入 / Inputs: YAML overriding only one sweep / 仅覆盖一个实验的 YAML。
        输出 / Outputs: None; assert override precedence and sibling isolation / 无；断言优先级和兄弟实验隔离。
        """
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "programs.yaml"
            path.write_text(
                "defaults:\n  quantum_programs: [vqe, qaoa, bv, s_ghz]\n"
                "experiments:\n  package_size:\n    defaults:\n      quantum_programs: [bv]\n"
                "    scalability:\n      quantum_program_to_test: [s_ghz]\n",
                encoding="utf-8",
            )
            config = scalability.load_experiment_config(path)
        for name, spec in config.experiments.items():
            effective = scalability._deep_update(config.defaults, spec.defaults)
            programs = spec.scalability.get("quantum_program_to_test", effective["quantum_programs"])
            self.assertEqual(programs, ["s_ghz"] if name == "package_size" else ["vqe", "qaoa", "bv", "s_ghz"])

    def test_legacy_default_and_old_pickle_fallback(self):
        """Check absent EQST fields deserialize as legacy / 缺失字段保持 legacy。"""
        old = OperationConfig()
        for name in list(vars(old)):
            if name.startswith("eqst_"):
                delattr(old, name)
        restored = pickle.loads(pickle.dumps(old))
        package = _package(restored)
        settings = scalability._copy_package_eqst_config(package)
        self.assertEqual(settings["model"], "legacy")
        self.assertEqual(settings["reference_time_ns"], 125)
        self.assertEqual(settings["reference_distance"], 5)
        self.assertEqual(settings["max_segment_steps"], 10)
        self.assertEqual(settings["beam_width"], 128)
        self.assertAlmostEqual(settings["g_max"], 3 * math.pi / 250)
        self.assertEqual(settings, json.loads(json.dumps(settings, allow_nan=False)))

    def test_overrides_copy_shared_operation_config(self):
        """Check overrides leave other packages intact / 覆盖参数不修改共享对象。"""
        shared = OperationConfig(eqst_reference_time_ns=150)
        first, second = _package(shared), _package(shared)
        settings = scalability._copy_package_eqst_config(
            first, {"model": "hamiltonian", "beam_width": 12}
        )
        self.assertIsNot(first.op_config, second.op_config)
        self.assertEqual(second.op_config.get_eqst_config()["model"], "legacy")
        self.assertEqual(second.op_config.eqst_beam_width, 128)
        self.assertEqual(settings["model"], "hamiltonian")
        self.assertEqual(settings["reference_time_ns"], 150)
        self.assertEqual(settings["beam_width"], 12)
        scalability._copy_package_eqst_config(first)
        self.assertEqual(first.op_config.eqst_model, "legacy")

    def test_invalid_eqst_mapping_is_rejected(self):
        """Check malformed model/calibration settings fail / 非法配置必须报错。"""
        for options in (
            [], {"unexpected": 1}, {"model": "unknown"},
            {"reference_time_ns": 0}, {"reference_distance": -1},
            {"g_max": float("nan")}, {"max_segment_steps": 1.5},
            {"beam_width": 0},
        ):
            with self.subTest(options=options), self.assertRaises(ValueError):
                scalability._copy_package_eqst_config(_package(), options)

    def test_yaml_inheritance_and_cli_precedence(self):
        """Check nested defaults and explicit CLI overrides / 检查嵌套配置优先级。"""
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "config.yaml"
            path.write_text(
                "defaults:\n  eqst:\n    model: hamiltonian\n    beam_width: 32\n"
                "experiments:\n  package_size:\n    defaults:\n      eqst:\n"
                "        reference_time_ns: 140\n        model: legacy\n",
                encoding="utf-8",
            )
            config = scalability.load_experiment_config(path)
        parser = argparse.ArgumentParser()
        scalability._add_eqst_arguments(parser)
        self.assertEqual(scalability._eqst_cli_overrides(parser.parse_args([])), {})
        args = parser.parse_args(["--eqst-model", "hamiltonian", "--eqst-beam-width", "16"])
        spec = config.experiments["package_size"]
        spec.defaults = scalability._deep_update(
            spec.defaults, {"eqst": scalability._eqst_cli_overrides(args)}
        )
        effective = scalability._deep_update(config.defaults, spec.defaults)
        self.assertEqual(effective["eqst"], {
            "model": "hamiltonian", "beam_width": 16, "reference_time_ns": 140,
        })
        self.assertEqual(config.defaults["eqst"]["beam_width"], 32)


class EvaluationCacheTests(unittest.TestCase):
    def setUp(self):
        """Create temporary package/results and inputs / 创建临时缓存输入，无真实实验。"""
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.common = {"layout_root_dir": str(self.root / "layouts")}
        self.spec = scalability.ExperimentSpec(
            "package_size", str(self.root / "results"), "log", "output.csv",
            scalability={"package_size_to_test": [40], "quantum_program_to_test": ["bv"]},
        )
        self.run_config = {
            "package_size": 40, "layout_design_strategy": "greedy", "reuse_result": True,
            "eqst": {"model": "hamiltonian"},
        }
        self.path = Path(scalability._build_layout_package_path(
            self.common, self.spec.name, self.run_config
        ))
        self.path.parent.mkdir(parents=True)
        self.path.write_bytes(b"synthetic package identity")
        self.package = _package()
        self.cache = Path(self.spec.exp_dir) / "40_greedy_bv" / "exp_result.json"

    def _fake_router(self, package, _program, **_kwargs):
        """Return a metric-only mock for cache tests / 输入 package，返回轻量统计替身。"""
        settings = package.op_config.get_eqst_config()
        details = {
            "intra_gate": 2, "inter_gate": 3, "qst_dist": 10, "meas_num": 1,
            "total_fide": 0.8, "qst_effective_dist": 7.5,
            "qst_neglog_fidelity": 0.1, "total_neglog_fidelity": 0.2,
            "eqst_model": settings["model"], "eqst_config": settings,
            "eqst_batch_count": 1, "eqst_batches": [{"effective_distance_edges": 7.5}],
        }
        router = Mock()
        router.route_program_on_package.return_value.to_pickle.side_effect = (
            lambda path: Path(path).write_bytes(b"compiled synthetic circuit")
        )
        router.statistic_route_details.return_value = details
        router.estimate_qunoc_circuit_duration.return_value = 200.0
        router.estimate_qunoc_logic_gate_fide.return_value = 0.99
        return router

    def _route(self, config=None):
        """Run one mocked cache case and return result/router spy / 返回结果与路由 spy。"""
        def reestimate(package, _circuit, _logic_gate_num):
            """Return fresh mock metrics from effective settings / 按当前配置返回新统计。"""
            fake = self._fake_router(package, None)
            return {
                **fake.statistic_route_details(),
                "circuit_duration": fake.estimate_qunoc_circuit_duration(),
                "gate_fidelity": fake.estimate_qunoc_logic_gate_fide(),
            }

        with patch.object(scalability.Package, "from_pickle", return_value=copy.deepcopy(self.package)), \
             patch.object(scalability, "get_program", return_value=_program()), \
             patch.object(scalability.Circuit, "from_pickle", return_value=object()), \
             patch.object(scalability, "Router", side_effect=self._fake_router) as router:
            router.reestimate_compiled_circuit.side_effect = reestimate
            result = scalability._route_once(
                self.common, self.spec, config or self.run_config, "bv"
            )
        return result, router

    def test_cache_hit_requires_new_signature(self):
        """Check legacy results miss and matching new signatures hit / 检查缓存版本隔离。"""
        self.cache.parent.mkdir(parents=True)
        self.cache.write_text(json.dumps({
            "algorithm": "bv", "routing_options": scalability.ROUTING_OPTION_DEFAULTS,
            "program_options": {}, "gate_fidelity": -1,
        }), encoding="utf-8")
        result, router = self._route()
        self.assertEqual(router.call_count, 1)
        self.assertIn("signature", result)
        self.assertNotEqual(result["gate_fidelity"], -1)
        cached, router = self._route()
        self.assertEqual(router.call_count, 0)
        self.assertEqual(result, cached)
        self.assertEqual(result, json.loads(json.dumps(result, allow_nan=False)))

    def test_each_eqst_setting_invalidates_cache(self):
        """Check model/version calibration and planner inputs / 所有 EQST 参数影响缓存。"""
        changed_values = {
            "model": "legacy", "reference_time_ns": 150, "reference_distance": 6,
            "g_max": 0.04, "max_segment_steps": 8, "beam_width": 32,
        }
        for name, value in changed_values.items():
            with self.subTest(name=name):
                baseline, _ = self._route()
                changed = copy.deepcopy(self.run_config)
                changed["eqst"][name] = value
                result, router = self._route(changed)
                self.assertEqual(router.call_count, 0)
                router.reestimate_compiled_circuit.assert_called_once()
                self.assertEqual(result["compilation_signature"], baseline["compilation_signature"])
                self.assertNotEqual(result["signature"], baseline["signature"])
        result, _ = self._route()
        result["signature"]["eqst_config"]["version"] = "obsolete-model-version"
        self.cache.write_text(json.dumps(result), encoding="utf-8")
        _, router = self._route()
        self.assertEqual(router.call_count, 0)
        router.reestimate_compiled_circuit.assert_called_once()

    def test_timing_fidelity_and_package_file_invalidate_cache(self):
        """Check effective physical inputs and package stat / 物理参数和文件变化使缓存失效。"""
        for owner, name, value in (
            (self.package.op_config, "qst_time_constant", 30),
            (self.package.op_config, "inter_time", 999),
            (self.package.op_config, "measure_time", 1234),
            (self.package.fab_model, "enhanced_fide", 0.95),
            (self.package.fab_model, "inter_fide", 0.94),
        ):
            with self.subTest(name=name):
                self._route()
                setattr(owner, name, value)
                _, router = self._route()
                self.assertEqual(router.call_count, 0)
                router.reestimate_compiled_circuit.assert_called_once()
        self._route()
        self.path.write_bytes(b"replacement package artifact with different size")
        _, router = self._route()
        self.assertEqual(router.call_count, 0)
        router.reestimate_compiled_circuit.assert_called_once()

    def test_full_route_uses_router_costs_and_one_batch_estimate(self):
        """Check real staged metrics propagate without raw-distance recomputation.

        输入合成分阶段电路；验证真实 Router 时间与保真度被直接使用，批次仅估计一次。
        """
        routers = []

        def factory(package, program, **_kwargs):
            """Wrap a real metric router for this test / 返回真实统计对象并记录 spy。"""
            router = _staged_router(package, program)
            router.circuit.to_pickle = Mock(side_effect=lambda path: Path(path).write_bytes(b"synthetic"))
            router.circuit.estimate_eqst_batches = Mock(wraps=router.circuit.estimate_eqst_batches)
            router.estimate_qunoc_logic_gate_fide = Mock(wraps=router.estimate_qunoc_logic_gate_fide)
            router.estimate_qunoc_circuit_duration = Mock(wraps=router.estimate_qunoc_circuit_duration)
            routers.append(router)
            return router

        with patch.object(scalability.Package, "from_pickle", return_value=copy.deepcopy(self.package)), \
             patch.object(scalability, "get_program", return_value=_program()), \
             patch.object(scalability, "Router", side_effect=factory):
            result = scalability._route_once(self.common, self.spec, self.run_config, "bv")
        router = routers[0]
        self.assertEqual(router.circuit.estimate_eqst_batches.call_count, 1)
        self.assertEqual(result["qst_dist"], 20)
        self.assertEqual(result["eqst_model"], "hamiltonian")
        self.assertEqual(result["eqst_batch_count"], 1)
        self.assertNotEqual(result["qst_effective_dist"], result["qst_dist"])
        expected = self.package.fab_model.enhanced_fide ** (result["qst_effective_dist"] / 10)
        self.assertAlmostEqual(result["gate_fidelity"], expected)
        batches = router.estimate_qunoc_circuit_duration.call_args.args[0]
        self.assertAlmostEqual(result["circuit_duration"], batches[0]["transfer_time_ns"])
        self.assertNotEqual(result["circuit_duration"], result["qst_dist"] * 25)
        self.assertNotIn("eqst_batches", result)
        self.assertEqual(result, json.loads(self.cache.read_text(encoding="utf-8")))

    def test_statistics_only_uses_saved_circuit_without_program_or_routing(self):
        """Re-estimate a real saved circuit without routing or rewriting its pickle.

        输入保存的 legacy 电路；强制禁止程序生成和编译，验证 Hamiltonian 重新统计且文件不变。
        """
        self._route({**self.run_config, "eqst": {"model": "legacy"}})
        package = copy.deepcopy(self.package)
        router = _staged_router(package, _program())
        circuit_path = self.cache.with_name("qunoc_circuit.pkl")
        # The production Circuit is picklable before attaching any method spies.
        router.circuit.to_pickle(str(circuit_path))
        cached = json.loads(self.cache.read_text(encoding="utf-8"))
        cached["compiled_circuit_identity"] = scalability._compiled_circuit_identity(circuit_path)
        self.cache.write_text(json.dumps(cached), encoding="utf-8")
        original = circuit_path.read_bytes()
        identity = scalability._compiled_circuit_identity(circuit_path)
        with patch.object(scalability.Package, "from_pickle", return_value=package), \
             patch.object(scalability, "get_program", side_effect=AssertionError("must not generate")), \
             patch.object(Router, "__init__", side_effect=AssertionError("must not construct router")), \
             patch.object(Router, "route_program_on_package", side_effect=AssertionError("must not route")), \
             patch.object(Circuit, "to_pickle", side_effect=AssertionError("must not overwrite circuit")):
            result = scalability._route_once(
                self.common, self.spec, {**self.run_config, "statistics_only": True}, "bv"
            )
        self.assertEqual(result["statistics_source"], "saved_circuit")
        self.assertEqual(result["eqst_model"], "hamiltonian")
        self.assertEqual(result["qst_dist"], 20)
        self.assertEqual(result["qst_effective_dist"], 10)
        self.assertEqual(circuit_path.read_bytes(), original)
        self.assertEqual(scalability._compiled_circuit_identity(circuit_path), identity)

    def test_statistics_only_missing_or_incompatible_metadata_never_routes(self):
        """Reject absent/mismatched artifacts rather than compile / 缺失或不匹配必须报错而非编译。"""
        config = {**self.run_config, "statistics_only": True}
        with patch.object(scalability.Package, "from_pickle", return_value=self.package), \
             patch.object(scalability, "get_program", side_effect=AssertionError("must not generate")), \
             patch.object(Router, "route_program_on_package", side_effect=AssertionError("must not route")):
            with self.assertRaisesRegex(ValueError, "statistics-only"):
                scalability._route_once(self.common, self.spec, config, "bv")
        self._route()
        config["routing"] = {"mapping_strategy": "sabre"}
        with self.assertRaisesRegex(ValueError, "statistics-only"):
            self._route(config)
        config.pop("routing")
        self.cache.with_name("qunoc_circuit.pkl").write_bytes(b"unrelated replacement artifact")
        with self.assertRaisesRegex(ValueError, "statistics-only"):
            self._route(config)

    def test_legacy_metadata_reuse_requires_explicit_statistics_only(self):
        """Allow explicit legacy migration with warning and no recompilation.

        输入无新版签名的历史元数据；仅显式 statistics-only 允许信任，并记录拓扑核验限制。
        """
        baseline, _ = self._route()
        old = {key: baseline[key] for key in (
            "quantum_program", "data_qubit_num", "logic_gate_num", "layout_package_path",
            "routing_options", "program_options",
        )}
        old["algorithm"] = old.pop("quantum_program")
        self.cache.write_text(json.dumps(old), encoding="utf-8")
        with self.assertLogs(level="WARNING") as warnings:
            result, router = self._route({**self.run_config, "statistics_only": True})
        self.assertEqual(router.call_count, 0)
        router.reestimate_compiled_circuit.assert_called_once()
        self.assertTrue(result["legacy_compilation_metadata_trusted"])
        self.assertIn("topology fingerprint", warnings.output[0])
        old["routing_options"] = {"mapping_strategy": "different"}
        self.cache.write_text(json.dumps(old), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "statistics-only"):
            self._route({**self.run_config, "statistics_only": True})

    def test_topology_change_recompiles_but_fidelity_change_reuses(self):
        """Separate compilation identity from estimator calibration / 拓扑变化才需重新编译。"""
        self._route()
        self.package.fab_model.enhanced_fide = 0.94
        result, router = self._route({**self.run_config, "reuse_result": False})
        self.assertEqual(router.call_count, 0)
        self.assertEqual(result["statistics_source"], "saved_circuit")
        self.package.package_graph.remove_edge((0, 0), (1, 0))
        result, router = self._route()
        self.assertEqual(router.call_count, 1)
        self.assertEqual(result["statistics_source"], "new_compilation")
        _, router = self._route({**self.run_config, "recompile": True})
        self.assertEqual(router.call_count, 1)

    def test_legacy_full_route_keeps_raw_distance_and_skips_planner(self):
        """Check legacy full-route estimates never call the Hamiltonian planner.

        输入 legacy 合成分阶段电路；验证旧时序与距离保持不变，不调用 Hamiltonian 规划器。
        """
        package = _package()
        router = _staged_router(package, _program())
        with patch.object(router.circuit, "estimate_eqst_batches") as planner:
            details = router.statistic_route_details()
            duration = router.estimate_qunoc_circuit_duration(details["eqst_batches"])
            fidelity = router.estimate_qunoc_logic_gate_fide(details)
        planner.assert_not_called()
        self.assertEqual(details["eqst_model"], "legacy")
        self.assertEqual(details["qst_dist"], 20)
        self.assertEqual(details["qst_effective_dist"], 20)
        self.assertEqual(details["eqst_batch_count"], 0)
        self.assertEqual(duration, 10 * package.op_config.qst_time_constant)
        self.assertAlmostEqual(fidelity, package.fab_model.enhanced_fide ** 2)

    def test_csv_preserves_raw_distance_and_rebuilds_old_header(self):
        """Check appended columns and no duplicate/stale rows / CSV 保留原始距离并重建旧表头。"""
        self.spec.defaults = self.run_config
        output = Path(self.spec.exp_dir) / self.spec.output_csv
        output.parent.mkdir(parents=True)
        output.write_text("old,header\nstale,row\n", encoding="utf-8")
        result, _ = self._route()
        for _ in range(2):
            with patch.object(scalability, "_route_once", return_value=result), \
                 patch.object(scalability.logging, "basicConfig"):
                scalability._run_experiment(self.common, self.spec)
            with output.open(newline="", encoding="utf-8") as file:
                reader = csv.DictReader(file)
                rows = list(reader)
                self.assertEqual(reader.fieldnames[-4:], scalability.EQST_CSV_FIELDS)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["QSTDist"], "10")
            self.assertEqual(rows[0]["QSTEffectiveDist"], "7.5")
            self.assertEqual(rows[0]["EQSTModel"], "hamiltonian")


class AblationCompatibilityTests(unittest.TestCase):
    def test_dummy_preserves_three_gate_swaps_and_raw_distance(self):
        """Check legacy dummy accounting remains separate / 保留 dummy 三门 SWAP 语义。"""
        package = _package()
        graph = nx.DiGraph()
        actions = [
            Board(0, (0, 0), (0, 1)), Transport(0, (0, 0), (10, 0)),
            SwapGate(0, (10, 0), (10, 1), inter_flag=False),
        ]
        graph.add_node(-1)
        for action in actions:
            graph.add_node(action.id, action=action)
        router = SimpleNamespace(
            package=package, package_graph=nx.Graph(),
            dummy_circuit=SimpleNamespace(action_depend_graph=graph),
        )
        details = ablation._statistic_dummy_route_details(router)
        self.assertEqual((details["intra_gate"], details["inter_gate"], details["qst_dist"]), (3, 3, 10))
        self.assertEqual(details["qst_effective_dist"], 10)
        self.assertEqual(details["eqst_model"], "legacy")
        expected = 0.98 ** 3 * package.fab_model.inter_fide ** 3 * package.fab_model.enhanced_fide ** 10
        self.assertAlmostEqual(details["total_fide"], expected)
        self.assertAlmostEqual(details["total_neglog_fidelity"], -math.log(expected))
        self.assertEqual(ablation._statistic_dummy_route_results(router)[:4], (3, 3, 10, 0))

    def test_dummy_normalized_fidelity_survives_total_underflow(self):
        """Check a long dummy circuit retains its per-gate fidelity / 长电路不影响稳定归一化。"""
        package = _package()
        package.fab_model.enhanced_fide = 0.9
        self.assertEqual(0.9 ** 100000, 0)
        self.assertAlmostEqual(ablation._logic_gate_fidelity(package, 100000, 0, 0, 100000), 0.9)

    def test_mech_baseline_is_labeled_separately(self):
        """Check MECH raw-distance metrics retain their distinct baseline policy.

        输入合成 MECH 结果；验证独立基线标签、legacy 距离和缺失总保真度保持不变。
        """
        args = SimpleNamespace(
            chiplet_length=10, inter_sparsity=3, geometry="square", seed=42,
            benchmark_options={}, mech_eqst_basic_time_ns=1000.0,
            save_circuits=False, inter_fidelity=0.983,
        )
        benchmark = {"logic_gate_num": 10, "benchmark_hash": "synthetic"}
        raw = {"logic_gate_num": 10, "on_chip": 2, "cross_chip": 3, "eqst_dist": 17}
        with patch.object(ablation, "_run_single_mech_eqst_benchmark", return_value=({"bv": raw}, {})), \
             patch.object(ablation, "get_mech_eqst_circuit_duration", return_value=1234):
            result = ablation._run_mech(40, "bv", 100, benchmark, Path("unused"), args)
        self.assertEqual(result["eqst_model"], "legacy")
        self.assertEqual(result["eqst_accounting_policy"], "mech_eqst_raw_distance_v1")
        self.assertEqual(result["qst_dist"], result["qst_effective_dist"])
        self.assertEqual(result["qst_dist"], 17)
        self.assertIsNone(result["total_neglog_fidelity"])
        self.assertIsNone(result["total_fide"])
        row = ablation._result_row({
            **result, "strategy": "mech_eqst", "package_size": 40, "quantum_program": "bv",
        })
        self.assertEqual(row["Baseline"], "mech_eqst")
        self.assertEqual(row["EQSTModel"], "legacy")
        self.assertEqual(result, json.loads(json.dumps(result, allow_nan=False)))

    def test_hamiltonian_ablation_is_explicitly_rejected(self):
        """Reject unsupported mode before layout loading / 不将 baseline 误标为 Hamiltonian。"""
        with self.assertRaisesRegex(ValueError, "staged Hamiltonian batches"):
            ablation._preflight(SimpleNamespace(eqst_model="hamiltonian"))

    def test_old_csv_rows_migrate_as_legacy_baselines(self):
        """Check historical rows stay labeled legacy / 迁移旧行时保持 baseline 标签。"""
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "output.csv"
            source = dict.fromkeys(ablation.PRE_EQST_ROW_FIELDS, "0")
            source.update(Baseline="qunoc_wo_opt", PackageSize="40", QuantumProgram="bv", QSTDist="19")
            with path.open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=ablation.PRE_EQST_ROW_FIELDS)
                writer.writeheader()
                writer.writerow(source)
            rows, migrated = ablation._load_existing_rows(path)
            self.assertTrue(migrated)
            row = next(iter(rows.values()))
            self.assertEqual(row["EQSTModel"], "legacy")
            self.assertEqual(row["QSTEffectiveDist"], "19")
            self.assertEqual(row["TotalNegLogFidelity"], "")
            ablation._write_rows(path, rows)
            self.assertFalse(ablation._load_existing_rows(path)[1])

    def test_ablation_signature_records_legacy_policy_and_effective_inputs(self):
        """Check baseline signatures cannot inherit Hamiltonian labels / 基线缓存记录实际 legacy 策略。"""
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "40_heuristic" / "package.pkl"
            path.parent.mkdir()
            path.write_bytes(b"package")
            args = SimpleNamespace(
                layout_root=Path(temporary), layout_strategy="heuristic", mapping_strategy="random",
                benchmark_options={}, seed=42, geometry="square", inter_sparsity=3,
                inter_fidelity=0.983, chiplet_length=10,
            )
            shared = OperationConfig(eqst_model="hamiltonian")
            package = _package(shared)
            benchmark = {"logic_gate_num": 10, "benchmark_hash": "synthetic"}
            with patch.object(ablation.Package, "from_pickle", return_value=package):
                signature = ablation._signature("qunoc_wo_opt", 40, "bv", benchmark, args)
            self.assertEqual(shared.eqst_model, "hamiltonian")
            self.assertEqual(signature["eqst_config"]["model"], "legacy")
            self.assertEqual(signature["eqst_model"], "legacy")
            self.assertEqual(signature["eqst_accounting_policy"], "dummy_raw_distance_three_gate_swap_v1")
            self.assertIn("enhanced_fide", signature["fabrication"])
            self.assertGreater(signature["schema_version"], 1)
            self.assertEqual(signature, json.loads(json.dumps(signature, allow_nan=False)))
            cache = Path(temporary) / "cache.json"
            cache.write_text(json.dumps({"status": "complete", "signature": {**signature, "schema_version": 1}}))
            self.assertIsNone(ablation._load_cached_result(cache, signature))


if __name__ == "__main__":
    unittest.main()
