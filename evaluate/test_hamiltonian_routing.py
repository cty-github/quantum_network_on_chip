"""Hamiltonian stage-boundary routing regressions / Hamiltonian 阶段边界路由回归测试.

Run / 运行: python -m unittest evaluate.test_hamiltonian_routing -v
"""

import json
import math
import pickle
import unittest
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch

import networkx as nx

from qunoc.element.benchmark import Control
from qunoc.element.chiplet import OperationConfig, TransferChiplet
from qunoc.element.package import Package
from qunoc.routing.circuit import (
    Alight,
    Board,
    Circuit,
    DirectGate,
    DummyCircuit,
    Enter,
    Exit,
    GhzPreparation,
    Interchange,
    SwapGate,
    Teleport,
    Transport,
)


class MinimalPackage:
    """Resolve real transfer dies without layout construction / 无需布局即可查询真实传输裸片."""

    def __init__(self, *dies):
        """Store dies / 保存裸片；input actual TransferChiplets / 输入实际传输裸片."""
        self.transfer_chiplets = dies
        self.compute_die = object()

    def get_node_chiplet(self, node):
        """Return node's die / 返回节点裸片；input coordinate / 输入坐标."""
        matches = [die for die in self.transfer_chiplets if die.has_node(node)]
        if len(matches) > 1:
            raise ValueError("test dies overlap")
        return matches[0] if matches else self.compute_die


def make_circuit(package, initial, concurrent=False, model="hamiltonian"):
    """Build a minimal real Circuit / 构造真实最小电路；inputs package/map/mode / 输入封装、映射及模式."""
    p2l = {(x, y): -1 for x in range(32) for y in range(32)}
    p2l.update({node: q for q, node in initial.items()})
    circuit = Circuit(
        len(initial), (32, 32), dict(initial), p2l,
        nx.DiGraph() if concurrent else None,
        OperationConfig(eqst_model=model, inter_time=100),
    )
    circuit.occupied_balises.update(
        node for node in initial.values()
        if isinstance(package.get_node_chiplet(node), TransferChiplet)
    )
    return circuit


def riding_stage(circuit):
    """Append a Boarding/Riding pair / 添加登乘与行驶阶段；input circuit / 输入电路."""
    circuit.extend_stage(True)
    return circuit.extend_stage(False)


def set_final_mapping(circuit, mapping):
    """Set synthetic history's final maps / 设置合成历史终态；inputs circuit/map / 输入电路与映射."""
    circuit.curr_l2p_map.clear()
    circuit.curr_l2p_map.update(mapping)
    for node in circuit.curr_p2l_map:
        circuit.curr_p2l_map[node] = -1
    circuit.curr_p2l_map.update({node: q for q, node in mapping.items()})


def fake_estimate(sources, targets, chain_length, **kwargs):
    """Return deterministic paired costs / 返回确定配对成本；input positions / 输入位置，output cost dict / 输出成本."""
    distance = sum(abs(source - target) for source, target in zip(sources, targets))
    return {"transfer_time_ns": distance * 10.0, "effective_distance_edges": float(distance)}


class OperationConfigTests(unittest.TestCase):
    """Model compatibility and validation / 模型兼容性与校验."""

    def test_defaults_old_pickle_and_json(self):
        """Old pickles remain legacy / 旧 pickle 保持线性模型；output assertions / 输出断言."""
        config = OperationConfig()
        self.assertEqual(config.get_eqst_config()["model"], "legacy")
        for name in tuple(vars(config)):
            if name.startswith("eqst_"):
                delattr(config, name)
        restored = pickle.loads(pickle.dumps(config))
        settings = restored.get_eqst_config()
        self.assertEqual(settings["model"], "legacy")
        self.assertEqual(settings["reference_time_ns"], 125)
        self.assertEqual(settings["reference_distance"], 5)
        self.assertAlmostEqual(settings["g_max"], 2 * math.pi * 6 / 1000)
        json.dumps(settings, allow_nan=False)
        self.assertNotIn("eqst_model", vars(restored))

    def test_get_config_validates_mutated_values(self):
        """Validate edits on copied configs / 校验复制后修改配置；output errors / 输出错误."""
        for name, value in (
            ("eqst_model", "bad"), ("eqst_reference_time_ns", 0),
            ("eqst_reference_time_ns", math.nan), ("eqst_reference_distance", math.inf),
            ("eqst_reference_distance", -1), ("eqst_g_max", 0),
            ("eqst_g_max", math.inf), ("eqst_beam_width", 2.5),
            ("eqst_max_segment_steps", True), ("eqst_beam_width", 0),
        ):
            with self.subTest(name=name, value=value):
                config = deepcopy(OperationConfig())
                setattr(config, name, value)
                with self.assertRaises(ValueError):
                    config.get_eqst_config()
        config = OperationConfig(eqst_reference_distance=2.5, eqst_g_max=None)
        self.assertEqual(config.get_eqst_config()["reference_distance"], 2.5)
        self.assertIsNone(config.get_eqst_config()["g_max"])
        with self.assertRaises(ValueError):
            OperationConfig(eqst_model="invalid")


class BoundaryReplayTests(unittest.TestCase):
    """Final-history replay, never shared-map snapshots / 最终历史重放而非共享映射快照."""

    def test_immediate_exchange_reused_source_and_stationary_resident(self):
        """Include implicit exchanges and spectators / 包含隐式交换与静止态；output ID-paired batch / 输出按 ID 配对批次."""
        die = TransferChiplet(.99, axis=1, pos=2, start=10, end=20)
        package = MinimalPackage(die)
        circuit = make_circuit(package, {0: (2, 10), 1: (2, 15), 2: (2, 13)})
        actions = [Transport(0, (2, 10), (2, 15)), Transport(1, (2, 10), (2, 18))]
        stage = riding_stage(circuit)
        for action in actions:
            self.assertTrue(stage.add_transport_action_immediate(action))
        original_actions = [dict(vars(action)) for action in actions]
        original_maps = deepcopy((circuit.curr_l2p_map, circuit.curr_p2l_map))
        with patch("qunoc.hamiltonian.estimate_transfer", side_effect=fake_estimate) as estimate:
            batches = circuit.estimate_eqst_batches(package)
        self.assertEqual(len(batches), 1)
        self.assertEqual(estimate.call_count, 1)
        batch = batches[0]
        self.assertEqual(batch["die"], (1, 2, 10, 20))
        self.assertEqual(batch["chain_length"], 11)
        self.assertEqual(batch["logical_ids"], (0, 1, 2))
        self.assertEqual(batch["sources"], (0, 5, 3))
        self.assertEqual(batch["targets"], (5, 8, 3))
        self.assertEqual(batch["transport_action_ids"], tuple(action.id for action in actions))
        self.assertTrue(estimate.call_args.kwargs["ordered"])
        self.assertEqual(original_actions, [dict(vars(action)) for action in actions])
        self.assertEqual(original_maps, (circuit.curr_l2p_map, circuit.curr_p2l_map))

    def test_continuous_legs_and_round_trip_identity(self):
        """Continuous legs use boundary positions / 连续传输使用边界位置；round trip is identity / 往返为恒等映射."""
        package = MinimalPackage(TransferChiplet(.99, 0, 4, 10, 20))
        circuit = make_circuit(package, {0: (10, 4)})
        stage = riding_stage(circuit)
        first = Transport(0, (10, 4), (15, 4))
        second = Transport(0, (15, 4), (19, 4))
        self.assertTrue(stage.add_transport_action_immediate(first))
        self.assertTrue(stage.add_transport_action_immediate(second))
        with patch("qunoc.hamiltonian.estimate_transfer", side_effect=fake_estimate):
            batch = circuit.estimate_eqst_batches(package)[0]
        self.assertEqual((batch["sources"], batch["targets"]), ((0,), (9,)))
        # Start was never a locked target; a return is admitted without scheduler changes.
        # 起点尚未锁定，原调度规则允许返回。
        back = Transport(0, (19, 4), (10, 4))
        self.assertTrue(stage.add_transport_action_immediate(back))
        batches = circuit.estimate_eqst_batches(package)
        self.assertEqual(batches[0]["transfer_time_ns"], 0)
        self.assertEqual(batches[0]["effective_distance_edges"], 0)
        self.assertEqual(sum(action.get_distance() for action in stage.circuit_actions), 18)

    def test_reciprocal_concurrent_is_directed_not_double_swap(self):
        """Concurrent reciprocal requests swap once / 并发双向请求只交换一次；output ordered permutation / 输出有序置换."""
        package = MinimalPackage(TransferChiplet(.99, 1, 2, 10, 20))
        circuit = make_circuit(package, {0: (2, 10), 1: (2, 15)}, concurrent=True)
        stage = riding_stage(circuit)
        forward = Transport(0, (2, 10), (2, 15))
        reverse = Transport(1, (2, 15), (2, 10))
        self.assertFalse(stage.add_transport_action(forward))
        self.assertTrue(stage.add_transport_action(reverse))
        self.assertTrue(stage.add_transport_action(forward))
        batch = circuit.estimate_eqst_batches(package)[0]
        self.assertEqual(batch["sources"], (0, 5))
        self.assertEqual(batch["targets"], (5, 0))
        self.assertGreater(batch["transfer_time_ns"], 0)
        self.assertGreater(batch["effective_distance_edges"], 0)
        self.assertEqual(stage.circuit_actions, [reverse, forward])

    def test_retroactive_insertion_ignores_shared_current_maps(self):
        """Earlier insertion invalidates construction snapshots / 早期插入使构造快照失效；output finalized boundaries / 输出最终边界."""
        package = MinimalPackage(TransferChiplet(.99, 1, 2, 10, 20))
        circuit = make_circuit(package, {0: (2, 10), 1: (3, 13), 2: (3, 14)})
        first = riding_stage(circuit)
        self.assertTrue(first.add_transport_action_immediate(Transport(0, (2, 10), (2, 15))))
        second = riding_stage(circuit)
        self.assertTrue(second.add_direct_action_immediate(DirectGate(Control(1, 2), (3, 13), (3, 14), False)))
        late_action = Transport(0, (2, 15), (2, 19))
        self.assertEqual(circuit.add_action_into_earliest_stage(late_action), 1)
        self.assertIs(first.curr_l2p_map, second.curr_l2p_map)
        self.assertEqual(first.curr_l2p_map[0], (2, 19))
        with patch("qunoc.hamiltonian.estimate_transfer", side_effect=fake_estimate):
            batches = circuit.estimate_eqst_batches(package)
        self.assertEqual([batch["stage_index"] for batch in batches], [1, 3])
        self.assertEqual((batches[0]["sources"], batches[0]["targets"]), ((0,), (9,)))
        self.assertEqual((batches[1]["sources"], batches[1]["targets"]), ((9,), (9,)))

    def test_ghz_board_alight_interchange_teleport_lifecycle(self):
        """Replay ancilla lifetime across actual die boundaries / 跨裸片重放辅助态生命周期；output batches / 输出批次."""
        for concurrent in (False, True):
            with self.subTest(concurrent=concurrent):
                package = MinimalPackage(TransferChiplet(.99, 1, 2, 10, 15), TransferChiplet(.99, 1, 2, 16, 20))
                circuit = make_circuit(package, {0: (3, 20)}, concurrent=concurrent)
                board = circuit.extend_stage(True)
                ghz = GhzPreparation(1, [5], [0], [(2, 10)], [(3, 19)], [((2, 10), None)], [])
                ghz.record_ancilla_qubit_nodes(5, [(2, 10), (2, 15), (2, 16), (2, 19), (3, 19)])
                self.assertTrue(board.add_ghz_action(ghz) if concurrent else board.add_ghz_action_immediate(ghz))
                first = circuit.extend_stage(False)
                move = Transport(5, (2, 10), (2, 15))
                self.assertTrue(first.add_transport_action(move) if concurrent else first.add_transport_action_immediate(move))
                board = circuit.extend_stage(True)
                move = Interchange(5, ((2, 15), (2, 16)))
                self.assertTrue(board.add_interchange_action(move) if concurrent else board.add_interchange_action_immediate(move))
                second = circuit.extend_stage(False)
                move = Transport(5, (2, 16), (2, 19))
                self.assertTrue(second.add_transport_action(move) if concurrent else second.add_transport_action_immediate(move))
                board = circuit.extend_stage(True)
                move = Alight(5, (2, 19), (3, 19))
                self.assertTrue(board.add_alight_action(move) if concurrent else board.add_alight_action_immediate(move))
                last = circuit.extend_stage(False)
                teleport = Teleport(5, 0, (3, 19), (3, 20), True, False)
                self.assertTrue(last.add_teleport_action(teleport) if concurrent else last.add_teleport_action_immediate(teleport))
                with patch("qunoc.hamiltonian.estimate_transfer", side_effect=fake_estimate) as estimate:
                    batches = circuit.estimate_eqst_batches(package)
                self.assertEqual(estimate.call_count, 2)
                self.assertEqual([batch["logical_ids"] for batch in batches], [(5,), (5,)])
                self.assertEqual([batch["sources"] for batch in batches], [(0,), (0,)])
                self.assertEqual([batch["targets"] for batch in batches], [(5,), (3,)])
                self.assertNotIn(5, circuit.curr_l2p_map)

    def test_boarding_implicit_exchange_changes_next_stage_residents(self):
        """Boarding swaps residents before Riding / 登乘交换决定行驶居民；output new resident IDs / 输出新驻留 ID."""
        package = MinimalPackage(TransferChiplet(.99, 1, 2, 10, 20))
        circuit = make_circuit(package, {0: (3, 10), 1: (2, 10)})
        board = circuit.extend_stage(True)
        self.assertTrue(board.add_board_action_immediate(Board(0, (2, 10), (3, 10))))
        ride = circuit.extend_stage(False)
        self.assertTrue(ride.add_transport_action_immediate(Transport(0, (2, 10), (2, 15))))
        with patch("qunoc.hamiltonian.estimate_transfer", side_effect=fake_estimate):
            batch = circuit.estimate_eqst_batches(package)[0]
        self.assertEqual(batch["logical_ids"], (0,))
        self.assertEqual(circuit.curr_l2p_map[1], (3, 10))

    def test_swap_metadata_resolves_mixed_enter_exit(self):
        """Stage metadata disambiguates temporary concurrent overlaps / 阶段元数据消除并发重叠歧义；output replay / 输出重放."""
        package = MinimalPackage()
        circuit = make_circuit(package, {0: (3, 10), 1: (3, 11), 2: (3, 12)}, concurrent=True)
        stage = riding_stage(circuit)
        circuit.occupied_platforms.add((3, 10))
        enter = Enter(1, (3, 10), (3, 11), False)
        exit_action = Exit(0, (3, 10), (3, 11), False)
        self.assertFalse(stage.add_exit_action(exit_action))
        self.assertTrue(stage.add_enter_action(enter))
        swap = SwapGate(1, (3, 10), (3, 12), False)
        before_action = dict(vars(swap))
        self.assertTrue(stage.add_swap_action(swap))
        self.assertTrue(stage.add_exit_action(exit_action))
        self.assertEqual(circuit.curr_l2p_map, {0: (3, 11), 1: (3, 12), 2: (3, 10)})
        self.assertEqual(stage.swap_logical_ids[swap.id], (1, 2))
        self.assertEqual(vars(swap), before_action)
        self.assertEqual(circuit.estimate_eqst_batches(package), [])
        del stage.swap_logical_ids
        with self.assertRaisesRegex(ValueError, "Ambiguous SwapGate.*recompile"):
            circuit.estimate_eqst_batches(package)

    def test_old_unambiguous_swap_and_mode_fallback(self):
        """Old unambiguous histories still replay / 旧无歧义历史仍可重放；output valid maps / 输出有效映射."""
        for concurrent in (False, True):
            with self.subTest(concurrent=concurrent):
                package = MinimalPackage()
                circuit = make_circuit(package, {0: (3, 10), 1: (3, 11)}, concurrent=concurrent)
                stage = riding_stage(circuit)
                swap = SwapGate(0, (3, 10), (3, 11), False)
                self.assertTrue(stage.add_swap_action(swap) if concurrent else stage.add_swap_action_immediate(swap))
                del stage.swap_logical_ids
                del circuit.routing_mode
                self.assertEqual(circuit.estimate_eqst_batches(package), [])
                self.assertEqual(circuit.curr_l2p_map, {0: (3, 11), 1: (3, 10)})

    def test_explicit_mode_does_not_follow_later_graph_assignment(self):
        """Recorded constructor mode remains authoritative / 构造模式保持权威；output correct swaps / 输出正确交换."""
        package = MinimalPackage(TransferChiplet(.99, 1, 2, 10, 20))
        circuit = make_circuit(package, {0: (2, 10), 1: (2, 15)})
        stage = riding_stage(circuit)
        self.assertTrue(stage.add_transport_action_immediate(Transport(0, (2, 10), (2, 15))))
        circuit.action_depend_graph = nx.DiGraph()
        with patch("qunoc.hamiltonian.estimate_transfer", side_effect=fake_estimate):
            self.assertEqual(circuit.estimate_eqst_batches(package)[0]["targets"], (5, 0))

    def test_stage_and_final_map_validation(self):
        """Reject collisions and stale final maps / 拒绝碰撞与过时终态；output clear errors / 输出清晰错误."""
        package = MinimalPackage(TransferChiplet(.99, 1, 2, 10, 20))
        circuit = make_circuit(package, {0: (2, 10), 1: (2, 15)}, concurrent=True)
        stage = riding_stage(circuit)
        stage.circuit_actions.append(Transport(0, (2, 10), (2, 15)))
        with self.assertRaisesRegex(ValueError, "duplicate logical positions.*stage 1"):
            circuit.estimate_eqst_batches(package)
        circuit = make_circuit(package, {0: (2, 10)})
        stage = riding_stage(circuit)
        self.assertTrue(stage.add_transport_action_immediate(Transport(0, (2, 10), (2, 15))))
        circuit.curr_l2p_map[0] = (2, 16)
        with patch("qunoc.hamiltonian.estimate_transfer", side_effect=fake_estimate) as estimate:
            with self.assertRaisesRegex(ValueError, "curr_l2p_map"):
                circuit.estimate_eqst_batches(package)
            estimate.assert_not_called()
        circuit.curr_l2p_map[0] = (2, 15)
        circuit.curr_p2l_map[(2, 10)] = 0
        with self.assertRaisesRegex(ValueError, "curr_p2l_map.*recompile"):
            circuit.estimate_eqst_batches(package)

    def test_reject_riding_resident_change_and_cross_die_transport(self):
        """Die boundaries require Boarding / 裸片边界必须登乘处理；output invariant errors / 输出不变量错误."""
        package = MinimalPackage(TransferChiplet(.99, 1, 2, 10, 15), TransferChiplet(.99, 1, 2, 16, 20))
        circuit = make_circuit(package, {0: (3, 10)})
        stage = riding_stage(circuit)
        stage.circuit_actions.append(Board(0, (2, 10), (3, 10)))
        set_final_mapping(circuit, {0: (2, 10)})
        with self.assertRaisesRegex(ValueError, "changes resident logical IDs"):
            circuit.estimate_eqst_batches(package)
        circuit = make_circuit(package, {0: (2, 15)})
        stage = riding_stage(circuit)
        self.assertTrue(stage.add_transport_action_immediate(Transport(0, (2, 15), (2, 16))))
        with self.assertRaisesRegex(ValueError, "one TransferChiplet"):
            circuit.estimate_eqst_batches(package)

    def test_mutated_history_is_not_identity_cached(self):
        """Replay changes on each call / 每次调用重新读取历史；output refreshed boundaries / 输出更新边界."""
        package = MinimalPackage(TransferChiplet(.99, 1, 2, 10, 20))
        circuit = make_circuit(package, {0: (2, 10)})
        stage = riding_stage(circuit)
        self.assertTrue(stage.add_transport_action_immediate(Transport(0, (2, 10), (2, 15))))
        with patch("qunoc.hamiltonian.estimate_transfer", side_effect=fake_estimate):
            before = circuit.estimate_eqst_batches(package)
            self.assertTrue(stage.add_transport_action_immediate(Transport(0, (2, 15), (2, 19))))
            after = circuit.estimate_eqst_batches(package)
        self.assertEqual(before[0]["targets"], (5,))
        self.assertEqual(after[0]["targets"], (9,))

    def test_real_package_lookup_both_axes_and_full_geometry(self):
        """Exercise Package lookup with actual Qubit ownership / 使用真实 Package 裸片归属查询；output local offsets / 输出局部偏移."""
        dies = [TransferChiplet(.99, 0, 4, 10, 20), TransferChiplet(.99, 1, 5, 10, 20)]
        package = Package.__new__(Package)
        package.transfer_chiplets = dies
        package.package_graph = nx.Graph()
        for die in dies:
            for node in die.get_nodes():
                package.package_graph.add_node(node, qubit=SimpleNamespace(chiplet=die))
        circuit = make_circuit(package, {0: (12, 4), 1: (5, 13)})
        stage = riding_stage(circuit)
        self.assertTrue(stage.add_transport_action_immediate(Transport(0, (12, 4), (15, 4))))
        self.assertTrue(stage.add_transport_action_immediate(Transport(1, (5, 13), (5, 18))))
        with patch("qunoc.hamiltonian.estimate_transfer", side_effect=fake_estimate):
            batches = circuit.estimate_eqst_batches(package)
        self.assertEqual([batch["die"] for batch in batches], [(0, 4, 10, 20), (1, 5, 10, 20)])
        self.assertEqual([batch["sources"] for batch in batches], [(2,), (3,)])
        self.assertEqual([batch["targets"] for batch in batches], [(5,), (8,)])
        self.assertEqual([batch["chain_length"] for batch in batches], [11, 11])


class HamiltonianDurationTests(unittest.TestCase):
    """Batch timing and original dependency preservation / 批次计时与原依赖保持."""

    def test_real_core_canonical_direct_and_conflict_batches(self):
        """Match canonical core calibration after actual stage replay / 真实阶段重放后匹配核心标定.
        Input / 输入: direct and conflicting transfer histories / 直接与冲突传输历史.
        Output / 输出: exact effective distances and calibrated durations / 精确等效距离与标定时间.
        """
        cases = (
            ((0, 5, 2), (9, 19, 13), (0, 1, 2), 34, 209.16500663351889),
            ((0, 9, 5), (9, 10, 19), (1, 0, 2), 24, 520.1381155644951),
        )
        for sources, targets, action_order, raw_distance, expected_time in cases:
            with self.subTest(sources=sources):
                package = MinimalPackage(TransferChiplet(.99, 1, 2, 7, 26))
                circuit = make_circuit(package, {q: (2, position + 7) for q, position in enumerate(sources)})
                stage = riding_stage(circuit)
                for q in action_order:
                    self.assertTrue(stage.add_transport_action_immediate(
                        Transport(q, (2, sources[q] + 7), (2, targets[q] + 7))
                    ))
                batches = circuit.estimate_eqst_batches(package)
                self.assertEqual(len(batches), 1)
                self.assertEqual(batches[0]["sources"], sources)
                self.assertEqual(batches[0]["targets"], targets)
                self.assertEqual(sum(action.get_distance() for action in stage.circuit_actions), raw_distance)
                self.assertEqual(batches[0]["effective_distance_edges"], 34)
                self.assertAlmostEqual(batches[0]["transfer_time_ns"], expected_time, places=9)
                self.assertAlmostEqual(circuit.estimate_duration(package, eqst_batches=batches), expected_time, places=9)

    def test_stage_die_parallel_max_and_single_primitive_call(self):
        """Dies run in parallel, stages sum / 裸片并行而阶段求和；output one estimate per batch / 每批一次估计."""
        package = MinimalPackage(TransferChiplet(.99, 1, 2, 10, 20), TransferChiplet(.99, 1, 4, 10, 20))
        circuit = make_circuit(package, {0: (2, 10), 1: (4, 10), 2: (3, 10), 3: (3, 11)})
        first = riding_stage(circuit)
        self.assertTrue(first.add_transport_action_immediate(Transport(0, (2, 10), (2, 15))))
        self.assertTrue(first.add_transport_action_immediate(Transport(1, (4, 10), (4, 18))))
        gate = DirectGate(Control(2, 3), (3, 10), (3, 11), False)
        self.assertTrue(first.add_direct_action_immediate(gate))
        second = riding_stage(circuit)
        self.assertTrue(second.add_transport_action_immediate(Transport(0, (2, 15), (2, 19))))
        with patch("qunoc.hamiltonian.estimate_transfer", side_effect=fake_estimate) as estimate:
            batches = circuit.estimate_eqst_batches(package)
            duration = circuit.estimate_duration(package, eqst_batches=batches)
        self.assertEqual(len(batches), 4)  # Includes second die's stationary second stage.
        self.assertEqual(estimate.call_count, 4)
        self.assertEqual(duration, 80 + 40)

    def test_external_dag_contraction_preserves_incoming_outgoing_edges(self):
        """Contract internal legs, keep surrounding work / 收缩内部腿并保留前后工作；output critical path / 输出关键路径."""
        package = MinimalPackage(TransferChiplet(.99, 1, 2, 10, 20))
        circuit = make_circuit(package, {0: (2, 10), 1: (3, 10), 2: (3, 11)}, concurrent=True)
        stage = riding_stage(circuit)
        before = DirectGate(Control(1, 2), (3, 10), (3, 11), False)
        first = Transport(0, (2, 10), (2, 15))
        second = Transport(0, (2, 15), (2, 19))
        after = DirectGate(Control(1, 2), (3, 10), (3, 11), False)
        stage.circuit_actions.extend([before, first, second, after])
        set_final_mapping(circuit, {0: (2, 19), 1: (3, 10), 2: (3, 11)})
        for action in stage.circuit_actions:
            circuit.action_depend_graph.add_node(action.id, action=action)
        circuit.action_depend_graph.add_edges_from([(before.id, first.id), (first.id, second.id), (second.id, after.id)])
        original_edges = set(circuit.action_depend_graph.edges)
        with patch("qunoc.hamiltonian.estimate_transfer", side_effect=fake_estimate) as estimate:
            self.assertEqual(circuit.estimate_duration(package), 25 + 90 + 25)
        self.assertEqual(estimate.call_count, 1)
        self.assertEqual(set(circuit.action_depend_graph.edges), original_edges)
        self.assertEqual(stage.circuit_actions, [before, first, second, after])

    def test_external_dag_rejects_observed_intermediate_state(self):
        """An observer between same-die legs cannot be hidden / 同裸片腿间观察者不可隐藏；output cycle error / 输出成环错误."""
        package = MinimalPackage(TransferChiplet(.99, 1, 2, 10, 20))
        circuit = make_circuit(package, {0: (2, 10), 1: (3, 10), 2: (3, 11)}, concurrent=True)
        stage = riding_stage(circuit)
        first = Transport(0, (2, 10), (2, 15))
        gate = DirectGate(Control(1, 2), (3, 10), (3, 11), False)
        second = Transport(0, (2, 15), (2, 19))
        stage.circuit_actions.extend([first, gate, second])
        set_final_mapping(circuit, {0: (2, 19), 1: (3, 10), 2: (3, 11)})
        for action in stage.circuit_actions:
            circuit.action_depend_graph.add_node(action.id, action=action)
        circuit.action_depend_graph.add_edges_from([(first.id, gate.id), (gate.id, second.id)])
        with patch("qunoc.hamiltonian.estimate_transfer", side_effect=fake_estimate):
            with self.assertRaisesRegex(ValueError, "contraction.*dependency cycle.*intermediate"):
                circuit.estimate_duration(package)

    def test_stage_local_nontransport_dependencies(self):
        """Local compute DAG remains serial beside parallel EQST / 本地计算依赖串行并与 EQST 并行；output max time / 输出最大时间."""
        package = MinimalPackage(TransferChiplet(.99, 1, 2, 10, 20))
        circuit = make_circuit(package, {0: (2, 10), 1: (3, 10), 2: (3, 11)})
        stage = riding_stage(circuit)
        self.assertTrue(stage.add_transport_action_immediate(Transport(0, (2, 10), (2, 13))))
        for _ in range(3):
            self.assertTrue(stage.add_direct_action_immediate(DirectGate(Control(1, 2), (3, 10), (3, 11), False)))
        with patch("qunoc.hamiltonian.estimate_transfer", side_effect=fake_estimate):
            self.assertEqual(circuit.estimate_duration(package), 75)

    def test_legacy_and_old_pickle_timing_unchanged(self):
        """Default and old configurations retain linear timing / 默认与旧配置保持线性计时；output legacy duration / 输出旧时长."""
        package = MinimalPackage(TransferChiplet(.99, 1, 2, 10, 20))
        circuit = make_circuit(package, {0: (2, 10)}, model="legacy")
        stage = riding_stage(circuit)
        self.assertTrue(stage.add_transport_action_immediate(Transport(0, (2, 10), (2, 15))))
        self.assertTrue(stage.add_transport_action_immediate(Transport(0, (2, 15), (2, 19))))
        with patch("qunoc.hamiltonian.estimate_transfer", side_effect=AssertionError("legacy must not call planner")):
            self.assertEqual(circuit.estimate_duration(package), 125)
            for name in tuple(vars(circuit.op_config)):
                if name.startswith("eqst_"):
                    delattr(circuit.op_config, name)
            self.assertEqual(pickle.loads(pickle.dumps(circuit)).estimate_duration(package), 125)

    def test_dummy_circuit_stays_legacy_even_with_hamiltonian_config(self):
        """Dummy baseline keeps legacy timing / Dummy 基线维持原线性计时；output unchanged baseline / 输出不变基线."""
        config = OperationConfig(eqst_model="hamiltonian", intra_time=25)
        dummy = DummyCircuit(1, (1, 6), {0: (0, 0)}, {(0, i): 0 if i == 0 else -1 for i in range(6)}, config)
        action = Transport(0, (0, 0), (0, 5))
        with patch("qunoc.hamiltonian.estimate_transfer", side_effect=AssertionError("dummy must not call planner")):
            self.assertEqual(dummy.get_action_duration(action), 5 * config.qst_time_constant)


if __name__ == "__main__":
    unittest.main()
