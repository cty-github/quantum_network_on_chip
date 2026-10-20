"""Regression tests / 回归测试：距离校准、自动路由、相干演化、公共接口和输出。

Run: PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s evaluate -p 'test_chain_evolution.py' -v
Tests write no result files / 测试不生成结果文件。
"""

import contextlib
import csv
import io
import json
import math
import random
import subprocess
import sys
import unittest
from dataclasses import FrozenInstanceError, replace
from itertools import permutations
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.sparse.linalg import expm_multiply

# 支持从仓库外直接运行测试脚本。 Support direct invocation from outside the repository.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qunoc.hamiltonian import chain_evolution as ce
from evaluate import run_hamiltonian_simulation as wrapper


NONCONFLICTING = ((0, 9), (5, 19), (2, 13))
CONFLICTING = ((0, 9), (9, 10), (5, 19))
EXPECTED_STAGES = (((5, 19),), ((0, 9), (10, 15)))


class RoutingAndCalibrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """功能 / Purpose: Prepare reusable calibrated evolutions / 准备可复用校准演化。

        输入 / Inputs: cls test class / cls 测试类。
        输出 / Outputs: None; stores two capped and one uncapped example / 无返回；保存限幅与不限幅示例。
        """
        cls.direct_config = ce.SimulationConfig(20, NONCONFLICTING, num_steps=101)
        cls.conflict_config = ce.SimulationConfig(20, CONFLICTING, num_steps=101)
        cls.direct = ce.run_simulation(cls.direct_config)
        cls.conflict = ce.run_simulation(cls.conflict_config)
        cls.uncapped = ce.run_simulation(replace(cls.conflict_config, g_max=None))

    def test_disjoint_endpoints_not_geometric_intervals_select_direct_xy(self):
        """功能 / Purpose: Verify endpoint-only direct routing / 验证按端点而非区间选择直接路由。

        输入 / Inputs: self test fixture / self 测试夹具。
        输出 / Outputs: None; checks route, basis dimension and completion / 无返回；核对路线、维数与完成时间。
        """
        result = self.direct[0]
        self.assertEqual(result.route, "direct-xy")
        self.assertEqual(result.route_details["direct_pairs"], NONCONFLICTING)
        self.assertEqual(result.hilbert_dim, math.comb(20, 3))
        self.assertAlmostEqual(result.final_time, 125 * math.sqrt(14 / 5))
        self.assertAlmostEqual(result.final_fidelity, 1.0, places=11)
        self.assertAlmostEqual(result.calibrated_final_fidelity, 0.992 ** (34 / 5), places=11)

    def test_direct_full_curve_and_independent_shutoff(self):
        """功能 / Purpose: Check unequal direct transfer times and shutoff / 检查不同直接传输用时及关断。

        输入 / Inputs: self test fixture / self 测试夹具。
        输出 / Outputs: None; compares full curve with independent analytic probabilities / 无返回；与独立解析概率比较。
        """
        result, times, _, _, ideal = self.direct
        durations = 125 * np.sqrt(np.array([9, 14, 11]) / 5)
        expected = np.prod(np.sin(math.pi / 2 * np.minimum(times[:, None] / durations, 1)) ** 2, axis=1)
        np.testing.assert_allclose(ideal, expected, atol=1e-11)
        primitives = result.route_details["primitives"]
        np.testing.assert_allclose([p["duration_ns"] for p in primitives], durations)
        self.assertTrue(all(not p["cap_limited"] for p in primitives))
        self.assertEqual(result.route_details["stage_count"], 1)
        self.assertEqual(result.route_details["hamiltonian_interval_count"], 3)
        for duration in durations:
            self.assertEqual(np.count_nonzero(np.isclose(times, duration)), 1)
        exposure = ce.distance_exposure(times, primitives)
        np.testing.assert_allclose(exposure, np.minimum(times[:, None] / durations, 1) @ [9, 14, 11])

    def test_any_conflict_routes_the_entire_task_through_pst(self):
        """功能 / Purpose: Check whole-task conflict routing / 检查整个冲突任务转为 PST。

        输入 / Inputs: self test fixture / self 测试夹具。
        输出 / Outputs: None; checks route and all intermediate occupations / 无返回；核对路线与中间占据。
        """
        result = self.conflict[0]
        self.assertEqual(result.route, "pst-segments")
        self.assertEqual(result.route_details["direct_pairs"], ())
        self.assertEqual(result.route_details["segment_stages"], EXPECTED_STAGES)
        self.assertEqual(result.route_details["occupation_path"], [(0, 5, 9), (0, 15, 19), (9, 10, 19)])
        self.assertEqual(result.route_details["stage_count"], 2)
        self.assertEqual(result.route_details["hamiltonian_interval_count"], 3)

    def test_capped_conflict_times_couplings_and_full_curve(self):
        """功能 / Purpose: Check bounded PST times, profiles and ideal curve / 检查限幅 PST 用时、耦合与理想曲线。

        输入 / Inputs: self test fixture / self 测试夹具。
        输出 / Outputs: None; verifies 520.138 ns completion and independent segment shutoff / 无返回；核对完成时间及独立关断。
        """
        result, times, _, _, ideal = self.conflict
        first, second = 125 * math.sqrt(56) / 3, 125 * 5 / 3
        np.testing.assert_allclose(result.route_details["stage_durations"], [first, second])
        primitives = result.route_details["primitives"]
        np.testing.assert_allclose([p["duration_ns"] for p in primitives], [first, second, 125])
        np.testing.assert_allclose([p["max_coupling_rad_per_ns"] for p in primitives], [result.config.g_max] * 3)
        self.assertEqual([p["cap_limited"] for p in primitives], [True, True, False])
        local = np.maximum(times - first, 0)
        expected = np.sin(math.pi / 2 * np.minimum(local / second, 1)) ** 18
        expected *= np.sin(math.pi / 2 * np.minimum(local / 125, 1)) ** 10
        np.testing.assert_allclose(ideal, expected, atol=2e-11)
        self.assertAlmostEqual(result.final_time, 520.1381155644951, places=10)
        self.assertAlmostEqual(result.calibrated_final_fidelity, 0.9468460476024284, places=10)

    def test_no_cap_satisfies_requested_sqrt_distance_times(self):
        """功能 / Purpose: Check strict requested timing without cap / 检查不限幅时的距离平方根用时。

        输入 / Inputs: self test fixture / self 测试夹具。
        输出 / Outputs: None; cap changes time but not total distance attenuation / 无返回；限幅只改变时间不改变总距离衰减。
        """
        result, times, _, _, ideal = self.uncapped
        first, second = 125 * math.sqrt(14 / 5), 125 * math.sqrt(9 / 5)
        for primitive in result.route_details["primitives"]:
            self.assertAlmostEqual(primitive["duration_ns"], primitive["requested_duration_ns"])
            self.assertFalse(primitive["cap_limited"])
        local = np.maximum(times - first, 0)
        expected = np.sin(math.pi / 2 * np.minimum(local / second, 1)) ** 18
        expected *= np.sin(math.pi / 2 * np.minimum(local / 125, 1)) ** 10
        np.testing.assert_allclose(ideal, expected, atol=2e-11)
        self.assertAlmostEqual(result.final_time, 376.8701049460031)
        self.assertAlmostEqual(result.calibrated_final_fidelity, self.conflict[0].calibrated_final_fidelity, places=10)

    def test_distance_cost_counts_internal_mirrors_and_detours(self):
        """功能 / Purpose: Count actual internal and detour displacements / 统计内部镜像与绕路的实际位移。

        输入 / Inputs: self test fixture / self 测试夹具。
        输出 / Outputs: None; rejects endpoint-span-only or request-only distance accounting / 无返回；核对非端点跨度或原请求距离算法。
        """
        details = self.conflict[0].route_details
        self.assertEqual([p["transport_distance"] for p in details["primitives"]], [20, 9, 5])
        self.assertEqual(details["total_transport_distance"], 34)
        # 先让0往返一次，再执行原路线，额外距离为2。 An actual round trip adds two edges.
        detour = (((0, 1),), ((0, 1),)) + EXPECTED_STAGES
        result = ce.run_simulation(replace(self.conflict_config, segments=detour, num_steps=2))[0]
        self.assertEqual(result.route_details["total_transport_distance"], 36)
        self.assertAlmostEqual(result.calibrated_final_fidelity, 0.992 ** (36 / 5), places=10)
        self.assertLess(result.calibrated_final_fidelity, self.conflict[0].calibrated_final_fidelity)

    def test_calibrated_curve_bounds_and_separate_sampled_peaks(self):
        """功能 / Purpose: Separate ideal amplitudes, attenuation and peaks / 区分理想振幅、距离衰减和峰值。

        输入 / Inputs: self test fixture / self 测试夹具。
        输出 / Outputs: None; checks complete calibrated arrays and sampled maxima / 无返回；核对完整校准曲线及采样最大值。
        """
        for result, times, target, _, ideal in (self.direct, self.conflict):
            exposure = ce.distance_exposure(times, result.route_details["primitives"])
            calibrated = ce.calibrated_target_fidelities(result.config, exposure, ideal)
            np.testing.assert_allclose(np.abs(target) ** 2, ideal, atol=1e-15)
            np.testing.assert_allclose(calibrated, ideal * 0.992 ** (exposure / 5), atol=1e-14)
            self.assertTrue(np.all(np.diff(exposure) >= -1e-13))
            self.assertTrue(np.all(calibrated >= 0))
            self.assertTrue(np.all(calibrated <= ideal))
            self.assertLessEqual(float(np.max(ideal)), 1 + 1e-10)
            self.assertEqual(result.peak_time, times[int(np.argmax(ideal))])
            self.assertEqual(result.calibrated_peak_time, times[int(np.argmax(calibrated))])
            self.assertEqual(result.calibrated_peak_fidelity, float(np.max(calibrated)))
            self.assertEqual(result.calibrated_final_fidelity, float(calibrated[-1]))
        self.assertLess(self.direct[0].calibrated_peak_time, self.direct[0].final_time)

    def test_stationary_excitation_has_no_idle_distance_penalty(self):
        """功能 / Purpose: Keep stationary excitations without idle penalty / 保留静止激发且不添加静止惩罚。

        输入 / Inputs: self test fixture / self 测试夹具。
        输出 / Outputs: None; only the moving excitation contributes distance / 无返回；仅移动激发贡献距离。
        """
        result, times, _, _, ideal = ce.run_simulation(ce.SimulationConfig(4, ((0, 3), (1, 1)), num_steps=31))
        self.assertEqual(result.route_details["total_transport_distance"], 3)
        self.assertEqual(result.route_details["direct_pairs"], ((0, 3),))
        np.testing.assert_allclose(ideal, np.sin(math.pi / 2 * times / result.final_time) ** 2, atol=1e-12)
        self.assertAlmostEqual(result.calibrated_final_fidelity, 0.992 ** (3 / 5), places=12)

    def test_center_excitation_has_zero_mirror_displacement(self):
        """功能 / Purpose: Count a mirror-center excitation correctly / 正确统计镜像中心激发。

        输入 / Inputs: self test fixture / self 测试夹具。
        输出 / Outputs: None; center contributes zero even during nontrivial evolution / 无返回；非平凡演化时中心位移仍为零。
        """
        result = ce.run_simulation(ce.SimulationConfig(5, ((0, 4), (4, 0), (2, 2)),
                                                    segments=(((0, 4),),), num_steps=2))[0]
        self.assertEqual(result.route_details["total_transport_distance"], 8)
        self.assertAlmostEqual(result.calibrated_final_fidelity, 0.992 ** (8 / 5), places=11)

    def test_identity_zero_time_zero_distance(self):
        """功能 / Purpose: Preserve zero-time identity / 保留零时刻恒等任务。

        输入 / Inputs: self test fixture / self 测试夹具。
        输出 / Outputs: None; checks stationary and equal-occupation identities / 无返回；核对静止及相同占据任务。
        """
        for pairs in (((0, 0), (2, 2)), ((0, 1), (1, 0))):
            result, times, _, _, ideal = ce.run_simulation(ce.SimulationConfig(3, pairs))
            np.testing.assert_array_equal(times, [0])
            np.testing.assert_array_equal(ideal, [1])
            self.assertEqual(result.route_details["total_transport_distance"], 0)
            self.assertEqual(result.route_details["hamiltonian_interval_count"], 0)
            self.assertEqual(result.calibrated_final_fidelity, 1)
            self.assertEqual(result.calibrated_peak_time, 0)

    def test_ideal_flag_and_reference_fidelity_one_disable_attenuation(self):
        """功能 / Purpose: Disable attenuation without changing dynamics / 关闭衰减而不改变动力学。

        输入 / Inputs: self test fixture / self 测试夹具。
        输出 / Outputs: None; both controls preserve times and ideal arrays / 无返回；两种控制均保留时间及理想数组。
        """
        base = ce.SimulationConfig(4, ((0, 3),), num_steps=11)
        lossy = ce.run_simulation(base)
        for config in (replace(base, ideal=True), replace(base, reference_fidelity=1.0)):
            actual = ce.run_simulation(config)
            self.assertEqual(config.attenuation_per_edge, 0)
            for a, b in zip(actual[1:], lossy[1:], strict=True):
                np.testing.assert_array_equal(a, b)
            self.assertEqual(actual[0].calibrated_final_fidelity, actual[0].final_fidelity)

    def test_custom_reference_and_cap_change_time_not_distance_fidelity(self):
        """功能 / Purpose: Respect independent reference and coupling controls / 检查独立参考参数与耦合控制。

        输入 / Inputs: self test fixture / self 测试夹具。
        输出 / Outputs: None; verifies a slower capped transfer with unchanged distance cost / 无返回；核对较慢传输的距离成本不变。
        """
        config = ce.SimulationConfig(3, ((0, 2),), reference_time_ns=100, reference_distance=2,
                                     reference_fidelity=0.9, g_max=math.pi / 400, num_steps=11)
        result = ce.run_simulation(config)[0]
        self.assertAlmostEqual(result.final_time, 200)
        self.assertAlmostEqual(result.calibrated_final_fidelity, 0.9, places=12)
        self.assertEqual(result.route_details["primitives"][0]["requested_duration_ns"], 100)
        fast = ce.run_simulation(replace(config, g_max=None))[0]
        self.assertAlmostEqual(fast.final_time, 100)
        self.assertAlmostEqual(fast.calibrated_final_fidelity, result.calibrated_final_fidelity, places=12)

    def test_sampling_includes_unique_events_and_is_not_integration(self):
        """功能 / Purpose: Check event sampling and endpoint invariance / 检查事件采样与末态不变性。

        输入 / Inputs: self test fixture / self 测试夹具。
        输出 / Outputs: None; each constant-H interval contributes num_steps-1 new samples / 无返回；核对每区间新增采样数。
        """
        for result, times, *_ in (self.direct, self.conflict):
            self.assertEqual(len(times), 1 + 3 * (101 - 1))
            self.assertEqual(result.sample_count, len(times))
            self.assertTrue(np.all(np.diff(times) > 0))
            for boundary in result.route_details["interval_end_times"]:
                self.assertEqual(np.count_nonzero(np.isclose(times, boundary)), 1)
            sparse = ce.run_simulation(replace(result.config, num_steps=2))[0]
            self.assertEqual(sparse.sample_count, 4)
            self.assertAlmostEqual(sparse.final_time, result.final_time)
            self.assertAlmostEqual(sparse.calibrated_final_fidelity, result.calibrated_final_fidelity, places=10)

    def test_manual_route_matches_automatic_route(self):
        """功能 / Purpose: Compare automatic and manual route evolution / 比较自动与手动路线演化。

        输入 / Inputs: self test fixture / self 测试夹具。
        输出 / Outputs: None; complete arrays match / 无返回；完整数组匹配。
        """
        manual = ce.run_simulation(replace(self.conflict_config, segments=EXPECTED_STAGES))
        self.assertTrue(manual[0].route_details["manual_segments"])
        for actual, expected in zip(manual[1:], self.conflict[1:], strict=True):
            np.testing.assert_allclose(actual, expected, atol=1e-12)


class HamiltonianTests(unittest.TestCase):
    def test_true_six_site_nearest_neighbor_reference(self):
        """功能 / Purpose: Recover the true six-site PST reference / 恢复真实六站点最近邻 PST 参考点。

        输入 / Inputs: self test fixture / self 测试夹具。
        输出 / Outputs: None; checks J profile, 125 ns ideal PST and calibrated 0.992 / 无返回；核对耦合、时间及概率。
        """
        config = ce.SimulationConfig(6, ((0, 5),))
        basis, index = ce.build_fixed_excitation_basis(6, 1)
        h, duration, g, maximum = ce.build_pst_segment_stage_hamiltonian(
            basis, index, ((0, 5),), math.pi / 250, config.g_max)
        self.assertAlmostEqual(duration, 125)
        self.assertAlmostEqual(maximum, 3 * g)
        self.assertEqual(h[0, 5], 0)  # 不用单 pair CLI 的直接长边。 Not a CLI direct long edge.
        for left, right in zip(*h.nonzero(), strict=True):
            self.assertEqual(abs(left - right), 1)
        np.testing.assert_allclose(h.diagonal(1).real, g * np.sqrt([5, 8, 9, 8, 5]))
        times, target, _, ideal = ce.simulate_schedule_time_grid(6, 0, 5, [h], [duration], 51)
        np.testing.assert_allclose(ideal, np.sin(g * times) ** 10, atol=2e-12)
        self.assertAlmostEqual(abs(target[-1]) ** 2, 1, places=12)
        self.assertAlmostEqual(ce.calibrated_target_fidelities(config, 5 * times / duration, ideal)[-1], 0.992, places=12)

    def test_normalized_low_level_pst_and_uncapped_stage(self):
        """功能 / Purpose: Preserve normalized mathematical PST tests / 保留归一化数学 PST 测试。

        输入 / Inputs: self test fixture / self 测试夹具。
        输出 / Outputs: None; explicit g=1 and no cap yield pi/2 PST / 无返回；显式 g=1 不限幅得到 pi/2 PST。
        """
        basis, index = ce.build_fixed_excitation_basis(6, 1)
        h, duration, g, maximum = ce.build_pst_segment_stage_hamiltonian(basis, index, ((0, 5),), 1.0, None)
        self.assertEqual(g, 1)
        self.assertEqual(maximum, 3)
        self.assertEqual(duration, math.pi / 2)
        times, _, _, ideal = ce.simulate_schedule_time_grid(6, 0, 5, [h], [duration], 21)
        np.testing.assert_allclose(ideal, np.sin(times) ** 10, atol=1e-12)

    def test_hard_core_hermiticity_and_commuting_disjoint_terms(self):
        """功能 / Purpose: Check hard-core Hermitian disjoint hopping / 检查硬核厄米不相交跃迁。

        输入 / Inputs: self test fixture / self 测试夹具。
        输出 / Outputs: None; checks matrix sum, commutators and no double occupancy / 无返回；核对求和、对易性及无双占据。
        """
        basis, index = ce.build_fixed_excitation_basis(20, 3)
        terms = [ce.build_edge_hamiltonian(basis, index, ((s, t, 1.0),)) for s, t in NONCONFLICTING]
        total = ce.build_edge_hamiltonian(basis, index, [(s, t, 1.0) for s, t in NONCONFLICTING])
        self.assertEqual((total - sum(terms)).nnz, 0)
        self.assertEqual((total - total.getH()).nnz, 0)
        for i, first in enumerate(terms):
            for second in terms[i + 1:]:
                self.assertEqual((first @ second - second @ first).nnz, 0)
        self.assertTrue(all(len(state) == len(set(state)) for state in basis))

    def test_overlapping_terms_do_not_commute_and_complex_hops_are_unitary(self):
        """功能 / Purpose: Check conflict noncommutation and complex unitarity / 检查冲突非对易性及复耦合幺正性。

        输入 / Inputs: self test fixture / self 测试夹具。
        输出 / Outputs: None; checks nonzero commutator and unit norm / 无返回；核对非零对易子与单位范数。
        """
        basis, index = ce.build_fixed_excitation_basis(3, 1)
        first = ce.build_edge_hamiltonian(basis, index, ((0, 1, 0.5 + 0.3j),))
        second = ce.build_edge_hamiltonian(basis, index, ((1, 2, 1),))
        self.assertGreater((first @ second - second @ first).nnz, 0)
        self.assertEqual((first - first.getH()).nnz, 0)
        state = np.array([1, 0, 0], dtype=complex)
        evolved = expm_multiply(-1j * (first + second), state)
        self.assertAlmostEqual(float(np.vdot(evolved, evolved).real), 1, places=12)

    def test_segments_isolate_internal_edges_and_mirror_interior(self):
        """功能 / Purpose: Check nearest-neighbor isolation and interior mirrors / 检查最近邻隔离与内部镜像。

        输入 / Inputs: self test fixture / self 测试夹具。
        输出 / Outputs: None; no inter-segment hopping and all internal occupations mirror / 无返回；段间无跃迁且内部占据镜像。
        """
        basis, index = ce.build_fixed_excitation_basis(20, 1)
        h, _, g, maximum = ce.build_pst_segment_stage_hamiltonian(basis, index, EXPECTED_STAGES[1], 1.0, 1.0)
        self.assertAlmostEqual(g, 0.2)
        self.assertAlmostEqual(maximum, 1.0)
        self.assertEqual(h[9, 10], 0)
        self.assertEqual(h[15, 16], 0)
        for left, right in zip(*h.nonzero(), strict=True):
            self.assertEqual(abs(left - right), 1)
        self.assertEqual(ce.apply_segment_stages((0, 5, 9), (((5, 19),),)), (0, 15, 19))
        self.assertEqual(ce.apply_segment_stages((0, 5, 9), EXPECTED_STAGES), (9, 10, 19))

    def test_large_cap_never_speeds_beyond_requested_scale(self):
        """功能 / Purpose: Check cap is only an upper bound / 检查耦合上限不会主动加速。

        输入 / Inputs: self test fixture / self 测试夹具。
        输出 / Outputs: None; uncapped and generous-cap scales agree / 无返回；不限幅与宽松限幅尺度一致。
        """
        self.assertEqual(ce.stage_scale(5, 0.2, 100), ce.stage_scale(5, 0.2, None))
        for value in (1e-320, 1e308):
            with self.assertRaises(ValueError):
                ce.stage_scale(10, value, None)


class MatrixFreePlanningTests(unittest.TestCase):
    def test_estimator_matches_numeric_examples_and_shared_plan(self):
        """功能 / Purpose: Compare shared matrix-free and numerical schedules / 比较共享无矩阵与数值调度。

        输入 / Inputs: Two calibrated examples, capped and uncapped / 两组校准示例，限幅与不限幅。
        输出 / Outputs: Equal times/distances and direct log-envelope agreement / 时间距离一致且对数包络吻合。
        """
        for pairs in (NONCONFLICTING, CONFLICTING):
            sources, targets = zip(*pairs, strict=True)
            for cap in (ce.DEFAULT_G_MAX_RAD_PER_NS, None):
                estimate = ce.estimate_transfer(sources, targets, 20, g_max=cap)
                config = ce.SimulationConfig(20, pairs, g_max=cap, num_steps=2, ordered=True)
                result = ce.run_simulation(config)[0]
                self.assertIs(estimate["plan"], ce.plan_transfer(config))
                self.assertAlmostEqual(estimate["transfer_time_ns"], result.final_time)
                self.assertEqual(estimate["effective_distance_edges"], 34)
                self.assertEqual(estimate["effective_distance_edges"], result.route_details["total_transport_distance"])
                self.assertAlmostEqual(math.exp(estimate["log_fidelity"]), result.calibrated_final_fidelity, places=10)
                self.assertAlmostEqual(estimate["neglog_fidelity"], -estimate["log_fidelity"])
        self.assertEqual(sum(abs(s - t) for s, t in CONFLICTING), 24)

    def test_reciprocal_cycle_and_full_occupancy_are_not_identity(self):
        """功能 / Purpose: Separate paired permutations from occupation identity / 区分逐对排列与占据恒等。

        输入 / Inputs: Reciprocal, cyclic and fully occupied permutations / 互换、循环和全占据排列。
        输出 / Outputs: Positive costs and exact paired final positions / 正成本及精确逐对最终位置。
        """
        for sources, targets in (((0, 1), (1, 0)), ((0, 1, 2), (1, 2, 0)),
                                 (tuple(range(6)), tuple(reversed(range(6))))):
            estimate = ce.estimate_transfer(sources, targets)
            plan = estimate["plan"]
            self.assertGreater(estimate["transfer_time_ns"], 0)
            self.assertGreater(estimate["effective_distance_edges"], 0)
            self.assertEqual(plan.labeled_path[-1], targets)
            legacy = ce.estimate_transfer(sources, targets, ordered=False)
            self.assertEqual(legacy["transfer_time_ns"], 0)
            self.assertEqual(legacy["effective_distance_edges"], 0)
            numeric = ce.run_simulation(ce.SimulationConfig(max(sources) + 1,
                                        tuple(zip(sources, targets)), ordered=True, num_steps=2))[0]
            self.assertAlmostEqual(numeric.final_time, estimate["transfer_time_ns"])
            self.assertAlmostEqual(numeric.final_fidelity, 1, places=11)
        self.assertEqual(wrapper.simulate_transfer([0, 1], [1, 0]),
                         {"transfer_time_ns": 0.0, "fidelity": 1.0})

    def test_holds_restore_disturbed_residents_and_charge_every_displacement(self):
        """功能 / Purpose: Restore and charge disturbed stationary residents / 恢复并计费被扰动的静止居民。

        输入 / Inputs: Endpoint swap with an interior i:i spectator / 带内部 i:i 居民的端点对换。
        输出 / Outputs: Hold restored, internal moves charged, untouched hold free / 静止目标恢复、内部移动计费且未扰动居民免费。
        """
        estimate = ce.estimate_transfer([0, 3, 1], [3, 0, 1], 4)
        plan = estimate["plan"]
        self.assertEqual(plan.labeled_path[-1], (3, 0, 1))
        self.assertTrue(any(positions[2] != 1 for positions in plan.labeled_path))
        self.assertEqual(estimate["effective_distance_edges"], 8)
        for index, stage in enumerate(plan.segment_stages):
            expected = sum(abs(left + right - 2 * pos) for left, right in stage
                           for pos in plan.labeled_path[index] if left <= pos <= right)
            cost = sum(primitive.transport_distance for primitive in plan.primitives
                       if primitive.stage_index == index)
            self.assertEqual(cost, expected)
        untouched = ce.estimate_transfer([0, 1], [3, 1], 4)
        self.assertEqual(untouched["effective_distance_edges"], 3)
        self.assertEqual(untouched["plan"].direct_pairs, ((0, 3),))
        identity = ce.estimate_transfer([1, 3], [1, 3], 4)
        self.assertEqual(identity["transfer_time_ns"], 0)
        self.assertEqual(identity["effective_distance_edges"], 0)
        self.assertEqual(identity["log_fidelity"], 0)

    def test_manual_ordered_route_cannot_stop_at_equal_occupation(self):
        """功能 / Purpose: Reject occupation-only success for ordered manual routes / 拒绝有序手动路线仅占据成功。

        输入 / Inputs: Full occupancy swap with one wrong or two restoring mirrors / 全占据对换，错误单镜像或恢复双镜像。
        输出 / Outputs: Wrong labels fail; restored hold succeeds / 错误标签被拒绝，恢复静止居民后通过。
        """
        pairs = ((0, 3), (3, 0), (1, 1), (2, 2))
        wrong = ce.SimulationConfig(4, pairs, segments=(((0, 3),),), ordered=True)
        with self.assertRaisesRegex(ValueError, "paired targets"):
            ce.plan_transfer(wrong)
        fixed = ce.plan_transfer(replace(wrong, segments=(((0, 3),), ((1, 2),))))
        self.assertEqual(fixed.labeled_path[-1], (3, 0, 1, 2))
        self.assertEqual(fixed.effective_distance_edges, 10)
        self.assertEqual(ce.plan_transfer(replace(wrong, ordered=False)).effective_distance_edges, 8)

    def test_exhausted_budget_uses_deterministic_adjacent_fallback(self):
        """功能 / Purpose: Complete valid goals after heuristic budget exhaustion / 启发式预算耗尽后完成合法目标。

        输入 / Inputs: Conflicting example with one stage and one beam state / 冲突示例，仅一个阶段和一个束状态预算。
        输出 / Outputs: Deterministic adjacent mirrors, valid labels and numerical agreement / 确定相邻镜像、合法标签及数值一致。
        """
        sources, targets = zip(*CONFLICTING, strict=True)
        estimate = ce.estimate_transfer(sources, targets, 20, max_segment_steps=1, beam_width=1)
        plan = estimate["plan"]
        self.assertEqual(plan.planner, "adjacent-fallback")
        self.assertGreater(len(plan.segment_stages), 1)
        self.assertTrue(all(right == left + 1 for stage in plan.segment_stages for left, right in stage))
        self.assertEqual(plan.segment_stages, ce.adjacent_permutation_stages(sources, targets))
        self.assertEqual(plan.labeled_path[-1], targets)
        ce.validate_segment_stages(20, plan.segment_stages)
        result = ce.run_simulation(plan.config)[0]
        self.assertAlmostEqual(result.final_time, estimate["transfer_time_ns"])
        self.assertAlmostEqual(result.calibrated_final_fidelity, math.exp(estimate["log_fidelity"]), places=10)

    def test_adjacent_permutation_invariants_exhaustive_and_with_holes(self):
        """功能 / Purpose: Check permutation fallback independently of occupation goals / 独立于占据目标检验排列兜底。

        输入 / Inputs: All four-site permutations and randomized sparse mappings / 全部四站点排列及随机稀疏映射。
        输出 / Outputs: Unique in-bounds states and exact labels after disjoint adjacent stages / 不相交相邻阶段后位置唯一合法且标签精确。
        """
        cases = [(tuple(range(4)), target, 4) for target in permutations(range(4))]
        rng = random.Random(137)
        cases += [(tuple(rng.sample(range(12), 7)), tuple(rng.sample(range(12), 7)), 12)
                  for _ in range(50)]
        for sources, targets, length in cases:
            stages = ce.adjacent_permutation_stages(sources, targets)
            if stages:
                ce.validate_segment_stages(length, stages)
            state = sources
            for stage in stages:
                state = ce.apply_segment_stage_labeled(state, stage)
                self.assertEqual(len(set(state)), len(sources))
                self.assertTrue(all(0 <= position < length for position in state))
            self.assertEqual(state, targets)
            estimate = ce.estimate_transfer(sources, targets, length, max_segment_steps=2, beam_width=4)
            path = estimate["plan"].labeled_path
            self.assertEqual(path[-1], targets)
            actual = sum(sum(abs(before - after) for before, after in zip(first, second))
                         for first, second in zip(path, path[1:]))
            self.assertEqual(actual, estimate["effective_distance_edges"])
            self.assertGreaterEqual(actual, sum(abs(source - target) for source, target in zip(sources, targets)))
            state = sources
            for stage in ce.sparse_permutation_stages(sources, targets):
                state = ce.apply_segment_stage_labeled(state, stage)
            self.assertEqual(state, targets)

    def test_large_dense_and_sparse_estimates_never_build_basis_matrices_or_profiles(self):
        """功能 / Purpose: Keep large estimates outside combinatorial Hilbert space / 大规模估计不进入组合 Hilbert 空间。

        输入 / Inputs: N=80,k=40 and full-occupancy/sparse extreme requests / N=80,k=40、全占据及极稀疏请求。
        输出 / Outputs: Valid exact plans despite forbidden basis/matrix/profile/beam construction / 禁用基底矩阵剖面及束搜索仍得到合法精确方案。
        """
        rng = random.Random(19)
        sources = tuple(range(40))
        targets = tuple(rng.sample(range(80), 40))
        cases = [(sources, targets, 80), (tuple(range(80)), tuple(reversed(range(80))), 80),
                 ((0, 10, 50000000), (50000000, 10, 0), 1000000000)]
        ce._cached_transfer_plan.cache_clear()
        with patch.object(ce, "build_fixed_excitation_basis", side_effect=AssertionError("basis")), \
                patch.object(ce, "build_edge_hamiltonian", side_effect=AssertionError("matrix")), \
                patch.object(ce, "engineered_pst_couplings", side_effect=AssertionError("profile")), \
                patch.object(ce, "labeled_stage_candidates", side_effect=AssertionError("beam")):
            for sources, targets, length in cases:
                estimate = ce.estimate_transfer(sources, targets, length, max_segment_steps=1)
                self.assertEqual(estimate["plan"].labeled_path[-1], targets)
                self.assertTrue(math.isfinite(estimate["transfer_time_ns"]))
                self.assertGreater(estimate["effective_distance_edges"], 0)
        self.assertGreater(math.comb(80, 40), 10**20)
        self.assertEqual(estimate["plan"].planner, "sparse-mirror-fallback")
        self.assertLessEqual(len(estimate["plan"].segment_stages), 2 * len(sources))
        self.assertGreater(estimate["effective_distance_edges"], sum(abs(s-t) for s,t in zip(sources, targets)))

    def test_log_fidelity_remains_finite_when_probability_underflows(self):
        """功能 / Purpose: Preserve stable log costs under probability underflow / 概率下溢时保留稳定对数成本。

        输入 / Inputs: Million-edge direct transfer and reference fidelity one / 百万边直接传输及完美参考概率。
        输出 / Outputs: Finite log/error, real exponential underflow and no 0*inf NaN / 对数误差有限，允许指数下溢且不产生 NaN。
        """
        estimate = ce.estimate_transfer([0], [1000000], 1000001)
        self.assertEqual(estimate["effective_distance_edges"], 1000000)
        self.assertEqual(math.exp(estimate["log_fidelity"]), 0)
        self.assertTrue(math.isfinite(estimate["log_fidelity"]))
        self.assertAlmostEqual(estimate["log_fidelity"], math.log(.992) * 1000000 / 5)
        self.assertGreater(math.exp(estimate["log_fidelity"] / 1000000), .99)
        perfect = ce.estimate_transfer([0], [1000000], 1000001, reference_fidelity=1)
        self.assertEqual(perfect["log_fidelity"], 0)
        identity = ce.estimate_transfer([0], [0], reference_distance=5e-324, reference_fidelity=.5)
        self.assertEqual(identity["log_fidelity"], 0)

    def test_cache_is_bounded_content_based_and_never_exposes_mutable_results(self):
        """功能 / Purpose: Protect cached plans from caller mutation / 保护缓存方案不被调用方修改。

        输入 / Inputs: Equal list/tuple requests, mutations and more than maxsize keys / 同内容列表元组请求、修改及超容量键。
        输出 / Outputs: Shared frozen plan, independent metadata and bounded eviction / 共享冻结方案、独立元数据及有界淘汰。
        """
        ce._cached_transfer_plan.cache_clear()
        first = ce.estimate_transfer([0, 1], [2, 1], 3)
        second = ce.estimate_transfer((0, 1), (2, 1), 3)
        self.assertIs(first["plan"], second["plan"])
        self.assertIsNot(first, second)
        first["transfer_time_ns"] = -1
        self.assertGreater(second["transfer_time_ns"], 0)
        with self.assertRaises(FrozenInstanceError):
            second["plan"].transfer_time_ns = -1
        with self.assertRaises(FrozenInstanceError):
            second["plan"].primitives[0].transport_distance = -1
        details = second["plan"].to_route_details()
        details["primitives"][0]["transport_distance"] = -1
        self.assertEqual(second["plan"].to_route_details()["primitives"][0]["transport_distance"], 2)
        pairs = [[0, 2], [1, 1]]
        frozen = ce.plan_transfer(ce.SimulationConfig(3, pairs, ordered=True))
        pairs[0][1] = 0
        self.assertEqual(frozen.config.pairs, ((0, 2), (1, 1)))
        for offset in range(ce.PLAN_CACHE_MAXSIZE + 1):
            ce.estimate_transfer([0], [1], 2, reference_time_ns=125 + offset)
        info = ce._cached_transfer_plan.cache_info()
        self.assertEqual(info.maxsize, ce.PLAN_CACHE_MAXSIZE)
        self.assertEqual(info.currsize, ce.PLAN_CACHE_MAXSIZE)

    def test_estimator_rejects_bad_requests_budgets_and_ordering_flags(self):
        """功能 / Purpose: Validate estimator inputs before cached planning / 缓存规划前校验估计输入。

        输入 / Inputs: Invalid positions, calibration, budgets and ordered flags / 非法位置、校准、预算及有序标志。
        输出 / Outputs: Explicit ValueError for each malformed request / 每个非法请求明确抛出 ValueError。
        """
        for sources, targets in (([], []), ([0], [1, 2]), ([False], [1]), ([0.5], [1]),
                                 ([0, 0], [1, 2]), ([0, 1], [2, 2]), ([-1], [2])):
            with self.subTest(sources=sources, targets=targets), self.assertRaises(ValueError):
                ce.estimate_transfer(sources, targets)
        for controls in ({"ordered": 1}, {"beam_width": .5}, {"beam_width": False},
                         {"max_segment_steps": 0}, {"max_segment_steps": 1.2},
                         {"reference_fidelity": 0}, {"chain_length": True}):
            with self.subTest(controls=controls), self.assertRaises(ValueError):
                ce.estimate_transfer([0], [1], **controls)


class InterfaceAndOutputTests(unittest.TestCase):
    def test_public_interface_matches_both_examples_and_returns_only_two_scalars(self):
        """功能 / Purpose: Check the sole external transfer interface / 检查唯一外部传输接口。

        输入 / Inputs: self test fixture / self 测试夹具。
        输出 / Outputs: None; public time/fidelity equals full core completion results / 无返回；公共时间概率与完整演化一致。
        """
        self.assertEqual(wrapper.__all__, ["simulate_transfer"])
        for pairs in (NONCONFLICTING, CONFLICTING):
            sources, targets = zip(*pairs, strict=True)
            public = wrapper.simulate_transfer(sources, targets, chain_length=20)
            core = ce.run_simulation(ce.SimulationConfig(20, pairs, num_steps=5))[0]
            self.assertEqual(set(public), {"transfer_time_ns", "fidelity"})
            self.assertAlmostEqual(public["transfer_time_ns"], core.final_time)
            self.assertAlmostEqual(public["fidelity"], core.calibrated_final_fidelity, places=10)
            self.assertTrue(all(isinstance(value, float) for value in public.values()))

    def test_public_infers_chain_and_handles_stationary_and_overrides(self):
        """功能 / Purpose: Check inferred size, identity and public overrides / 检查自动链长、恒等任务与覆盖参数。

        输入 / Inputs: self test fixture / self 测试夹具。
        输出 / Outputs: None; validates no-file scalar results / 无返回；验证不写文件的标量结果。
        """
        inferred = wrapper.simulate_transfer([0, 5, 2], [9, 19, 13])
        explicit = wrapper.simulate_transfer([0, 5, 2], [9, 19, 13], 20)
        self.assertEqual(inferred, explicit)
        self.assertEqual(wrapper.simulate_transfer([2], [2]), {"transfer_time_ns": 0.0, "fidelity": 1.0})
        result = wrapper.simulate_transfer([0], [5], reference_time_ns=100, reference_fidelity=0.9, g_max=None)
        self.assertAlmostEqual(result["transfer_time_ns"], 100)
        self.assertAlmostEqual(result["fidelity"], 0.9, places=12)
        self.assertAlmostEqual(wrapper.simulate_transfer([0], [5], ideal=True)["fidelity"], 1, places=12)

    def test_public_rejects_bad_lists_before_pairing(self):
        """功能 / Purpose: Reject malformed source/target lists / 拒绝非法源目标位置列表。

        输入 / Inputs: self test fixture / self 测试夹具。
        输出 / Outputs: None; unequal, empty, noninteger or duplicate requests fail / 无返回；拒绝不等长、空、非整数及重复请求。
        """
        for sources, targets in (([], []), ([0, 1], [2]), ([0.0], [1]), ([True], [1]),
                                 ([-1], [2]), ([0, 0], [1, 2]), ([0, 1], [2, 2])):
            with self.subTest(sources=sources, targets=targets), self.assertRaises(ValueError):
                wrapper.simulate_transfer(sources, targets)
        for length in (0, 2, 3.5, True):
            with self.subTest(length=length), self.assertRaises(ValueError):
                wrapper.simulate_transfer([0], [3], chain_length=length)

    def test_json_units_calibration_and_distinct_metric_semantics(self):
        """功能 / Purpose: Make units and model assumptions explicit / 明确单位与模型假设。

        输入 / Inputs: self test fixture / self 测试夹具。
        输出 / Outputs: None; metadata identifies user reference and phenomenological limits / 无返回；元数据标明用户参考与唯象限制。
        """
        result = ce.run_simulation(ce.SimulationConfig(2, ((0, 1),), num_steps=2))[0]
        payload = wrapper.result_to_jsonable(result)
        self.assertEqual(payload["time_unit"], "ns")
        self.assertIn("rad/ns", payload["coupling_unit"])
        self.assertIn("ideal", payload["fidelity_fields"]["peak_time, peak_fidelity, final_fidelity"])
        metadata = payload["calibration"]
        self.assertIn("user-provided", metadata["reference_source"])
        self.assertEqual(metadata["reference_time_ns"], 125)
        self.assertEqual(metadata["reference_distance_edges"], 5)
        self.assertAlmostEqual(metadata["attenuation_per_edge"], -math.log(0.992) / 5)
        self.assertIn("not local amplitude damping", metadata["interpretation"])
        self.assertIn("not asserted hardware maximum", metadata["default_cap_source"])
        json.dumps(payload, allow_nan=False)

    def test_csv_separates_conditional_amplitudes_and_distance_fidelity(self):
        """功能 / Purpose: Verify CSV data without filesystem writes / 无磁盘写入地验证 CSV 数据。

        输入 / Inputs: self test fixture / self 测试夹具。
        输出 / Outputs: None; amplitude square is ideal, distance envelope gives calibrated result / 无返回；振幅平方为理想概率。
        """
        data = ce.run_simulation(ce.SimulationConfig(6, ((0, 5),), num_steps=5))
        buffer = io.StringIO()
        with patch.object(Path, "mkdir"), patch.object(Path, "write_text"), \
                patch.object(Path, "open", return_value=contextlib.nullcontext(buffer)), \
                patch.object(wrapper, "write_fidelity_plot") as plot:
            summary = wrapper.write_outputs(Path("unused-test-output"), *data)
        rows = list(csv.DictReader(io.StringIO(buffer.getvalue())))
        self.assertEqual(len(rows), data[0].sample_count)
        row = {key: float(value) for key, value in rows[-1].items()}
        self.assertEqual(row["time_ns"], data[0].final_time)
        self.assertAlmostEqual(row["conditional_ideal_target_amp_real"] ** 2 +
                               row["conditional_ideal_target_amp_imag"] ** 2, row["ideal_target_fidelity"])
        self.assertEqual(row["distance_exposure_edges"], 5)
        self.assertAlmostEqual(row["calibrated_target_fidelity"],
                               row["ideal_target_fidelity"] * row["distance_survival_factor"])
        self.assertLess(row["calibrated_target_fidelity"], row["ideal_target_fidelity"])
        self.assertEqual(summary["calibrated_final_fidelity"], row["calibrated_target_fidelity"])
        plot.assert_called_once()

    def test_plot_uses_ns_and_distinct_legend_line_styles(self):
        """功能 / Purpose: Check plot units and redundant curve encoding / 检查图像单位及双重曲线编码。

        输入 / Inputs: self test fixture / self 测试夹具。
        输出 / Outputs: None; checks axis, legend and curves without PNG writes / 无返回；不写 PNG 地核对坐标轴、图例和曲线。
        """
        result, times, _, _, ideal = ce.run_simulation(ce.SimulationConfig(6, ((0, 5),), num_steps=5))
        with patch("matplotlib.figure.Figure.savefig", autospec=True) as save:
            wrapper.write_fidelity_plot(Path("unused-test.png"), result, times, ideal)
        ax = save.call_args.args[0].axes[0]
        self.assertEqual(ax.get_xlabel(), "Time (ns)")
        self.assertEqual(ax.lines[0].get_linestyle(), "-")
        self.assertEqual(ax.lines[1].get_linestyle(), "--")
        self.assertEqual(len(ax.get_legend().get_texts()), 2)
        exposure = ce.distance_exposure(times, result.route_details["primitives"])
        np.testing.assert_allclose(ax.lines[0].get_ydata(), ce.calibrated_target_fidelities(result.config, exposure, ideal))
        np.testing.assert_allclose(ax.lines[1].get_ydata(), ideal)

    def test_cli_defaults_overrides_and_output_dispatch(self):
        """功能 / Purpose: Check CLI calibration and export dispatch / 检查命令行校准与导出调用。

        输入 / Inputs: self test fixture / self 测试夹具。
        输出 / Outputs: None; validates JSON and mocks actual file writing / 无返回；验证 JSON 且模拟文件写入。
        """
        for options, cap in (([], ce.DEFAULT_G_MAX_RAD_PER_NS), (["--no-coupling-cap"], None), (["--g-max", "0.01"], 0.01)):
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                self.assertEqual(wrapper.main(["-n", "6", "--pairs", "0:5", "--num-steps", "2"] + options), 0)
            summary = json.loads(output.getvalue())
            self.assertEqual(summary["config"]["g_max"], cap)
            self.assertAlmostEqual(summary["calibrated_final_fidelity"], 0.992, places=12)
        with patch.object(wrapper, "write_outputs", return_value=summary) as writer, contextlib.redirect_stdout(io.StringIO()):
            wrapper.main(["-n", "2", "--pairs", "0:1", "--ideal", "--reference-time-ns", "100",
                     "--reference-distance", "1", "--reference-fidelity", "0.9", "--num-steps", "2",
                     "--output-dir", "unused-test-output"])
        writer.assert_called_once()
        result = writer.call_args.args[1]
        self.assertAlmostEqual(result.final_time, 100)
        self.assertAlmostEqual(result.calibrated_final_fidelity, 1)


class ValidationTests(unittest.TestCase):
    def test_strict_pair_and_segment_parsers(self):
        """功能 / Purpose: Check strict pair and stage parsing / 检查严格请求与阶段解析。

        输入 / Inputs: self test fixture / self 测试夹具。
        输出 / Outputs: None; valid inputs parse and malformed inputs fail / 无返回；合法输入通过且非法输入失败。
        """
        self.assertEqual(wrapper.parse_edge_pairs("0:9, 5:19,2:13"), NONCONFLICTING)
        self.assertEqual(wrapper.parse_segment_stages("5:19;0:9,10:15"), EXPECTED_STAGES)
        for text in ("", "0:1,", ",0:1", "0->1", "0:1:2", "a:2"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                wrapper.parse_edge_pairs(text)
        for text in ("", ";", "0:1;", ";0:1"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                wrapper.parse_segment_stages(text)

    def test_invalid_reference_values_and_coupling_caps(self):
        """功能 / Purpose: Reject invalid calibration scales / 拒绝非法校准尺度。

        输入 / Inputs: self test fixture / self 测试夹具。
        输出 / Outputs: None; checks finite positive scales and probability bounds / 无返回；核对有限正尺度与概率界限。
        """
        base = ce.SimulationConfig(2, ((0, 1),))
        for field in ("g_max", "reference_time_ns", "reference_distance"):
            for value in (0, -1, float("nan"), float("inf"), -float("inf")):
                with self.subTest(field=field, value=value), self.assertRaisesRegex(ValueError, "finite and positive"):
                    ce.run_simulation(replace(base, **{field: value}))
        for value in (0, -0.1, 1.01, float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "in \\(0, 1\\]"):
                ce.run_simulation(replace(base, reference_fidelity=value, ideal=True))

    def test_invalid_requests_and_budgets(self):
        """功能 / Purpose: Reject invalid occupations and search controls / 拒绝非法占据与搜索参数。

        输入 / Inputs: self test fixture / self 测试夹具。
        输出 / Outputs: None; illegal requests or budgets raise ValueError / 无返回；非法请求或预算报错。
        """
        bad = [ce.SimulationConfig(0, ((0, 1),)), ce.SimulationConfig(3, ()),
               ce.SimulationConfig(3, ((0, 1), (0, 2))), ce.SimulationConfig(3, ((0, 2), (1, 2))),
               ce.SimulationConfig(3, ((-1, 1),)), ce.SimulationConfig(3, ((0, 3),)),
               ce.SimulationConfig(3, ((0, 1),), num_steps=1), ce.SimulationConfig(3, ((0, 1),), beam_width=0),
               ce.SimulationConfig(3, ((0, 1),), max_segment_steps=0)]
        for config in bad:
            with self.subTest(config=config), self.assertRaises(ValueError):
                ce.run_simulation(config)

    def test_manual_validation_and_explicit_planner_failure(self):
        """功能 / Purpose: Reject invalid manual routes and report heuristic failure / 拒绝非法手动路线并报告启发式失败。

        输入 / Inputs: self test fixture / self 测试夹具。
        输出 / Outputs: None; invalid routes and exhausted searches fail explicitly / 无返回；非法路线及搜索预算耗尽明确报错。
        """
        base = ce.SimulationConfig(20, CONFLICTING)
        for stages in ((), ((),), (((0, 9), (9, 15)),), (((5, 5),),), (((19, 5),),),
                       (((-1, 10),),), (((0, 20),),), (((0, 9),),),
                       (((0.5, 9),),), (((False, 9),),)):
            with self.subTest(stages=stages), self.assertRaises(ValueError):
                ce.run_simulation(replace(base, segments=stages))
        with self.assertRaisesRegex(ValueError, "only used for conflicting"):
            ce.run_simulation(ce.SimulationConfig(20, NONCONFLICTING, segments=EXPECTED_STAGES))
        with self.assertRaisesRegex(ValueError, "heuristic could not find"):
            ce.run_simulation(replace(base, max_segment_steps=1, beam_width=1))

    def test_evaluate_runner_and_public_import_without_routing_dependencies(self):
        """功能 / Purpose: Verify evaluation CLI and lightweight core imports / 验证评估命令行及轻量核心导入。

        输入 / Inputs: Absolute wrapper script path and isolated import probes / wrapper 绝对路径及独立导入探针。
        输出 / Outputs: Correct JSON and public function, no plotting/routing/Qiskit core imports / 正确 JSON、公共函数且核心不导入绘图路由 Qiskit。
        """
        root = Path(__file__).resolve().parents[1]
        script = root / "evaluate" / "run_hamiltonian_simulation.py"
        process = subprocess.run([sys.executable, str(script), "-n", "6", "--pairs", "0:5",
                                  "--num-steps", "2"], cwd="/tmp", capture_output=True, text=True, check=True)
        payload = json.loads(process.stdout)
        self.assertAlmostEqual(payload["final_time"], 125)
        self.assertAlmostEqual(payload["calibrated_final_fidelity"], .992, places=12)
        probe = ("import sys; from qunoc.hamiltonian import estimate_transfer; "
                 "assert not any(n.startswith(('qunoc.routing', 'qiskit', 'matplotlib')) for n in sys.modules); "
                 "from qunoc.hamiltonian import simulate_transfer; "
                 "assert set(simulate_transfer([0],[1])) == {'transfer_time_ns','fidelity'}")
        subprocess.run([sys.executable, "-c", "import sys; sys.path[:0] = " +
                        repr([str(root), str(script.parent)]) + "; " + probe],
                       cwd="/tmp", capture_output=True, text=True, check=True)

    def test_required_cli_removed_modes_and_conflicting_cap_flags(self):
        """功能 / Purpose: Check required CLI and reject removed controls / 检查必填命令行参数并拒绝旧控制选项。

        输入 / Inputs: self test fixture / self 测试夹具。
        输出 / Outputs: None; bad argument sets exit with code two / 无返回；非法参数以状态码二退出。
        """
        for argv in ([], ["-n", "20"], ["--pairs", "0:9"],
                     ["-n", "20", "--pairs", "0:9", "--g-base", "1"],
                     ["-n", "20", "--pairs", "0:9", "--coupling-model", "permutation"],
                     ["-n", "20", "--pairs", "0:9", "--t-stop", "2"],
                     ["-n", "20", "--pairs", "0:9", "--g-max", "1", "--no-coupling-cap"]):
            with self.subTest(argv=argv), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as exc:
                wrapper.main(argv)
            self.assertEqual(exc.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
