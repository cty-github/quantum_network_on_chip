"""Small layout feasibility regressions / 小规模布局可行性回归测试。

Run with: python -m unittest evaluate.test_layout_feasibility
No training, scoring experiments, or figures / 不训练、不运行评分实验、不生成图片。
"""

import copy
import pickle
import unittest
from itertools import product
from unittest.mock import patch

from qunoc.element.chiplet import FabricateModel, FuncArea, Gap, NodeType
from qunoc.element.package import DesignOrientation, DesignTemplate, Package
from qunoc.layout.env import QuNoCEnv


GEOMETRIES = ("square", "hexagon", "heavy_square", "heavy_hexagon")
BOUNDARIES = ("left", "top", "right", "bottom")


def _package(size, geometry, strategy="rl"):
    """Build an unscored small package / 输入尺寸、几何和策略，返回无评分小封装。"""
    return Package(
        package_size=size,
        fab_model=FabricateModel(compute_geometry=geometry),
        design_strategy=strategy,
        record_design_history=False,
    )


class LayoutFeasibilityTests(unittest.TestCase):
    def setUp(self):
        """Save shared fabrication settings / 保存共享制造设置，供测试后恢复。"""
        self.addCleanup(setattr, FuncArea, "fab_model", FuncArea.fab_model)
        self.addCleanup(setattr, Gap, "fab_model", Gap.fab_model)

    def _assert_plan_matches_design(self, package, module_index, template, orient, interval):
        """Compare planned node sets with a real copied split; return the copy.

        输入封装、模块索引及切分参数；核对完整节点集合，返回实际切分的副本。
        """
        old_module = package.compute_modules[module_index]
        self.assertTrue(Package.is_valid_to_design(old_module, template, orient, interval))
        children, lines = package.plan_module_top_down(old_module, template, orient, interval)
        # Repacking can create AND remove compute sites / 重新铺片会增减计算节点。
        predicted_compute = package.compute_nodes - set(old_module.get_nodes())
        for child in children:
            predicted_compute.update(child.get_nodes())
        predicted_transfer = set(package.transfer_nodes)
        for line in lines:
            predicted_transfer.update(line.get_nodes())
        self.assertFalse(predicted_compute & predicted_transfer)

        actual = copy.deepcopy(package)
        actual.design_module_top_down(
            actual.compute_modules[module_index], template, orient, interval, record=False
        )
        self.assertSetEqual(predicted_compute, actual.compute_nodes)
        self.assertSetEqual(predicted_transfer, actual.transfer_nodes)
        for node_type, expected in (
            (NodeType.Compute, predicted_compute),
            (NodeType.Transfer, predicted_transfer),
            (NodeType.Empty, set(package.package_graph) - predicted_compute - predicted_transfer),
        ):
            self.assertSetEqual(
                expected,
                {node for node, data in actual.package_graph.nodes(data=True)
                 if data["type"] == node_type},
            )
        self.assertEqual(
            len(actual.compute_modules), len(package.compute_modules) - 1 + len(children)
        )
        self.assertEqual(len(actual.transfer_lines), len(package.transfer_lines) + len(lines))
        self.assertEqual(actual.design_history, [])
        return actual

    def _environment(self, size, interval_ratio):
        """Reset a real heavy-square environment with scoring alone stubbed.

        输入尺寸和间隔比例；仅替换评分，返回真实重方格环境及初始观测。
        """
        environment = QuNoCEnv(
            package_size=size,
            fab_model=FabricateModel(compute_geometry="heavy_square"),
            compute_ratio=0.8,
            max_module_slots=64,
            relative_interval_bins=(interval_ratio,),
            record_design_history=False,
            estimate_workers=1,
        )
        self.addCleanup(environment.close)
        score_patch = patch.object(environment, "estimate_with_details", return_value=(1.0, {}))
        score_patch.start()
        self.addCleanup(score_patch.stop)
        observation, _ = environment.reset(seed=0)
        return environment, observation

    def test_radial_repacking_rejects_infeasible_action(self):
        """Reject 13x13 heavy-square Radial X: 105 -> 80 below 0.8.

        断言重方格径向切分的真实保留量为 80/105，动作掩码必须拒绝。
        """
        environment, observation = self._environment((13, 13), 3 / 11)
        module = environment.package.compute_modules[0]
        interval = environment._interval_for_module(module, 0)
        self.assertEqual(interval, (3, 3))
        self.assertEqual(environment.initial_compute_node_count, 105)
        action = dict(old_module=module, template=DesignTemplate.Radial,
                      orient=DesignOrientation.X, interval=interval)
        actual = self._assert_plan_matches_design(
            environment.package, 0, action["template"], action["orient"], interval
        )
        self.assertEqual(len(actual.compute_nodes), 80)
        self.assertLess(80 / 105, environment.compute_ratio_target)
        self.assertFalse(environment._would_preserve_compute_ratio(action))
        index = environment.encode_action_index(
            0, DesignTemplate.Radial.value, DesignOrientation.X.value, 0
        )
        self.assertFalse(environment.get_action_mask()[index])
        self.assertEqual(observation["action_mask"][index], 0.0)
        self.assertIsNone(environment.deserialize_action(index))
        self.assertEqual(len(environment.package.compute_nodes), 105)

    def test_stripe_repacking_allows_action_and_step(self):
        """Allow and apply 7x7 heavy-square Stripe X: 21 -> 20 above 0.8.

        断言重方格条带切分保留 20/21，真实 step 成功且保持可行。
        """
        environment, observation = self._environment((7, 7), 0.4)
        module = environment.package.compute_modules[0]
        interval = environment._interval_for_module(module, 0)
        self.assertEqual(interval, (2, 2))
        self.assertEqual(environment.initial_compute_node_count, 21)
        action = dict(old_module=module, template=DesignTemplate.Stripe,
                      orient=DesignOrientation.X, interval=interval)
        actual = self._assert_plan_matches_design(
            environment.package, 0, action["template"], action["orient"], interval
        )
        self.assertEqual(len(actual.compute_nodes), 20)
        self.assertTrue(actual.compute_nodes - environment.package.compute_nodes)
        self.assertTrue(environment._would_preserve_compute_ratio(action))
        index = environment.encode_action_index(
            0, DesignTemplate.Stripe.value, DesignOrientation.X.value, 0
        )
        self.assertTrue(environment.get_action_mask()[index])
        self.assertEqual(observation["action_mask"][index], 1.0)
        self.assertEqual(environment.deserialize_action(index), action)
        next_observation, reward, _, truncated, info = environment.step(index)
        self.assertEqual(info["step_outcome"], "applied")
        self.assertTrue(info["action_valid"])
        self.assertTrue(info["feasible"])
        self.assertFalse(truncated)
        self.assertEqual(reward, 0.0)
        self.assertAlmostEqual(info["compute_ratio"], 20 / 21)
        self.assertSetEqual(environment.package.compute_nodes, actual.compute_nodes)
        self.assertSetEqual(environment.package.transfer_nodes, actual.transfer_nodes)
        self.assertTrue(environment.observation_space.contains(next_observation))

    def test_prediction_matrix_matches_actual_node_sets(self):
        """Check valid small splits across four geometries, templates, and orientations.

        遍历四种几何、全部模板和方向的小尺寸合法切分，核对完整节点集合。
        """
        sizes = ((7, 7), (9, 11), (12, 8), (13, 13))
        intervals = ((1, 1), (2, 2), (3, 3), (1, 2), (2, 3))
        covered = set()
        for strategy, geometry, size in product(("rl", "manual"), GEOMETRIES, sizes):
            package = _package(size, geometry, strategy)
            for template, orient, interval in product(
                DesignTemplate, DesignOrientation, intervals
            ):
                if not Package.is_valid_to_design(
                    package.compute_modules[0], template, orient, interval
                ):
                    continue
                with self.subTest(strategy=strategy, geometry=geometry, size=size,
                                  template=template, orient=orient, interval=interval):
                    self._assert_plan_matches_design(package, 0, template, orient, interval)
                    covered.add((strategy, geometry, template, orient))
        self.assertEqual(
            covered,
            set(product(("rl", "manual"), GEOMETRIES, DesignTemplate, DesignOrientation)),
        )

    def test_repeated_splits_preserve_untouched_modules(self):
        """Predict successive splits including existing siblings and transfer boundaries.

        对连续切分预测全封装节点，覆盖已有兄弟模块和传输边界。
        """
        splits = (
            (DesignTemplate.Stripe, DesignOrientation.X, (5, 5)),
            (DesignTemplate.Radial, DesignOrientation.Y, (1, 1)),
            (DesignTemplate.Grid, DesignOrientation.X, (1, 1)),
            (DesignTemplate.Stripe, DesignOrientation.Y, (1, 1)),
        )
        for geometry in GEOMETRIES:
            package = _package((17, 15), geometry)
            for step, (template, orient, interval) in enumerate(splits):
                with self.subTest(geometry=geometry, step=step):
                    candidates = [
                        index for index, module in enumerate(package.compute_modules)
                        if Package.is_valid_to_design(module, template, orient, interval)
                    ]
                    self.assertTrue(candidates, "Repeated-split fixture must remain valid")
                    module_index = candidates[0]
                    untouched = set().union(*(
                        set(module.get_nodes()) for index, module in enumerate(package.compute_modules)
                        if index != module_index
                    ))
                    package = self._assert_plan_matches_design(
                        package, module_index, template, orient, interval
                    )
                    self.assertTrue(untouched <= package.compute_nodes)

    def test_planner_is_pure_and_retains_boundary_identity(self):
        """Keep all package state unchanged and reuse boundary objects, not copies.

        输入含整数或传输线边界的封装；断言规划无副作用且保留边界对象身份。
        """
        for strategy, geometry in product(("rl", "manual"), GEOMETRIES):
            package = _package((13, 13), geometry, strategy)
            old_module = package.compute_modules[0]
            for template, orient in product(DesignTemplate, DesignOrientation):
                with self.subTest(strategy=strategy, geometry=geometry,
                                  template=template, orient=orient):
                    # Bytes include graph attributes, caches, tree and history / 字节快照覆盖图、缓存、树及历史。
                    before = pickle.dumps(package, protocol=pickle.HIGHEST_PROTOCOL)
                    identities = {
                        name: getattr(package, name) for name in (
                            "package_graph", "compute_nodes", "transfer_nodes", "compute_modules",
                            "transfer_lines", "design_tree_root", "design_tree_dict", "design_history"
                        )
                    }
                    children, lines = package.plan_module_top_down(
                        old_module, template, orient, (2, 2)
                    )
                    self.assertEqual(pickle.dumps(package, protocol=pickle.HIGHEST_PROTOCOL), before)
                    for name, original in identities.items():
                        self.assertIs(getattr(package, name), original)
                    self.assertNotIn(old_module, children)
                    self.assertTrue(children)
                    self.assertTrue(lines)
                    for side in BOUNDARIES:
                        old_coord, old_line, _ = getattr(old_module, side)
                        inherited = [child for child in children
                                     if getattr(child, side)[0] == old_coord]
                        self.assertTrue(inherited)
                        for child in inherited:
                            self.assertIs(getattr(child, side)[1], old_line)
                    for child in children:
                        for side in BOUNDARIES:
                            coord, line, _ = getattr(child, side)
                            old_coord, old_line, _ = getattr(old_module, side)
                            if coord == old_coord:
                                self.assertIs(line, old_line)
                            else:
                                self.assertTrue(any(line is new_line for new_line in lines))
                    second_children, second_lines = package.plan_module_top_down(
                        old_module, template, orient, (2, 2)
                    )
                    self.assertEqual([str(child) for child in children],
                                     [str(child) for child in second_children])
                    self.assertTrue(all(first is not second
                                        for first, second in zip(children, second_children)))
                    self.assertTrue(all(first is not second
                                        for first, second in zip(lines, second_lines)))
                    self.assertEqual(pickle.dumps(package, protocol=pickle.HIGHEST_PROTOCOL), before)


if __name__ == "__main__":
    unittest.main()
