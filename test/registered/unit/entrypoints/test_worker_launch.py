"""Unit tests for `entrypoints/worker_launch.py`.

The rank layout helpers moved here from `entrypoints/engine.py`; the engine
module re-exports them, and subclasses (the Ray engine) still import them from
there.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=12, suite="base-a-test-cpu")

import unittest

from sglang.srt.entrypoints import engine, worker_launch
from sglang.srt.entrypoints.worker_launch import (
    LaunchResources,
    _calculate_rank_ranges,
)
from sglang.test.test_utils import CustomTestCase


class TestRankRanges(CustomTestCase):
    def test_single_node(self):
        pp, tp, pp_per_node, tp_per_node = _calculate_rank_ranges(
            nnodes=1, pp_size=1, tp_size=4, node_rank=0
        )
        self.assertEqual((list(pp), list(tp)), ([0], [0, 1, 2, 3]))
        self.assertEqual((pp_per_node, tp_per_node), (1, 4))

    def test_tp_group_spanning_two_nodes(self):
        _, tp, _, tp_per_node = _calculate_rank_ranges(
            nnodes=2, pp_size=1, tp_size=8, node_rank=1
        )
        self.assertEqual(list(tp), [4, 5, 6, 7])
        self.assertEqual(tp_per_node, 4)

    def test_one_pp_stage_per_node(self):
        pp, tp, pp_per_node, tp_per_node = _calculate_rank_ranges(
            nnodes=2, pp_size=2, tp_size=4, node_rank=1
        )
        self.assertEqual((list(pp), list(tp)), ([1], [0, 1, 2, 3]))
        self.assertEqual((pp_per_node, tp_per_node), (1, 4))


class TestEngineReExports(CustomTestCase):
    def test_same_objects(self):
        for name in (
            "SchedulerInitResult",
            "_set_envs_and_config",
            "_calculate_rank_ranges",
            "_compute_parallelism_ranks",
            "_wait_for_scheduler_ready",
        ):
            self.assertIs(getattr(engine, name), getattr(worker_launch, name), name)


class TestLaunchResources(CustomTestCase):
    def test_defaults(self):
        res = LaunchResources(port_args=object())
        self.assertIsNone(res.engine_info_bootstrap_server)
        self.assertEqual(res.weight_cache_daemon_procs, [])


if __name__ == "__main__":
    unittest.main()
