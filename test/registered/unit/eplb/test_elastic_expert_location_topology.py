"""Faulted PG membership must not renumber the expert slot topology."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.eplb import expert_location_updater as updater


class TestElasticExpertLocationTopology(unittest.TestCase):
    def test_reorders_local_experts_with_one_survivor(self):
        old = SimpleNamespace(
            ep_size=2,
            num_local_physical_experts=2,
            physical_to_logical_map_cpu=torch.tensor([[0, 1, 0, 1]]),
        )
        new = SimpleNamespace(physical_to_logical_map_cpu=torch.tensor([[1, 0, 0, 1]]))
        weights = torch.tensor([[10.0], [20.0]])
        state = SimpleNamespace(active_ranks_cpu=torch.tensor([1, 0]))
        with (
            patch.object(updater.ElasticEPStateManager, "instance", return_value=state),
            patch.object(updater.torch.distributed, "get_world_size", return_value=1),
            patch.object(updater, "get_bool_env_var", return_value=False),
        ):
            missing = updater._update_expert_weights_raw(
                {0: [weights]}, old, new, [0], nnodes=2, rank=0
            )
        self.assertEqual(missing, {})
        torch.testing.assert_close(weights, torch.tensor([[20.0], [10.0]]))

    def test_surviving_global_rank_keeps_original_node_boundary(self):
        old = SimpleNamespace(
            ep_size=4,
            num_local_physical_experts=2,
            physical_to_logical_map_cpu=torch.tensor([[0, 1] * 4]),
        )
        new = SimpleNamespace(physical_to_logical_map_cpu=torch.tensor([[0, 1] * 4]))
        state = SimpleNamespace(active_ranks_cpu=torch.tensor([1, 0, 1, 0]))
        with (
            patch.object(updater.ElasticEPStateManager, "instance", return_value=state),
            patch.object(updater.torch.distributed, "get_world_size", return_value=2),
            patch.object(updater, "get_bool_env_var", return_value=False),
            patch.object(updater, "update_expert_weights_single_layer") as update,
        ):
            updater._update_expert_weights_raw(
                {0: [torch.zeros(2, 1)]}, old, new, [0], nnodes=2, rank=2
            )
        self.assertEqual(update.call_args.kwargs["num_gpu_per_node"], 2)
        self.assertEqual(update.call_args.kwargs["world_size"], 4)
        self.assertEqual(update.call_args.kwargs["rank"], 2)


if __name__ == "__main__":
    unittest.main()
