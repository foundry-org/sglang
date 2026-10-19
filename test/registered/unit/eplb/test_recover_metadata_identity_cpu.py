"""CPU-only identity regression for the actual broadcast/update functions.

Extracts their source AST to avoid importing GPU runtime dependencies. The tiny
list-backed tensor implements only the operations exercised here; this verifies
object/storage-owner preservation and update ordering, not CUDA execution.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import ast
import copy
import unittest
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock


class Tensor:
    def __init__(self, values, contiguous=True):
        self.values = copy.deepcopy(values)
        self.device = "cpu"
        self.is_contiguous = contiguous

    @property
    def shape(self):
        def dimensions(value):
            return (
                (len(value), *dimensions(value[0])) if isinstance(value, list) else ()
            )

        return dimensions(self.values)

    def dim(self):
        return len(self.shape)

    def contiguous(self):
        return self if self.is_contiguous else Tensor(self.values)

    def view(self, *shape):
        return self

    def to(self, *args, **kwargs):
        return self

    def __setitem__(self, key, other):
        assert key is Ellipsis
        self.values = copy.deepcopy(other.values)


def load_actual_functions():
    path = (
        Path(__file__).resolve().parents[4]
        / "python/sglang/srt/eplb/expert_location.py"
    )
    tree = ast.parse(path.read_text())
    selected = [
        node
        for node in tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef))
        and node.name
        in ("ExpertLocationMetadata", "broadcast_global_expert_location_metadata")
    ]
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    torch = SimpleNamespace(
        Tensor=Tensor,
        tensor=Tensor,
        distributed=SimpleNamespace(broadcast=Mock()),
        where=lambda mask, other, old: Tensor(
            [
                other.values[i] if chosen else old.values[i]
                for i, chosen in enumerate(mask.values)
            ]
        ),
    )
    namespace = {"dataclass": dataclass, "torch": torch}
    exec(
        compile(
            ast.fix_missing_locations(
                ast.Module(body=[future, *selected], type_ignores=[])
            ),
            str(path),
            "exec",
        ),
        namespace,
    )
    return namespace


def metadata(cls, value, ep_size=4, width=8, dispatch=True):
    return cls(
        physical_to_logical_map=Tensor([[value] * width for _ in range(2)]),
        physical_to_logical_map_cpu=Tensor([[value] * width for _ in range(2)]),
        logical_to_all_physical_map=Tensor(
            [[[value] * width for _ in range(2)] for _ in range(2)]
        ),
        logical_to_all_physical_map_cpu=Tensor(
            [[[value] * width for _ in range(2)] for _ in range(2)]
        ),
        logical_to_all_physical_map_num_valid=Tensor([[value] * 2 for _ in range(2)]),
        ep_size=ep_size,
        logical_to_rank_dispatch_physical_map=Tensor([[value] * 2 for _ in range(2)])
        if dispatch
        else None,
    )


class TestRecoveryMetadataIdentity(unittest.TestCase):
    def setUp(self):
        self.ns = load_actual_functions()
        self.cls = self.ns["ExpertLocationMetadata"]
        self.setter = Mock()
        self.ns["set_global_expert_location_metadata"] = self.setter

    def broadcast(self, old, new):
        self.ns["get_global_expert_location_metadata"] = lambda: old
        self.cls.init_by_mapping = Mock(return_value=new)
        return self.ns["broadcast_global_expert_location_metadata"](
            model_config=object(), moe_ep_rank=2, src_rank=0, group="cpu-group"
        )

    def test_all_six_tensor_owners_and_views_survive_repeated_recovery(self):
        old = metadata(self.cls, 0)
        tensors = {
            name: tensor
            for name, tensor in vars(old).items()
            if isinstance(tensor, Tensor)
        }
        self.assertEqual(len(tensors), 6)
        for value in (1, 2):
            new = metadata(self.cls, value)
            self.assertIs(self.broadcast(old, new), old)
            self.setter.assert_not_called()
            for name, tensor in tensors.items():
                self.assertIs(getattr(old, name), tensor)
                self.assertEqual(tensor.values, getattr(new, name).values)
        self.ns["torch"].distributed.broadcast.assert_called_with(
            old.physical_to_logical_map, src=0, group="cpu-group"
        )
        # A later native EPLB update must still reach those same retained views.
        latest = metadata(self.cls, 3)
        old.update(latest, update_layer_ids=[1])
        for name, tensor in tensors.items():
            self.assertEqual(tensor.values[0], getattr(new, name).values[0])
            self.assertEqual(tensor.values[1], getattr(latest, name).values[1])

    def test_optional_dispatch_map_remains_none(self):
        old, new = (
            metadata(self.cls, 0, dispatch=False),
            metadata(self.cls, 1, dispatch=False),
        )
        self.assertIs(self.broadcast(old, new), old)
        self.assertIsNone(old.logical_to_rank_dispatch_physical_map)

    def test_noncontiguous_physical_map_keeps_original_owner(self):
        old, new = metadata(self.cls, 0), metadata(self.cls, 1)
        original = old.physical_to_logical_map
        original.is_contiguous = False
        self.ns["torch"].distributed.broadcast.side_effect = lambda tensor, **kwargs: (
            tensor.__setitem__(Ellipsis, new.physical_to_logical_map)
        )
        self.assertIs(self.broadcast(old, new), old)
        self.assertIs(old.physical_to_logical_map, original)
        self.assertEqual(original.values, new.physical_to_logical_map.values)
        self.assertFalse(original.is_contiguous)
        broadcast_map = self.ns["torch"].distributed.broadcast.call_args.args[0]
        self.assertIsNot(broadcast_map, original)
        self.assertTrue(broadcast_map.is_contiguous)
        self.assertEqual(broadcast_map.values, new.physical_to_logical_map.values)
        self.assertIs(self.cls.init_by_mapping.call_args.args[1], broadcast_map)
        self.setter.assert_not_called()

    def test_changed_geometry_keeps_existing_scale_replacement_path(self):
        for kwargs in ({"ep_size": 8}, {"ep_size": 8, "width": 16}):
            with self.subTest(kwargs=kwargs):
                old, new = metadata(self.cls, 0), metadata(self.cls, 1, **kwargs)
                self.assertIs(self.broadcast(old, new), new)
                self.setter.assert_called_with(new, allow_overwrite=True)


if __name__ == "__main__":
    unittest.main()
