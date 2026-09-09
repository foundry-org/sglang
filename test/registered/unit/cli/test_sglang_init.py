"""The `sglang` package resolves its frontend-language API lazily (PEP 562) so
that server processes do not import it; the public names must still resolve."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest

import sglang
from sglang.test.test_utils import CustomTestCase


class TestLazyPublicApi(CustomTestCase):
    def test_all_public_names_resolve(self):
        for name in sglang.__all__:
            self.assertIsNotNone(getattr(sglang, name), name)

    def test_lazy_name_is_the_real_object(self):
        from sglang.lang.api import function

        self.assertIs(sglang.function, function)

    def test_from_import_works(self):
        from sglang import global_config  # noqa: F401

    def test_unknown_attribute(self):
        with self.assertRaises(AttributeError):
            sglang.no_such_name


if __name__ == "__main__":
    unittest.main()
