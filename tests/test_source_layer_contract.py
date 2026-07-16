from __future__ import annotations

import importlib.util
import unittest

from llmoop.source_oracle import check_source_model_contract
from tests.fixtures import compiled_model_or_skip


ORACLE_DEPS_AVAILABLE = all(importlib.util.find_spec(name) is not None for name in ("torch", "transformers"))


@unittest.skipUnless(ORACLE_DEPS_AVAILABLE, "source oracle dependencies are not installed")
class SourceLayerContractTest(unittest.TestCase):
    def test_all_layers_match_real_source_contracts(self) -> None:
        fixture = compiled_model_or_skip()
        report = check_source_model_contract(
            model_dir=fixture.source_model_dir,
            pedals_dir=fixture.transpiled_dir / "layers",
        )

        self.assertTrue(report.ok, report.errors)
        self.assertEqual(14, len(report.layer_reports))

        layer_00 = report.layer_reports[0]
        self.assertEqual("layer_00", layer_00.layer_id)
        self.assertEqual("conv", layer_00.operator_type)
        self.assertEqual([1, 1024, 3], layer_00.details["state"]["source_cache_shape"])
        self.assertEqual([3, 1024], layer_00.details["state"]["declared_pedal_shape"])

        layer_02 = report.layer_reports[2]
        self.assertEqual("layer_02", layer_02.layer_id)
        self.assertEqual("full_attention", layer_02.operator_type)
        self.assertEqual([1, 8, 1, 64], layer_02.details["state"]["source_key_shape"])
        self.assertEqual([1, 8, 1, 64], layer_02.details["state"]["source_value_shape"])
        self.assertEqual([8, 64], layer_02.details["state"]["declared_key_shape_per_token"])
        self.assertEqual([8, 64], layer_02.details["state"]["declared_value_shape_per_token"])


if __name__ == "__main__":
    unittest.main()
