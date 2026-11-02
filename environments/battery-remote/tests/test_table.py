"""The maple table URL moves from the staging bucket (now 404) to production. No Isaac Sim."""

from __future__ import annotations

import os
import unittest
from unittest import mock

from my_package.assets.table import OVERRIDE_ENV, PRODUCTION, STAGING, maple_table_usd

ARENA_URL = STAGING + "Assets/Isaac/6.1/Isaac/IsaacLab/Arena/assets/object_library/srl_robolab_assets/scenes/maple_table.usda"


class MapleTableTests(unittest.TestCase):
    def test_staging_url_moves_to_production_with_the_same_path(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(OVERRIDE_ENV, None)
            url = maple_table_usd(ARENA_URL)
        self.assertTrue(url.startswith(PRODUCTION))
        self.assertTrue(url.endswith("/scenes/maple_table.usda"))
        self.assertEqual(url[len(PRODUCTION):], ARENA_URL[len(STAGING):])

    def test_other_paths_are_untouched(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(OVERRIDE_ENV, None)
            self.assertEqual(maple_table_usd("/local/table.usda"), "/local/table.usda")

    def test_override_wins(self):
        with mock.patch.dict(os.environ, {OVERRIDE_ENV: "/mirror/maple_table.usda"}):
            self.assertEqual(maple_table_usd(ARENA_URL), "/mirror/maple_table.usda")


if __name__ == "__main__":
    unittest.main()
