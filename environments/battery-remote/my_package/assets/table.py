"""Arena's maple table background, loaded from the bucket that still serves it. No Isaac imports."""

from __future__ import annotations

import os

STAGING = "https://omniverse-content-staging.s3-us-west-2.amazonaws.com/"
PRODUCTION = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/"
# Checked 2026-11-02: the installed Arena points maple_table_robolab at the staging
# bucket, where the 6.1 file and its ../fixtures/table_maple.usd now return 404. The same
# paths return 200 on the production bucket, and the scene file is byte-for-byte the size
# of the copy Isaac Lab cached earlier (10153 bytes), so the measured tabletop still holds.
OVERRIDE_ENV = "ROBOCURVE_MAPLE_TABLE_USD"


def maple_table_usd(arena_url: str) -> str:
    """USD path to load: the override if set, else Arena's URL moved to the production bucket."""
    override = os.environ.get(OVERRIDE_ENV)
    if override:
        return override
    if arena_url.startswith(STAGING):
        return PRODUCTION + arena_url[len(STAGING):]
    return arena_url


def maple_table(asset_registry):
    """Instantiate Arena's maple_table_robolab with a reachable usd_path."""
    cls = asset_registry.get_asset_by_name("maple_table_robolab")
    usd = maple_table_usd(cls.usd_path)
    if usd == cls.usd_path:
        return cls()
    # Arena opens usd_path inside __init__, so it must be set on the class before construction.
    return type(cls.__name__, (cls,), {"usd_path": usd})()
