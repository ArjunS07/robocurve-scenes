"""Fetch Isaac Lab's Factory M16 nut and bolt at runtime. Not vendored in this repo.

The YAML names the production-bucket URLs (the same bucket as the maple table).
Isaac Lab's retrieve_file_path copies them into its /tmp cache; an absolute
local path is used as it is. Either way a missing asset stops the launch with
an error that names the URL and how to override it.
"""

from __future__ import annotations

from pathlib import Path

from my_package.scoring.config import NutThreadingConfig

MISSING_HINT = (
    "The nut task needs Isaac Lab's Factory M16 assets (NVIDIA, loaded at runtime). "
    "Check network access to the asset bucket, or download the file and set "
    "fastener.{key} in the nut config to its absolute local path."
)


def resolve_factory_asset(source: str, key: str) -> str:
    """Return a local path for a Factory USD, or raise FileNotFoundError naming the source."""
    if not source.startswith(("https://", "http://", "omniverse://")):
        path = Path(source)
        if not path.is_file():
            raise FileNotFoundError(f"Factory asset {key} not found at {source}. " + MISSING_HINT.format(key=key))
        return str(path)
    from isaaclab.utils.assets import retrieve_file_path

    try:
        local = retrieve_file_path(source, force_download=False)
    except Exception as exc:  # noqa: BLE001 - re-raised with the asset named
        raise FileNotFoundError(
            f"Factory asset {key} could not be fetched from {source}: {exc}. " + MISSING_HINT.format(key=key)
        ) from exc
    if not Path(local).is_file():
        raise FileNotFoundError(f"Factory asset {key} from {source} is missing at {local}. " + MISSING_HINT.format(key=key))
    return str(local)


def resolve_fastener_assets(cfg: NutThreadingConfig) -> tuple[str, str]:
    """(nut, bolt) local USD paths."""
    return (
        resolve_factory_asset(cfg.fastener.nut_usd, "nut_usd"),
        resolve_factory_asset(cfg.fastener.bolt_usd, "bolt_usd"),
    )
