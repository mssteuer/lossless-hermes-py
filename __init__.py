"""Lossless Hermes — LCM Context Engine Plugin for Hermes Agent.

This root __init__.py enables `hermes plugins install` by providing the
register(ctx) entry point that the Hermes plugin system expects.

When installed via `hermes plugins install mssteuer/lossless-hermes-py`,
this file is discovered as the plugin module. It bootstraps the engine
from the src/lossless_hermes/ package and registers it as a context engine.
"""

import sys
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def register(ctx):
    """Called by Hermes plugin system on load.

    Instantiates the LCM context engine and registers it so run_agent.py
    can pick it up via get_plugin_context_engine().
    """
    # Ensure src/ is importable (the git clone won't have it pip-installed)
    src_dir = str(Path(__file__).parent / "src")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

    try:
        from lossless_hermes import LCMContextEngine
        engine = LCMContextEngine()
        ctx.register_context_engine(engine)
        logger.info("LCM context engine registered via plugin system")
    except Exception as exc:
        logger.error("Failed to register LCM context engine: %s", exc)
        raise
