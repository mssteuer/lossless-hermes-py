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


def _ensure_context_engine_symlink():
    """Create a symlink in hermes-agent/plugins/context_engine/ so the
    standard context engine scanner discovers us.

    This bridges the gap between the general plugin system
    (~/.hermes/plugins/) and the context engine discovery path
    (hermes-agent/plugins/context_engine/<name>/).
    """
    try:
        plugin_dir = Path(__file__).parent
        src_dir = plugin_dir / "src" / "lossless_hermes"
        if not src_dir.is_dir():
            return

        # Find hermes-agent/plugins/context_engine/ relative to HERMES_HOME
        import os
        hermes_home = Path(os.environ.get("HERMES_HOME", Path.home() / ".hermes"))
        ce_dir = hermes_home / "hermes-agent" / "plugins" / "context_engine" / "lossless-hermes"

        if ce_dir.exists() or ce_dir.is_symlink():
            # Already exists — check if it points to the right place
            if ce_dir.is_symlink() and ce_dir.resolve() == src_dir.resolve():
                return
            # Stale or wrong — remove and recreate
            ce_dir.unlink(missing_ok=True)

        ce_dir.parent.mkdir(parents=True, exist_ok=True)
        ce_dir.symlink_to(src_dir)
        logger.info("Created context engine symlink: %s -> %s", ce_dir, src_dir)
    except Exception as exc:
        logger.debug("Could not create context engine symlink: %s", exc)


def register(ctx):
    """Called by Hermes plugin system on load.

    Instantiates the LCM context engine and registers it so run_agent.py
    can pick it up via get_plugin_context_engine().
    """
    # Ensure src/ is importable (the git clone won't have it pip-installed)
    src_dir = str(Path(__file__).parent / "src")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

    # Bridge to standard context engine discovery path
    _ensure_context_engine_symlink()

    try:
        from lossless_hermes import LcmContextEngine
        engine = LcmContextEngine()
        ctx.register_context_engine(engine)
        logger.info("LCM context engine registered via plugin system")
    except Exception as exc:
        logger.error("Failed to register LCM context engine: %s", exc)
        raise
