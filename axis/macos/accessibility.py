"""macOS Accessibility permission helper (PRD §10).

Not required for the click-through virtual cursor, but needed later for
real cursor/clicks/scroll. Checks AXIsProcessTrusted() and prints clear
instructions on first run.
"""

from __future__ import annotations

import sys


def is_accessibility_trusted() -> bool:
    """True if this process has macOS Accessibility trust."""
    try:
        import ApplicationServices

        return bool(ApplicationServices.AXIsProcessTrusted())
    except (ImportError, AttributeError):
        # Fallback: cannot determine → treat as trusted so the app runs.
        return True


def ensure_accessibility() -> bool:
    """Check trust; if missing, print instructions. Returns True if trusted."""
    if is_accessibility_trusted():
        return True
    print(
        "\nAxis needs Accessibility permission for future cursor/click/scroll "
        "actions.\n"
        "Enable it in System Settings → Privacy & Security → Accessibility,\n"
        "then restart Axis.\n"
    )
    return False


__all__ = ["is_accessibility_trusted", "ensure_accessibility"]
