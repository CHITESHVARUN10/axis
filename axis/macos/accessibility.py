"""macOS Accessibility permission helper (PRD §10).

Needed for real cursor/clicks/scroll (moving the actual macOS pointer via
Quartz requires the process to be a trusted Accessibility client since
macOS 10.14). The click-through avatar cursor does NOT need it.
"""

from __future__ import annotations

import sys

_GUIDANCE_SHOWN = False


def is_accessibility_trusted() -> bool:
    """True if this process has macOS Accessibility trust."""
    if sys.platform != "darwin":
        return False
    try:
        import ApplicationServices

        return bool(ApplicationServices.AXIsProcessTrusted())
    except (ImportError, AttributeError):
        return False


def print_accessibility_guidance() -> None:
    """Print the enable-accessibility instructions (once per process)."""
    global _GUIDANCE_SHOWN
    if _GUIDANCE_SHOWN:
        return
    _GUIDANCE_SHOWN = True
    print(
        "\nAxis needs Accessibility permission for real-cursor control "
        "(hovering over apps, clicks).\n"
        "Enable it in System Settings → Privacy & Security → Accessibility,\n"
        "add your terminal app, then relaunch Axis. Until then, the avatar\n"
        "cursor will be shown instead.\n"
    )


def ensure_accessibility() -> bool:
    """Check trust; if missing, print guidance once. Returns True if trusted."""
    trusted = is_accessibility_trusted()
    if not trusted:
        print_accessibility_guidance()
    return trusted


__all__ = ["is_accessibility_trusted", "print_accessibility_guidance", "ensure_accessibility"]