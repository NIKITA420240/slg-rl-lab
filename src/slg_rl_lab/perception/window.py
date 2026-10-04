"""Locate a Windows game window and track its client area."""

import ctypes
import sys

if sys.platform == "win32":
    import win32gui


def enable_dpi_awareness() -> None:
    """Keep window coordinates in physical pixels when display scaling is enabled."""
    api = ctypes.WinDLL("user32", use_last_error=True)
    api.SetProcessDpiAwarenessContext.argtypes = [ctypes.c_void_p]
    api.SetProcessDpiAwarenessContext.restype = ctypes.c_bool
    if not api.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4)):
        error = ctypes.get_last_error()
        if error != 5:  # Another library may have already set DPI awareness.
            raise ctypes.WinError(error)


class GameWindow:
    def __init__(self, title: str):
        if sys.platform != "win32":
            raise RuntimeError("Game window capture is currently supported on Windows only.")
        if not title.strip():
            raise ValueError("Window title must not be empty.")

        enable_dpi_awareness()
        matches = []

        def collect(hwnd, _):
            if win32gui.IsWindowVisible(hwnd):
                name = win32gui.GetWindowText(hwnd)
                if title.casefold() in name.casefold():
                    matches.append((hwnd, name))

        win32gui.EnumWindows(collect, None)
        if not matches:
            raise RuntimeError(f"Window containing {title!r} was not found. Open the game first.")
        if len(matches) > 1:
            titles = ", ".join(name for _, name in matches)
            raise RuntimeError(f"Multiple windows match {title!r}: {titles}. Use a more specific title.")
        self.handle, self.title = matches[0]

    def get_region(self) -> dict[str, int] | None:
        """Return client coordinates, or None while the window is minimized."""
        if not win32gui.IsWindow(self.handle):
            raise RuntimeError("The game window was closed. Restart live inference after opening it.")
        if win32gui.IsIconic(self.handle) or not win32gui.IsWindowVisible(self.handle):
            return None

        left, top, right, bottom = win32gui.GetClientRect(self.handle)
        x, y = win32gui.ClientToScreen(self.handle, (left, top))
        width, height = right - left, bottom - top
        if width <= 0 or height <= 0:
            return None
        return {"left": x, "top": y, "width": width, "height": height}
