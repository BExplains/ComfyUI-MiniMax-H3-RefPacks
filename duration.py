"""MiniMax H3 video duration conversion, independent of model dependencies."""
import math


def h3_duration_frames(seconds):
    """Round a positive duration up to 17n+5 frames at native H3's 24 fps."""
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("H3 video duration must be a finite number greater than zero.")
    requested = seconds * 24.0
    # Avoid advancing a whole H3 block due to floating-point boundary noise.
    if math.isclose(requested, round(requested), rel_tol=0, abs_tol=1e-9):
        requested = round(requested)
    return 5 + 17 * max(0, math.ceil((requested - 5) / 17))
