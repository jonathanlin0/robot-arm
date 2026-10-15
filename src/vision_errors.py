"""Recoverable unavailable vision and fatal one-second camera timeouts."""


MAXIMUM_FRAME_AGE_SECONDS = 1.0


class VisionDataUnavailableError(RuntimeError):
    """Required vision is missing or temporarily unusable for this tick.

    LoopRunner.step() will catch this error, keep existing actuator targets,
    skip policy/history updates, and still advance/pace the control interval.
    """


class StaleVisionDataError(RuntimeError):
    """A camera read failure or frame age has exceeded the one-second limit.

    Use elapsed seconds, not a number of camera or control-loop iterations.
    Camera workers retain fatal failures for capture() to raise on the main
    loop; raising only in a worker would not terminate the run. This deliberately
    does not inherit VisionDataUnavailableError. Propagate through capture(),
    step(), run(), and the caller without suppressing the error.
    """


class CameraAcquisitionError(RuntimeError):
    """A camera producer failed unexpectedly; re-raised on the main loop.

    Ordinary unsuccessful reads use the timed StaleVisionDataError path.
    Renderer/setup exceptions must not disappear inside a background thread.
    """
