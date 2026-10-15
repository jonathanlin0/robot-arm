"""The shared, paced perception → action → joint-command loop."""

from collections.abc import Callable
from time import monotonic, sleep

from camera_feed_getter import CameraFeedGetter
from cartesian_controller import CartesianController
from cube_state_provider import CubeStateProvider
from joint_command_backend import JointCommandBackend
from robot_state_provider import RobotStateProvider
from schemas import StepResult
from schemas.config import LoopRunnerConfig
from trained_policy import TrainedPolicy
from vision_errors import VisionDataUnavailableError


class LoopRunner:
    """Coordinate externally owned components; never open or duplicate resources."""

    def __init__(
        self,
        robot_state_provider: RobotStateProvider,
        camera_feed_getter: CameraFeedGetter,
        cube_state_provider: CubeStateProvider,
        trained_policy: TrainedPolicy,
        controller: CartesianController,
        backend: JointCommandBackend,
        config: LoopRunnerConfig,
        *,
        deterministic: bool = True,
        clock: Callable[[], float] = monotonic,
        wait: Callable[[float], None] = sleep,
    ) -> None:
        self.robot_state_provider = robot_state_provider
        self.camera_feed_getter = camera_feed_getter
        self.cube_state_provider = cube_state_provider
        self.trained_policy = trained_policy
        self.controller = controller
        self.backend = backend
        self.config = config
        self.deterministic = deterministic
        self.clock = clock
        self.wait = wait
        self.running = False
        self.next_deadline: float | None = None

    def reset(self) -> None:
        """Initialize bookkeeping from measured state; do not reset/move physics."""
        self.controller.reset(self.robot_state_provider.read())
        self.trained_policy.reset()
        self.cube_state_provider.reset()
        self.next_deadline = None

    def step(self) -> StepResult | None:
        """Execute one tick, retaining the last command on unavailable vision.

        Only missing perception is recoverable. Stale camera errors, IK errors,
        and transmission errors propagate. Acceptance/history of issued actions
        are committed only after a successful send.
        """
        interval = 1.0 / self.config.control_hz
        if self.next_deadline is None:
            self.next_deadline = self.clock() + interval
        try:
            frames = self.camera_feed_getter.capture()
            robot = self.robot_state_provider.read()
            cubes = self.cube_state_provider.estimate(
                frames=frames, robot_state=robot, current_time=robot.timestamp,
            )
        except VisionDataUnavailableError:
            result = None
        else:
            if self.controller.prev_accepted_target is None:
                raise RuntimeError("Call LoopRunner.reset() before stepping.")
            action = self.trained_policy.predict(
                robot_state=robot, cube_states=cubes,
                prev_accepted_target=self.controller.prev_accepted_target,
                deterministic=self.deterministic,
            )
            command = self.controller.compute(action=action, robot_state=robot)
            receipt = self.backend.send_joint_targets(command.joint_targets)
            self.controller.update_prev_accepted_xyz(command=command, receipt=receipt)
            self.trained_policy.record_action(action)
            result = StepResult(robot, cubes, action, command, receipt)
        self.backend.advance(duration_seconds=interval)
        remaining = self.next_deadline - self.clock()
        if remaining > 0:
            self.wait(remaining)
        self.next_deadline += interval
        return result

    def run(self) -> None:
        """Run until stop() or a propagated error; task completion is manual."""
        self.running = True
        try:
            while self.running:
                self.step()
        finally:
            self.running = False

    def stop(self) -> None:
        """Finish the current iteration, then stop; do not home or disconnect."""
        self.running = False
