"""Application-owned simulation lifecycle and component wiring."""

from time import monotonic, sleep

import mujoco
import torch

from camera_feed_getter import SIM_CAMERA_NAMES, SimCameraFeedGetter
from camera_frame_buffer import CameraFrameBuffer
from camera_producers import SimCameraProducer
from cartesian_controller import CartesianController
from cube_state_provider import CubeStateProvider
from environment import CubeStackEnvironment
from joint_command_backend import SimJointCommandBackend
from loop_runner import LoopRunner
from robot_state_provider import SimRobotStateProvider
from schemas import OBSERVATION_FIELDS
from schemas.config import ProjectConfig
from trained_policy import TrainedPolicy


class SimulationSession:
    """Share one simulator among all components and own its camera worker.

    Models are supplied by the application, so loading or training weights does
    not happen inside sensor providers. Use as a context manager to close the
    renderer even when IK or camera exceptions escape the loop.
    """

    def __init__(self, config: ProjectConfig, vision_model: torch.nn.Module,
                 action_policy: torch.nn.Module, device: torch.device) -> None:
        self.config = config
        self.environment = CubeStackEnvironment(scene_path=config.environment.scene_path)
        self.buffers = {
            name: CameraFrameBuffer(
                camera_id=name,
                maximum_frame_age_seconds=config.cameras.maximum_frame_age_seconds,
                clock=monotonic,
            ) for name in SIM_CAMERA_NAMES
        }
        self.camera_producer = SimCameraProducer(
            environment=self.environment, buffers=self.buffers, fps=config.cameras.fps,
            renderer_factory=mujoco.Renderer,
        )
        self.backend = SimJointCommandBackend(environment=self.environment)
        self.runner = LoopRunner(
            robot_state_provider=SimRobotStateProvider(environment=self.environment),
            camera_feed_getter=SimCameraFeedGetter(environment=self.environment, buffers=self.buffers),
            cube_state_provider=CubeStateProvider(
                model=vision_model, device=device,
                maximum_frame_age_seconds=config.cameras.maximum_frame_age_seconds,
            ),
            trained_policy=TrainedPolicy(
                policy=action_policy, observation_fields=OBSERVATION_FIELDS,
                history_length=config.action_policy.history_length,
            ),
            controller=CartesianController(
                robot_model=self.environment.model, config=config.cartesian_controller,
            ),
            backend=self.backend, config=config.loop,
            deterministic=config.action_policy.deterministic, clock=monotonic, wait=sleep,
        )

    def reset(self) -> None:
        """Clear camera images atomically with simulator reset, then clear history."""
        self.camera_producer.reset_session(reset_environment=True)
        self.runner.reset()

    def __enter__(self):
        self.reset()
        self.camera_producer.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.runner.stop()
        self.camera_producer.stop()
