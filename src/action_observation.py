"""Synchronous learned vision observations for scripted action demonstrations."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import ExitStack
from typing import Any

import mujoco
import numpy as np
import torch

from camera_feed_getter import SIM_CAMERA_NAMES
from cube_state_provider import CubeStateProvider
from environment import CubeStackEnvironment
from schemas import CAMERA_IMAGE_SHAPES, CameraFrame, RobotState
from trained_policy import build_policy_observation


class VisionActionObserver:
    """Reuse two renderers and derive policy features from learned cube poses.

    Rendering observes the current simulation state without advancing time.
    Velocity history belongs to one complete physical episode, including its
    drop recovery; ``reset`` starts a new episode without recreating renderers.
    """

    def __init__(
        self, environment: CubeStackEnvironment, vision_model: torch.nn.Module,
        device: torch.device, maximum_frame_age_seconds: float,
        renderer_factory: Callable[..., Any] = mujoco.Renderer,
    ) -> None:
        self.environment = environment
        self.provider = CubeStateProvider(
            vision_model, device, maximum_frame_age_seconds=maximum_frame_age_seconds,
        )
        self.renderer_factory = renderer_factory
        self.renderers: dict[str, Any] = {}
        self._resources: ExitStack | None = None

    def __enter__(self) -> VisionActionObserver:
        if self._resources is not None:
            raise RuntimeError("VisionActionObserver is already open.")
        self.reset()
        # Register each allocation immediately so a later allocation failure
        # also releases every renderer that was successfully constructed.
        with ExitStack() as resources:
            renderers = {}
            for name in SIM_CAMERA_NAMES:
                height, width, _ = CAMERA_IMAGE_SHAPES[name]
                renderer = self.renderer_factory(self.environment.model, height=height, width=width)
                resources.callback(renderer.close)
                renderers[name] = renderer
            self.renderers = renderers
            self._resources = resources.pop_all()
        return self

    def __exit__(self, *exception: Any) -> bool | None:
        resources, self._resources = self._resources, None
        self.renderers = {}
        try:
            if resources is not None:
                return resources.__exit__(*exception)
            return None
        finally:
            self.reset()

    def reset(self) -> None:
        """Clear inferred velocity history at a full simulation reset only."""
        self.provider.reset()

    def observe(self, robot_state: RobotState) -> np.ndarray:
        """Return 37 pre-action values, with cube features entirely from vision."""
        if self._resources is None:
            raise RuntimeError("Open VisionActionObserver before observing.")
        with self.environment.state_lock:
            timestamp = float(self.environment.data.time)
            frames = {}
            for name, camera_name in SIM_CAMERA_NAMES.items():
                renderer = self.renderers[name]
                renderer.update_scene(self.environment.data, camera=camera_name)
                frames[name] = CameraFrame(
                    camera_id=name, timestamp=timestamp,
                    rgb=np.array(renderer.render(), copy=True),
                )
        cubes = self.provider.estimate(frames, robot_state, current_time=timestamp)
        return build_policy_observation(robot_state, cubes)
