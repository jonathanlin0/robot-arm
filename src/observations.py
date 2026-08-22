import numpy as np

from environment import CubeStackEnvironment, StateSnapshot


# joint positions (6) + joint velocities (6) + controls (6)
# + gripper position (3)
# + orange position/orientation/velocity (3 + 4 + 6)
# + blue position/orientation/velocity (3 + 4 + 6)
PRIVILEGED_OBSERVATION_SIZE = 47


class PrivilegedObservationBuilder:
    """Build the first policy's vector from perfect simulator state.

    The intended field order is:

    1. robot joint positions
    2. robot joint velocities
    3. current actuator control targets
    4. gripper XYZ position
    5. orange cube XYZ position, quaternion, and free-joint velocity
    6. blue cube XYZ position, quaternion, and free-joint velocity

    Values are returned in their native simulation units without
    normalization. Simulation time is intentionally excluded.
    """

    def __init__(self, environment: CubeStackEnvironment) -> None:
        self.environment = environment

    def build(self, state: StateSnapshot) -> np.ndarray:
        """Return one flat float32 privileged observation."""
        observation = np.concatenate(
            (
                state["joint_positions"],
                state["joint_velocities"],
                state["controls"],
                state["gripper_position"],
                state["orange_position"],
                state["orange_orientation"],
                state["orange_velocity"],
                state["blue_position"],
                state["blue_orientation"],
                state["blue_velocity"],
            )
        ).astype(np.float32, copy=False)

        expected_shape = (PRIVILEGED_OBSERVATION_SIZE,)
        if observation.shape != expected_shape:
            raise ValueError(
                f"Privileged observation must have shape {expected_shape}; "
                f"received {observation.shape}."
            )
        if not np.all(np.isfinite(observation)):
            raise ValueError(
                "Privileged observation must contain only finite values."
            )

        return observation
