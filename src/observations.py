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

    Normalization rules will be chosen when this class is implemented.
    """

    def __init__(self, environment: CubeStackEnvironment) -> None:
        self.environment = environment

    def build(self, state: StateSnapshot) -> np.ndarray:
        """Return one flat float32 privileged observation."""
        raise NotImplementedError(
            "Privileged observation construction has not been implemented "
            "yet."
        )
