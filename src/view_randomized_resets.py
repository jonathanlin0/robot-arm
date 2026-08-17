#!/usr/bin/env python3

import argparse
import threading
import time

import mujoco
import mujoco.viewer

from environment import CubeStackEnvironment, StateSnapshot


def print_cube_positions(state: StateSnapshot) -> None:
    print(
        f"orange_xy={state['orange_position'][:2]}, "
        f"blue_xy={state['blue_position'][:2]}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for reproducible reset sequences.",
    )
    args = parser.parse_args()

    environment = CubeStackEnvironment(seed=args.seed)

    reset_requested = threading.Event()

    def key_callback(keycode: int) -> None:
        if keycode == ord("R"):
            reset_requested.set()

    state = environment.reset()
    print_cube_positions(state)
    print("Focus the viewer and press R to sample a new layout.")

    with mujoco.viewer.launch_passive(
        environment.model,
        environment.data,
        key_callback=key_callback,
    ) as viewer:
        while viewer.is_running():
            step_start = time.perf_counter()

            if reset_requested.is_set():
                reset_requested.clear()
                state = environment.reset()
                print_cube_positions(state)

            environment.step_physics()
            viewer.sync()

            remaining_time = environment.model.opt.timestep - (
                time.perf_counter() - step_start
            )
            if remaining_time > 0:
                time.sleep(remaining_time)


if __name__ == "__main__":
    main()
