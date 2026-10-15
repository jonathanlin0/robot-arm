import matplotlib
import numpy as np


matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from scripts.plot_intended_trajectory_rewards import (  # noqa: E402
    plot_reward_trace,
    run_intended_pickup_trajectory,
)


def test_intended_pickup_records_and_plots_exact_rewards() -> None:
    trace = run_intended_pickup_trajectory()

    assert trace.success_tick == len(trace.total_rewards)
    assert trace.final_hold_time >= 2.0
    assert trace.stages[0] == "Approach waypoint"
    assert trace.stages[-1] == "Lift orange"
    assert list(dict.fromkeys(trace.stages)) == [
        "Approach waypoint",
        "Descend to orange",
        "Close gripper",
        "Lift orange",
    ]

    tick_count = len(trace.total_rewards)
    assert len(trace.simulation_times) == tick_count
    assert all(
        len(values) == tick_count
        for values in trace.component_rewards.values()
    )
    np.testing.assert_allclose(
        np.diff(trace.simulation_times),
        trace.action_interval,
        rtol=0.0,
        atol=1e-12,
    )

    component_sum_per_tick = np.sum(
        np.asarray(list(trace.component_rewards.values())),
        axis=0,
    )
    np.testing.assert_allclose(
        trace.total_rewards,
        component_sum_per_tick,
        rtol=0.0,
        atol=1e-9,
    )
    assert {
        "approach_orange_progress",
        "approach_orange_waypoint",
        "grasp_candidate",
        "grasp",
        "hold_orange_duration",
        "lift_orange_height",
        "successful_stack",
    }.issubset(trace.component_rewards)

    figure = plot_reward_trace(trace)

    assert len(figure.axes) == 2
    assert len(figure.axes[0].lines[0].get_xdata()) == tick_count
    assert len(figure.axes[0].lines[0].get_ydata()) == tick_count
    plt.close(figure)
