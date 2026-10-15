"""Control-loop ordering, transactional command commits, and time budgets."""

from types import SimpleNamespace

import numpy as np
import pytest

from loop_runner import LoopRunner
from schemas import CommandReceipt, PlannedCommand, RobotState
from schemas.config import LoopRunnerConfig
from vision_errors import StaleVisionDataError, VisionDataUnavailableError


class Clock:
    now = 0.0
    def __call__(self):
        return self.now
    def wait(self, seconds):
        self.now += seconds


@pytest.fixture
def rig():
    events = []
    clock = Clock()
    state = RobotState(12.0, np.zeros(6), np.zeros(6), np.zeros(6), np.array([0.2,0,0.05]))
    command = PlannedCommand(np.zeros(6), state.gripper_position.copy(), {})
    def log(name, result=None):
        def method(*args, **kwargs):
            events.append((name, args, kwargs))
            return result
        return method
    robot = SimpleNamespace(read=log('read',state))
    camera = SimpleNamespace(capture=log('capture',{'wrist':'frame'}))
    perception = SimpleNamespace(estimate=log('estimate',{'orange':'pose'}),reset=log('perception_reset'))
    policy = SimpleNamespace(predict=log('predict',np.zeros(4)),record_action=log('record'),reset=log('policy_reset'))
    controller = SimpleNamespace(prev_accepted_target=state.gripper_position.copy(),reset=log('controller_reset'),
                                 compute=log('compute',command),update_prev_accepted_xyz=log('accept'))
    backend = SimpleNamespace(send_joint_targets=log('send',CommandReceipt(12,np.zeros(6))),advance=log('advance'))
    runner = LoopRunner(robot,camera,perception,policy,controller,backend,LoopRunnerConfig(control_hz=20.0),
                        deterministic=False,clock=clock,wait=clock.wait)
    return SimpleNamespace(**locals())


def names(rig):
    return [item[0] for item in rig.events]


def test_normal_tick_order_and_latest_timestamp(rig):
    result = rig.runner.step()
    assert result.robot_state is rig.state
    assert names(rig) == ['capture','read','estimate','predict','compute','send','accept','record','advance']
    assert rig.events[2][2]['current_time'] == 12.0
    assert rig.events[3][2]['deterministic'] is False
    assert rig.events[-1][2]['duration_seconds'] == 0.05
    assert rig.clock.now == pytest.approx(0.05)
    rig.runner.step()
    assert rig.clock.now == pytest.approx(0.1)


def test_reset_only_resets_bookkeeping(rig):
    rig.runner.reset()
    assert names(rig) == ['read','controller_reset','policy_reset','perception_reset']
    assert rig.runner.next_deadline is None


@pytest.mark.parametrize('source',['camera','perception'])
def test_missing_input_skips_history_and_commands_but_advances_once(rig,source):
    def missing(*args,**kwargs):
        raise VisionDataUnavailableError('not ready')
    if source == 'camera':
        rig.camera.capture = missing
    else:
        rig.perception.estimate = missing
    assert rig.runner.step() is None
    assert names(rig).count('advance') == 1
    assert not {'predict','compute','send','accept','record'}.intersection(names(rig))
    assert rig.clock.now == pytest.approx(0.05)


def test_stale_is_fatal_and_run_resets_running_state(rig):
    def stale():
        raise StaleVisionDataError('timeout')
    rig.camera.capture = stale
    with pytest.raises(StaleVisionDataError):
        rig.runner.run()
    assert not rig.runner.running
    assert 'advance' not in names(rig)


def test_failed_send_does_not_commit_acceptance_or_previous_action(rig):
    def failure(targets):
        raise OSError('disconnected')
    rig.backend.send_joint_targets = failure
    with pytest.raises(OSError):
        rig.runner.step()
    assert 'accept' not in names(rig)
    assert 'record' not in names(rig)
    assert 'advance' not in names(rig)


def test_wait_accounts_for_compute_time(rig):
    def computation(*args,**kwargs):
        rig.clock.now += 0.03
        return rig.command
    rig.controller.compute = computation
    rig.runner.step()
    assert rig.clock.now == pytest.approx(0.05)


def test_stop_finishes_current_tick(rig):
    original = rig.backend.advance
    def finish(duration_seconds):
        original(duration_seconds=duration_seconds)
        rig.runner.stop()
    rig.backend.advance = finish
    rig.runner.run()
    assert names(rig).count('advance') == 1
    assert not rig.runner.running
