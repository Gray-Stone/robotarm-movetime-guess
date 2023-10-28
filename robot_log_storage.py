import dataclasses
import enum
from typing import Optional
from sensor_msgs.msg import JointState


@dataclasses.dataclass
class JSCommand():
    command_serial: int
    start_js: list[float]
    end_js: list[float]


class RobotLogStorage():

    @enum.Enum
    class EventType():
        MOVE_START = enum.auto()
        MOVE_END = enum.auto()

    @dataclasses.dataclass
    class StateInfo():
        position: list[float]
        velocity: list[float]
        # This could be a serial number
        ros_time_ns: int

        # TODO(LEO) Remove later after proving they are useless
        # serial:int
        # js_command: Optional[JSCommand] = None
        # event_type: Optional['RobotLogStorage.EventType'] = None
        indexed_js_command: Optional['RobotLogStorage.IndexedJSCommand']

    @dataclasses.dataclass
    class IndexedJSCommand():
        js_command: JSCommand
        event_type: 'RobotLogStorage.EventType'
        index: int  # To reverse search back to the state logs

    def __init__(self, joint_order: list[str]) -> None:
        self.state_logs: list[RobotLogStorage.StateInfo] = []

        self.start_end_logs: list[RobotLogStorage.IndexedJSCommand] = []

        if not joint_order:
            raise RuntimeWarning("Did not provide proper joint name list!! ")
        self.joint_order: list[str] = joint_order

        # self._last_serial = 0

    def log_js(self,
               js: JointState,
               command_event_tuple: Optional[tuple[JSCommand, EventType]] = None) -> None:

        if len(js.name) != len(self.joint_order):
            raise RuntimeError(
                f"Given mismatch length JS names : {js.name} , expects {self.joint_order}")
        if not all(expect == given for expect, given in zip(js.name, self.joint_order)):
            raise RuntimeError(f"Given wrong JS names : {js.name} , expects {self.joint_order}")

        new_state = self.StateInfo(js.position, js.velocity, js.header.stamp.nanosec)

        self.state_logs.append(new_state)

        if command_event_tuple:
            indexed_command = self.IndexedJSCommand(command_event_tuple[0], command_event_tuple[1],
                                                    len(self.state_logs) - 1)
            # I hope the append is shallow copy, so this kind of edits works
            new_state.indexed_js_command = indexed_command
            # self.state_logs[-1].indexed_js_command = indexed_command
            # new_state.js_command = command_event_tuple[0]
            # new_state.event_type = command_event_tuple[1]
            self.start_end_logs.append(indexed_command)

        # self._last_serial +=1
