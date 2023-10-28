import dataclasses
import enum
from typing import Optional
from sensor_msgs.msg import JointState

class EventType(enum.Enum):
    MOVE_START = enum.auto()
    MOVE_END = enum.auto()


@dataclasses.dataclass
class CommandEvent():
    command_serial: int
    start_js: list[float]
    end_js: list[float]
    event_type: 'RobotLogStorage.EventType'

@dataclasses.dataclass
class IndexedCommandEvent():
    # To help reverse search back to the state logs
    js_command: CommandEvent
    link_index: int
    # This help when serializing just IndexedCommandEvent to separate file
    state_info : Optional['RobotLogStorage.StateInfo'] = None

class RobotLogStorage():

    @dataclasses.dataclass
    class StateInfo():
        position: list[float]
        velocity: list[float]
        # This could be a serial number
        ros_time_ns: int

        # TODO(LEO) Remove later after proving they are useless
        # serial:int
        indexed_js_command: Optional[CommandEvent]

        def velocity_by_diff(self,last_state_info:'RobotLogStorage.StateInfo'):

            dt = (self.ros_time_ns - last_state_info.ros_time_ns) / 1e9

            return [(new_pos - old_pos) / dt
                    for new_pos, old_pos in zip(self.position, last_state_info.position)]


    def __init__(self, joint_order: list[str]) -> None:
        self.state_logs: list[RobotLogStorage.StateInfo] = []

        self.start_end_logs: list[RobotLogStorage.IndexedJSCommand] = []

        if not joint_order:
            raise RuntimeWarning("Did not provide proper joint name list!! ")
        self.joint_order: list[str] = joint_order

        # self._last_serial = 0

    def log_js(self,
               js: JointState,
               command_event: Optional[CommandEvent] = None) -> None:

        # Some general error handling !
        if len(js.name) != len(self.joint_order):
            raise RuntimeError(
                f"Given mismatch length JS names : {js.name} , expects {self.joint_order}")
        if not all(expect == given for expect, given in zip(js.name, self.joint_order)):
            raise RuntimeError(f"Given wrong JS names : {js.name} , expects {self.joint_order}")

        # Creating data object to store
        new_state = self.StateInfo(js.position, js.velocity, js.header.stamp.nanosec,command_event)

        self.state_logs.append(new_state)

        if command_event:
            indexed_command = IndexedCommandEvent(command_event,
                                                  len(self.state_logs) - 1, new_state)
            self.start_end_logs.append(indexed_command)

    # TODO LEO Make serializing a thing. 