#! /usr/bin/env python3
import time

from sensor_msgs.msg import JointState
import signal

import asyncio.events
from threading import Event

from interbotix_xs_modules.xs_robot.arm import InterbotixManipulatorXS



int_event = Event()
def int_handle (signal , frame):
    print("Sig Int Catched")
    int_event.set()

signal.signal(signal.SIGINT)




# The robot object is what you use to control the robot
robot = InterbotixManipulatorXS("px100", "arm", "gripper")




last_js  = robot.core.joint_states

last_cycle_start = time.monotonic_ns()


# On each loop: check if js is updated, log js if so.
# Check if there are move currently going on. if not, start a new move. and log the move info, with current cycle's js serial number
# Check if a commanded joint move is done. if so, also log the move finished. but anchor to a js serial number

# Interbotix have no checking on if a move is done. We need to do our own tolerance check.

# use the python time monotonic_ns to be current cycle serial number.

while True:
    # These to only get saved into last_* if this cycle is meaningful.
    js = robot.core.joint_states
    cycle_start = time.monotonic_ns()
    if js.header == last_js.header:
        # There are totally nothing to do if JS is not updated.

        print("Same js, no change!")
        continue

    # New js.
    js_t_delta = js.header.stamp.nanosec - last_js.header.stamp.nanosec
    py_t_delta = cycle_start - last_cycle_start

    print(f"\n ...\ncycle_delta {(py_t_delta) /1000000} , js_t_delta = {js_t_delta/1000000}")

    # Store the JS.
    print(f"JS {js.position}")





    robot.arm.set_joint_positions
    # tui_cmd = getch()
    # print("\n==============")
    # print(f"User intput {tui_cmd} , value {ord(tui_cmd)}")

    # # Special stuff to exit
    # if ord(tui_cmd) == 3:
    #     break
    # if ord(tui_cmd) == 28:
    #     break

    last_js = js
    last_cycle_start = cycle_start
    time.sleep(0.001) # 10ms
