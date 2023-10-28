#! /usr/bin/env python3
import time

from sensor_msgs.msg import JointState
import signal

import asyncio.events
from threading import Event

from interbotix_xs_modules.xs_robot.arm import InterbotixManipulatorXS

import robot_log_storage

int_event = Event()
def int_handle (signal , frame):
    print("Sig Int Catched")
    int_event.set()

signal.signal(signal.SIGINT,int_handle)

def str_list_float(print_list):
    
   return  [ f"{num:.2f} " for num in print_list]






# On each loop: check if js is updated, log js if so.
# Check if there are move currently going on. if not, start a new move. and log the move info, with current cycle's js serial number
# Check if a commanded joint move is done. if so, also log the move finished. but anchor to a js serial number

# Interbotix have no checking on if a move is done. We need to do our own tolerance check.

# use the python time monotonic_ns to be current cycle serial number.
# The robot object is what you use to control the robot
robot = InterbotixManipulatorXS("px100", "arm", "gripper")

last_js  = robot.core.joint_states

last_cycle_start = time.monotonic_ns()

log_store = robot_log_storage.RobotLogStorage(last_js.name)

while True:
    # These to only get saved into last_* if this cycle is meaningful.
    js = robot.core.joint_states
    cycle_start = time.monotonic_ns()
    if js.header == last_js.header:
        # There are totally nothing to do if JS is not updated.
        print("Same js, no change!")
        continue

    if js.header.stamp == last_js.header.stamp:
        raise ValueError(f"\ncurrent header {js.header} , \nlast_js header{last_js.header}")
    # New js.
    js_t_delta = js.header.stamp.nanosec - last_js.header.stamp.nanosec
    py_t_delta = cycle_start - last_cycle_start

    js_t_delta_s = js_t_delta / 1e9
    print(f"\n ...\py_ncycle_delta_ms {(py_t_delta) /1e6} , js_t_delta_ms = {js_t_delta/1e6}")

    # Store the JS.  
    print(f"JS {str_list_float(js.position)}")
    print(f"JS V {str_list_float(js.velocity)}")

    log_store.log_js(js)

    robot.arm.set_joint_positions
    
    last_js = js
    last_cycle_start = cycle_start

    if Event.wait(int_event,timeout=1):
        print("Finish looping due to int flag")
        break