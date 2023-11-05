#! /usr/bin/env python3
import time
import queue
from typing import Optional

from sensor_msgs.msg import JointState
import signal

from threading import Event


from interbotix_xs_modules.xs_robot.arm import InterbotixManipulatorXS

import robot_log_storage
import threading

from rclpy.qos import QoSDurabilityPolicy, QoSProfile ,QoSHistoryPolicy
from rclpy.node import Node as RosNode
from rclpy.wait_for_message import wait_for_message
import rclpy
import rclpy.time

from interbotix_xs_msgs.msg import JointGroupCommand
import random

int_event = Event()
def int_handle (signal , frame):
    print("Sig Int Catched")
    int_event.set()

signal.signal(signal.SIGINT,int_handle)

def str_list_float(print_list):
    return '[' + ",".join([f"{num:.3f} " for num in print_list]) + ']'

target_pos_list = [
    [-0.01 , -0.69 , -0.67, -0.31],
    [-0.009 ,0.077 ,0.272 ,1.276]
]

END_MOVE_MARGIN = 0.02

def all_in_tolerance(value :list[float], target:list[float] , margin: float):
    i =0
    for v , t in zip(value,target):

        if abs(v-t) > margin:
            return False
        ++i
    return True

class RobotListener():
 
    def __init__(self):
        self.ros_node = RosNode("receiver_node")

        qos = QoSProfile(history=QoSHistoryPolicy.KEEP_LAST,
                         depth=10,
                         durability=QoSDurabilityPolicy.VOLATILE)

        self.ros_node.create_subscription(JointState,"/px100/joint_states" , self.joint_state_callback,qos)

        self.jcmd_pub =  self.ros_node.create_publisher(
            JointGroupCommand, f'/px100/commands/joint_group', 10
        )



        self.js_queue :queue.Queue[JointState] = queue.Queue()
        self.last_callback_py_ns = time.monotonic_ns()

    def spin_in_thread(self):
        self.ros_spin_thread = threading.Thread(target=rclpy.spin , args = [self.ros_node])
        self.ros_spin_thread.start()

    def cmd_pos(self, pos_list):

        self.jcmd_pub.publish(JointGroupCommand(name='arm', cmd=pos_list))
        print(f"Ros msg sent {JointGroupCommand(name='arm', cmd=pos_list)}")

    def joint_state_callback(self,msg:JointState):
        called_time = time.monotonic_ns()
        dt_ns = called_time - self.last_callback_py_ns
        # print(f"Callback-happened! dt {dt_ns/1e6 :.3f} ms")

        if dt_ns / 1e6 > 120:
            print(f"!!!!!!!!!!!!! callback happened toooooo late !")
        self.last_callback_py_ns = called_time
        self.js_queue.put(msg)

def main():
    # On each loop: check if js is updated, log js if so.
    # Check if there are move currently going on. if not, start a new move. and log the move info, with current cycle's js serial number
    # Check if a commanded joint move is done. if so, also log the move finished. but anchor to a js serial number

    # Fuck interbotix_xs_modules. It's hopeless
    # use the python time monotonic_ns to be current cycle serial number.
    # The robot object is what you use to control the robot


    rclpy.init()

    # robot = InterbotixManipulatorXS("px100", "arm", "gripper" , moving_time=0 , accel_time=0)
    # robot.arm.set_trajectory_time(0.5,1)

    # robot.arm.
    robot_listener = RobotListener()
    robot_listener.spin_in_thread()
    print("Waiting for a initial msg")


    print("Got init js")
    last_js  = robot_listener.js_queue.get()
    log_store = robot_log_storage.RobotLogStorage(last_js.name)
    last_cycle_start = time.monotonic_ns()

    cmd_serial = 0
    last_target = None
    current_cmd : Optional [ robot_log_storage.CommandEvent] = None

    for pos in target_pos_list:
        print(f"Possible cmd pos {pos}")

    cmd_sent_time = time.monotonic_ns()

    while True:
        # We want to loop a lot lot more faster then publish rate
        if Event.wait(int_event,timeout=0.0001):
            print("Finish looping due to int flag")
            break


        cycle_start = time.monotonic_ns()
        if robot_listener.js_queue.empty():
            # Totally natural in high speed async spinning
            continue

        if robot_listener.js_queue.qsize() >1 :
            # totally possible when things are threaded
            print(f"{robot_listener.js_queue.qsize()} msgs on queue! ROS is quicker!")

        js = robot_listener.js_queue.get_nowait()
        if js.header == last_js.header:
            # There are totally nothing to do if JS is not updated.
            print("Same js header, no change! This is not possible!")
            print(f"last {last_js.header} , new {js.header}")
            raise RuntimeError
        if js.header.stamp == last_js.header.stamp:
            raise ValueError(f"\ncurrent header {js.header} , \nlast_js header{last_js.header}")

        # New js.

        js_t_delta = rclpy.time.Time.from_msg(
            js.header.stamp).nanoseconds - rclpy.time.Time.from_msg(
                last_js.header.stamp).nanoseconds
        py_t_delta = cycle_start - last_cycle_start


        # Store the JS.
        # js_t_delta_s = js_t_delta / 1e9
        print(f"py_ncycle_delta_ms {(py_t_delta) /1e6} , js_t_delta_ms = {js_t_delta/1e6}")
        print(f"JS {str_list_float(js.position)}")
        print(f"JS V {str_list_float(js.velocity)}")
        print(f"JS eff {str_list_float(js.effort)}")

        dont_send = True
        if dont_send:
            last_js = js
            last_cycle_start = cycle_start
            continue


        if js_t_delta/1e6 < 8 or js_t_delta/1e6 > 12 :
            print(f"!!!!!!!!!!!! \nROS time is really bad! \n")

            print(f"last_js {last_js}")
            print(f"new_js {js}")
        log_cmd_event = None
        if current_cmd is None:
            # Pick a command and send it !

            new_cmd = random.choice(target_pos_list)
            while new_cmd == last_target:
                new_cmd = random.choice(target_pos_list)
            last_target = new_cmd
            print(f"Going to send target pos {new_cmd}")
            current_cmd = robot_log_storage.CommandEvent(
                cmd_serial, js.position, new_cmd, event_type=robot_log_storage.EventType.MOVE_START)
            log_cmd_event = current_cmd

            robot_listener.cmd_pos(new_cmd)
            cmd_sent_time = time.monotonic_ns()

        elif current_cmd.event_type == robot_log_storage.EventType.MOVE_START:
            # A move is in place, check for it's termination.
            if all_in_tolerance(js.position,current_cmd.end_js ,END_MOVE_MARGIN ):
                current_cmd.event_type =robot_log_storage.EventType.MOVE_END
                log_cmd_event = current_cmd
                move_time = (time.monotonic_ns() -cmd_sent_time) / 1e9
                print(f"Robot at target! after {move_time:.5f} ")

        elif current_cmd.event_type == robot_log_storage.EventType.MOVE_END:
            # This allows a extra cycle before starting the next one.
            current_cmd = None


        log_store.log_js(js,log_cmd_event)
        last_js = js
        last_cycle_start = cycle_start


if __name__ == "__main__":
    main()
