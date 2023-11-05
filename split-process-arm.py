#! /usr/bin/env python3
import multiprocessing
import threading
import time
import queue
import signal

# Custom helpers
import robot_log_storage

# Ros imports
from rclpy.qos import QoSDurabilityPolicy, QoSProfile ,QoSHistoryPolicy
from rclpy.node import Node as RosNode
from rclpy.wait_for_message import wait_for_message
import rclpy
import rclpy.time
from sensor_msgs.msg import JointState

# End of importing



def main():

    terminate_event = multiprocessing.Event()

    robot_process = multiprocessing.Process(target=wave_arm_loop , name="robot-side",args = (terminate_event,))
    robot_process.start()
    time.sleep(1)

    rclpy.init()
    arm_logger = ArmDataLogger()
    terminate_event = multiprocessing.Event()

    # def int_handle (signal , frame):
    #     print("Sig Int Catched")
    #     terminate_event.set()
    #     print("try rclpy shutdown")
    #     rclpy.shutdown()

    #     # robot_process.terminate()
    # signal.signal(signal.SIGINT,int_handle)

    
    arm_logger.spin_in_thread()
    arm_logger.main_loop(terminate_event)

    print("Seems all finished ! ")


# Robot controlling part.


def wave_arm_loop(terminate_event):
    waver =BotWaver()
    while not terminate_event.is_set():
        waver.wave_arm_once()
        pass

class BotWaver():
    target_pos_list = [
        [-0.01 , -0.69 , -0.67, -0.31],
        [-0.009 ,0.077 ,0.272 ,1.276]
    ]
    END_MOVE_MARGIN = 0.02

    def __init__(self) -> None:
        from interbotix_xs_modules.xs_robot.arm import InterbotixManipulatorXS
        robot = InterbotixManipulatorXS("px100", "arm", "gripper")
        robot.arm.set_trajectory_time(0.5,1)


    def wave_arm_once(self):
        time.sleep(1)
        print(f"wave arm still running")

# Data acquiring part

class ArmDataLogger():

    def __init__(self) -> None:
        self.ros_node = RosNode("arm_logger")
        qos = QoSProfile(history=QoSHistoryPolicy.KEEP_LAST,
                         depth=10,
                         durability=QoSDurabilityPolicy.VOLATILE)

        
        print(f"Waiting for init msg")
        worked , init_js = wait_for_message(JointState,self.ros_node,"/px100/joint_states",time_to_wait=5)
        if not worked:
            raise RuntimeError("No initial js msg")
        print(f"Robot Starting at {init_js.position}")

        self.js_queue :queue.Queue[JointState] = queue.Queue()
        self.last_js:JointState = init_js
        self.last_cycle_start = time.monotonic()
        self.last_callback_py_ns = time.monotonic_ns()
        self.ros_node.create_subscription(JointState,"/px100/joint_states" , self.js_callback,qos)



    def main_loop(self,terminate_event):
        with self.js_queue.mutex:
            self.js_queue.queue.clear()

        while not terminate_event.wait(timeout=0.0003):
            if terminate_event.is_set():
                print("Detected terminate event set")
                break
            cycle_start = time.monotonic()
            if self.js_queue.empty():
                continue
            js = self.js_queue.get_nowait()
        
            if js.header == self.last_js.header:
                print("Same js header, no change! This is not possible!")
                print(f"last {self.last_js.header} , new {js.header}")
                raise RuntimeError
            if js.header.stamp == self.last_js.header.stamp:
                raise ValueError(f"\ncurrent header {js.header} , \nlast_js header{self.last_js.header}")
            self.check_timing(cycle_start,js)

            self.last_js = js
            self.last_cycle_start = cycle_start


    def check_timing(self,cycle_start:float , js:JointState):
        js_t_delta = rclpy.time.Time.from_msg(
        js.header.stamp).nanoseconds - rclpy.time.Time.from_msg(
                self.last_js.header.stamp).nanoseconds
        py_t_delta = cycle_start - self.last_cycle_start
        print(f"py_ncycle_delta_ms {(py_t_delta)*1e3} , js_t_delta_ms = {js_t_delta/1e6}")
        # print(f"JS {str_list_float(js.position)}")
        # print(f"JS V {str_list_float(js.velocity)}")
        # print(f"JS eff {str_list_float(js.effort)}")
        if js_t_delta/1e6 < 8 or js_t_delta/1e6 > 12 :
            print(f"!!!!!!!!!!!! \nROS time is really bad! \n")

            print(f"last_js {self.last_js}")
            print(f"new_js {js}")

        
            



    def js_callback(self,msg:JointState):
        # This should be short, and minimal of other multi-threading objects.
        # 
        called_time = time.monotonic_ns()
        dt_ns = called_time - self.last_callback_py_ns
        # print(f"Callback-happened! dt {dt_ns/1e6 :.3f} ms")

        if dt_ns / 1e6 > 120:
            print(f"!!!!!!!!!!!!! callback happened toooooo late !")
        self.last_callback_py_ns = called_time

        if not self.js_queue.empty():
            print(f"!!!! js queue is stacking! ROS is quicker")
        self.js_queue.put(msg)

    def spin_in_thread(self):
        self.ros_spin_thread = threading.Thread(target=rclpy.spin , args = [self.ros_node])
        self.ros_spin_thread.start()


# Other helper 

def str_list_float(print_list):
    return '[' + ",".join([f"{num:.3f} " for num in print_list]) + ']'

def all_in_tolerance(value :list[float], target:list[float] , margin: float):
    i =0
    for v , t in zip(value,target):

        if abs(v-t) > margin:
            return False
        ++i
    return True


# main 

if __name__ == "__main__":
    main()
