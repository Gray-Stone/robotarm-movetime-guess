# EE-475 Project

Group member: Qingyuan (Leo) Chen

Topic: 
**Estimate trajectory time from arbitrary start-end configuration of specific after market robot arm.**

## Background

At current times, robot arms are a well established product. Specially for industrial robot.

These after market robot arm come with a full controller that handle everything about the arm. User is expected to use the full arm-controller package and simply command the arm where to go. The controller handles all the complicated math behind it. 

The performance of a robot arm is limited by its physical property. The dynamic limit of the the arm is not a simple measure-able property since it change as a function of joint configuration. This function of dynamics is dependent on the dynamic property (mass, cg, inertia, etc) of the physical link, joints, as well as the actuators (torque limit, speed-torque relationship, etc). However, manufactures rarely provide any of these data and regard them as internal trade secrete. When using these after market arm, a simple task such as estimating the duration of an arbitrary motion becomes very difficult due to the lack of a arm's dynamic information.

## Project's goal

The goal of the project is to bypass these missing information about the robot arm's dynamics by using machine learning on observable movement data collected from using the arm in practice. I would like to build a prediction model (for a specific arm) that estimate the duration of a move given the start-end joint-configuration of the move (assuming the trajectory is deterministic from start-end config). The model will be designed to do estimation on a specific robot.

### Data collection.

Since robot arms very greatly on their dynamics, the data is to be collected on the same robot arm that the model will apply for.

We will find a after market robot arm that comes with motion control (like most industrial robot arm). We will wave the robot arm around randomly between different joint configuration to collection timing data of different moves. The arm selection needs to be close enough to industrial robot arm for low enough noise in repeatability.

### Training the model.

The data collected from previous steps will be used to train the model. We can start with a simple linear regression then extend to other methods.

### Extending the model's ability. 

For future development and improvement: 

The performance of a robot arm changes greatly when different end of arm tooling (EOAT) is attached (or grabbing different object). It will be very useful if the model could be extended to make prediction on different EOAT (given the dynamic knowledge of the EOAT, like mass, cg, inertia).

