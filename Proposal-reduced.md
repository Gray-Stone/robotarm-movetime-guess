# EE-475 Project

Group member: Qingyuan (Leo) Chen

Topic: 
**Estimate trajectory time from arbitrary start-end configuration of specific after market robot arm.**

## Background

At current times, robot arms are a well established product. Specially for industrial robot. These after market robot arm come with a full controller that handle everything about the arm. User is expected to use the full arm-controller package and simply command the arm where to go. The controller handles all the complicated math behind it. 

Manufactures rarely provide any dynamics data about the arm. When using these after market arm, although not needing dynamics data for control algorism, simple estimation of the duration of an arbitrary motion becomes impossible.

## Project's goal

The goal of the project is to bypass these missing information about the robot arm's dynamics by using machine learning on observable movement data collected from using the arm in practice. For a specific arm, assuming the trajectory is deterministic fro the same start-end configuration, the duration of a move should be predicted from a given pair of start-end joint configuration.

### Data collection.

Since robot arms very greatly on their dynamics, the data is to be collected on the same robot arm that the model will apply for.

I will find a after market robot arm that comes with motion control (like most industrial robot arm). 
By moving the robot arm around randomly with different joint configuration, timing data over different moves can be collected. The arm selection needs to be close enough to industrial robot arm for low noise and high repeatability.

### Training the model.

Linear regression will be used to create the initial version of the model. Could extend to other methods if necessary.

### Extending the model's ability. 

For future development and improvement: 

The dynamics of a robot arm changes greatly when different tooling is attached. The model could extends its input with parameters of the tooling.