# Learning Model Predictive Control (LMPC) for autonomous racing

The Learning Model Predictive Control (LMPC) is a data-driven control framework developed at UCB in the MPC lab. In this example, we implemented the LMPC for the autonomous. The controller drives several laps of a race track and it learns from experience to improve the lap time.

### Prerequisites

The packeges needed for running the code can be installed using pip

```
pip install cvxopt
pip install pathos
```

## Description

### The Plant
The vehicle is modelled using the dynamics signle track bicycle model and the tire forces are modelled using the Pacejka formula.

### The Path Following
1) Lap 1: a PID path following controller is used to drive the vehicle around the track.
2) Lap 2: the data from lap 1 are used to estimate a LTI model used to design a MPC for path following
3) Lap 3: the data from lap 1 are used to estimate a LTV model used to design a MPC for path following

### The Learning Model Predictive Controller
The data from the previous laps are used to build a safety set and a terminal cost which are used to initialize the LMPC. Futhermor, the LMPC uses a LTV model identified from data.

## Results

The LMPC used forecast to plan the vehicle trajectory looking few seconds into the future. This trajectory is planned in order to minimize the lap time, but it is constrained to land in a set of safe states.
In the animation below we see the closed-loop trajectory (in black) of the vehicle after 5 laps of learning. The LMPC plans an open-loop trajectory (in red) which minimizes the lap time and lands in the safety set (in blue).

<p align="center">
<img src="https://github.com/urosolia/RacingLMPC/blob/master/src/ClosedLoop.gif" width="500" />
</p>

## References

This code is based on the following:

* Ugo Rosolia and Francesco Borrelli. "Learning Model Predictive Control for Iterative Tasks. A Data-Driven Control Framework." In IEEE Transactions on Automatic Control (2017). [PDF](https://ieeexplore.ieee.org/document/8039204/)
* Ugo Rosolia, Ashwin Carvalho, and Francesco Borrelli. "Autonomous racing using learning model predictive control." In IEEE 2017 American Control Conference (ACC). [PDF](https://ieeexplore.ieee.org/abstract/document/7963748/)
* Maximilian Brunner, Ugo Rosolia, Jon Gonzales and Francesco Borrelli "Repetitive learning model predictive control: An autonomous racing example" In 2017 IEEE Conference on Decision and Control (CDC). [PDF](https://ieeexplore.ieee.org/abstract/document/8264027/)
* Ugo Rosolia and Francesco Borrelli. "Learning Model Predictive Control for Iterative Tasks: A Computationally Efficient Approach for Linear System." IFAC-PapersOnLine 50.1 (2017). [PDF](https://www.sciencedirect.com/science/article/pii/S2405896317306523)
