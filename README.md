# Learning Model Predictive Control (LMPC) for autonomous racing

The Learning Model Predictive Control (LMPC) is a data-driven control framework developed at UCB in the MPC lab. In this example, we implemented the LMPC for the autonomous racing problem. The controller drives several laps on race track and it learns from experience how to drive faster.


<p align="center">
<img src="https://github.com/urosolia/RacingLMPC/blob/master/src/ClosedLoop_multiLap.gif" width="500" />
</p>

In the above animation we see the vehicle's closed-loop trajectory (in black) for laps 5, 30, 31 and 32. At each time instant the LMPC leverages forecast to plan the vehicle trajectory (in red) few seconds into the future. This trajectory is planned to minimize the lap time, but it is constrained to land into the safe set (in green). This safe set is the domain of the approximation to the value function and it is updated after each lap using historical data.

### Prerequisites

The packeges needed for running the code can be installed using pip

```
pip install cvxopt
pip install osqp
pip install pathos
```

## Description

### The Plant
The vehicle is modelled using the dynamics signle track bicycle model and the tire forces are modelled using the Pacejka formula.

### The Path Following
1) Lap 1: a PID path following controller is used to drive the vehicle around the track.
2) Lap 2: the data from lap 1 are used to estimate a LTI model used to design a MPC for path following
3) Lap 3: the data from lap 1 are used to estimate a LTV model used to design a MPC for path following


## References

This code is based on the following:

* Ugo Rosolia and Francesco Borrelli. "Learning Model Predictive Control for Iterative Tasks. A Data-Driven Control Framework." In IEEE Transactions on Automatic Control (2017). [PDF](https://ieeexplore.ieee.org/document/8039204/)
* Ugo Rosolia and Francesco Borrelli. "Learning how to autonomously race a car: a predictive control approach." IEEE Transactions on Control Systems Technology (2019) [PDF](https://ieeexplore.ieee.org/abstract/document/8896988).
* Ugo Rosolia and Francesco Borrelli. "Learning Model Predictive Control for Iterative Tasks: A Computationally Efficient Approach for Linear System." IFAC-PapersOnLine 50.1 (2017). [PDF](https://arxiv.org/pdf/1702.07064.pdf)
