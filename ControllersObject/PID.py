import numpy as np
import datetime


class PID:
    """Create the PID controller used for path following at constant speed
    Attributes:
        solve: given x0 computes the control action
    """
    def __init__(self, vt, noise=[0,0], mode="simulations"):
        """Initialization
        Arguments:
            vt: target velocity
        """
        self.vt = vt
        self.uPred = np.zeros([1,2])

        startTimer = datetime.datetime.now()
        endTimer = datetime.datetime.now(); deltaTimer = endTimer - startTimer
        self.solverTime = deltaTimer
        self.linearizationTime = deltaTimer
        self.feasible = 1
        self.integral = np.array([0.0, 0.0])
        self.noise = noise
        self.mode = mode

    def solve(self, x0):
        """Computes control action
        Arguments:
            x0: current state position
        """
        vt = self.vt
        if self.mode == "simulations":
            Steering = - 0.5 * 2.0 * x0[5] - 2 * 0.5 * x0[3] - 0.001 * self.integral[0]
            Accelera = 0.5 * 1.5 * (vt - x0[0]) + 0.1 * self.integral[1]
        else:
            Steering = - 0.5 * x0[5] - x0[3] - 0.1 * self.integral[0]
            Accelera = 1.5 * (vt - x0[0]) + 0.1 * self.integral[1]

        self.integral[0] = self.integral[0] +  0.1 * x0[5] + 0.1 * x0[3]
        self.integral[1] = self.integral[1] +  (vt - x0[0])

        self.uPred[0, 0] = self.truncate(Steering, 0.3) + self.truncate( np.random.randn() * 0.25 * self.noise[0], 0.3)
        self.uPred[0, 1] = self.truncate(Accelera, 2.0) + self.truncate( np.random.randn() * 0.1 * self.noise[1], 0.3)


    def truncate(self, val, bound):
        return np.maximum(-bound, np.minimum(val, bound))



    # def solve(self, x0):
    #     """Computes control action
    #     Arguments:
    #         x0: current state position
    #     """
    #     vt = self.vt
    #     self.uPred[0, 0] = - 0.6 * x0[5] - 0.9 * x0[3] + np.maximum(-0.9, np.minimum(np.random.randn() * 0.25, 0.9))
    #     self.uPred[0, 1] = 1.5 * (vt - x0[0]) + np.maximum(-0.2, np.minimum(np.random.randn() * 0.10, 0.2))
