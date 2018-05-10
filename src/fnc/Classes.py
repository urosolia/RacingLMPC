import numpy as np

class ClosedLoopData():
    """Object collecting closed loop data points
    Attributes:
        updateInitialConditions: function which updates initial conditions and clear the memory
    """
    def __init__(self, dt, Time, v0):
        """Initialization
        Arguments:
            dt: discretization time
            Time: maximum time [s] which can be recorded
            v0: velocity initial condition
        """
        self.dt = dt
        self.Points = int(Time / dt)  # Number of points in the simulation
        self.u = np.zeros((self.Points, 2))  # Initialize the input vector
        self.x = np.zeros((self.Points + 1, 6))  # Initialize state vector (In curvilinear abscissas)
        self.x_glob = np.zeros((self.Points + 1, 6))  # Initialize the state vector in absolute reference frame
        self.SimTime = 0.0
        self.x[0,0] = v0
        self.x_glob[0,0] = v0

    def updateInitialConditions(self, x, x_glob):
        """Clears memory and resets initial condition
        x: initial condition is the curvilinear reference frame
        x_glob: initial condition in the inertial reference frame
        """
        self.x[0, :] = x
        self.x_glob[0, :] = x_glob

        self.x[1:, :] = 0*self.x[1:, :]
        self.x_glob[1:, :] = 0*self.x_glob[1:, :]


class LMPCprediction():
    """Object collecting the predictions and SS at eath time step
    """
    def __init__(self, N, n, d, TimeLMPC, numSS_Points, Laps):
        """
        Initialization:
            N: horizon length
            n, d: input and state dimensions
            TimeLMPC: maximum simulation time length [s]
            num_SSpoints: number used to buils SS at each time step
        """
        self.PredictedStates = np.zeros((N+1, n, TimeLMPC, Laps))
        self.PredictedInputs = np.zeros((N, d, TimeLMPC, Laps))

        self.SSused   = np.zeros((n , numSS_Points, TimeLMPC, Laps))
        self.Qfunused = np.zeros((numSS_Points, TimeLMPC, Laps))
