import numpy as np
from PredictiveControllers import MPC, LMPC, MPCParams

def initMPCParams(n, d, N, vt):
    # Buil the matrices for the state constraint in each region. In the region i we want Fx[i]x <= bx[b]
    Fx = np.array([[0., 0., 0., 0., 0., 1.],
                   [0., 0., 0., 0., 0., -1.]])

    bx = np.array([[2.],   # max ey
                   [2.]]), # max ey

    Fu = np.kron(np.eye(2), np.array([1, -1])).T
    bu = np.array([[0.5],   # -Min Steering
                   [0.5],   # Max Steering
                   [10.0],  # -Min Acceleration
                   [10.0]]) # Max Acceleration

    # Tuning Parameters
    Q = np.diag([1.0, 1.0, 1, 1, 0.0, 100.0]) # vx, vy, wz, epsi, s, ey
    R = np.diag([1.0, 10.0])                  # delta, a
    xRef   = np.array([vt, 0, 0, 0, 0, 0])
    Qslack = 1 * np.array([0, 50])
    
    mpcParameters    = MPCParams(n=n, d=d, N=N, Q=Q, R=R, Fx=Fx, bx=bx, Fu=Fu, bu=bu, xRef=xRef, slacks=True, Qslack=Qslack)
    mpcParametersLTV = MPCParams(n=n, d=d, N=N, Q=Q, R=R, Fx=Fx, bx=bx, Fu=Fu, bu=bu, xRef=xRef, slacks=True, Qslack=Qslack)       
    return mpcParameters, mpcParametersLTV

def initLMPCParams(map, N):
    # Buil the matrices for the state constraint in each region. In the region i we want Fx[i]x <= bx[b]
    Fx = np.array([[0., 0., 0., 0., 0., 1.],
                   [0., 0., 0., 0., 0., -1.]])

    bx = np.array([[map.halfWidth],   # max ey
                   [map.halfWidth]]), # max ey

    Fu = np.kron(np.eye(2), np.array([1, -1])).T
    bu = np.array([[0.5],   # -Min Steering
                   [0.5],   # Max Steering
                   [10.0],  # -Min Acceleration
                   [10.0]]) # Max Acceleration

   # Safe Set Parameters
    numSS_it = 4                  # Number of trajectories used at each iteration to build the safe set
    numSS_Points = 12*numSS_it    # Number of points to select from each trajectory to build the safe set

    Laps       = 40+numSS_it      # Total LMPC laps
    TimeLMPC   = 400              # Simulation time

    # Tuning Parameters
    QterminalSlack  = 500 * np.diag([1, 1, 1, 1, 1, 1])  # Cost on the slack variable for the terminal constraint
    Qslack  =  1 * np.array([5, 25])                           # Quadratic and linear slack lane cost
    Q_LMPC  =  0 * np.diag([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])     # State cost x = [vx, vy, wz, epsi, s, ey]
    R_LMPC  =  0 * np.diag([1.0, 1.0])                         # Input cost u = [delta, a]
    dR_LMPC =  5 * np.array([1.0, 10.0])                       # Input rate cost u
    n       = Q_LMPC.shape[0]
    d       = R_LMPC.shape[0]

    lmpcParameters    = MPCParams(n=n, d=d, N=N, Q=Q_LMPC, R=R_LMPC, dR=dR_LMPC, Fx=Fx, bx=bx, Fu=Fu, bu=bu, slacks=True, Qslack=Qslack)
    return numSS_it, numSS_Points, Laps, TimeLMPC, QterminalSlack, lmpcParameters