import numpy as np
import datetime
import sys
import os
import rospy
import pickle
from trackInitialization import Map
sys.path.append('../ControllersObject')
import LMPC_PWA 


# ============================== parameters =========================================

Nc = 2 # number of clusters
dt         = 1.0/10.0        # Controller discretization time
Time       = 100             # Simulation time for PID
TimeMPC    = 100             # Simulation time for path following MPC
TimeMPC_tv = 100             # Simulation time for path following LTV-MPC
vt         = 0.8             # Reference velocity for path following controllers
v0         = 0.5             # Initial velocity at lap 0
N          = 12              # Horizon length
n = 6;   d = 2               # State and Input dimension
TimeLMPC   = 400              # Simulation time
Laps       = 5+2              # Total LMPC laps
map = Map('oval')                            # Initialize the map
# Safe Set Parameter
LMPC_Solver = "OSQP"           # Can pick CVX for cvxopt or OSQP. For OSQP uncomment line 14 in LMPC.py
numSS_it = 2                  # Number of trajectories used at each iteration to build the safe set
numSS_Points = 32 + N         # Number of points to select from each trajectory to build the safe set
numSS_Points_PWA = 8 + N
shift = N / 2                 # Given the closed point, x_t^j, to the x(t) select the SS points from x_{t+shift}^j
# Tuning Parameters
Qslack  = 50*np.diag([10, 1, 1, 1, 10, 1])          # Cost on the slack variable for the terminal constraint
Q_LMPC  =  0 * np.diag([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # State cost x = [vx, vy, wz, epsi, s, ey]
R_LMPC  =  1 * np.diag([1.0, 1.0])                      # Input cost u = [delta, a]
dR_LMPC =  5 * np.array([1.0, 1.0])                     # Input rate cost u (2,1)


# ============================== make controller, load data =========================================

LMPController = LMPC_PWA.PWAControllerLMPC(Nc, numSS_Points_PWA, numSS_it, N, Qslack, Q_LMPC, R_LMPC, dR_LMPC, 
	                              n, d, shift, dt, map, Laps, TimeLMPC, LMPC_Solver)

# load data
homedir = os.path.expanduser("~")    
file_data = open(homedir+'/barc_data/'+'/ClosedLoopDataTI_MPC.obj', 'rb')
ClosedLoopDataTI_MPC = pickle.load(file_data)
file_data.close()
LMPController.addTrajectory(ClosedLoopDataTI_MPC)

# estimate model
LMPController._estimate_pwa(forceEstimate=True, verbose=True)