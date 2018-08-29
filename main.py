# ----------------------------------------------------------------------------------------------------------------------
# Licensing Information: You are free to use or extend these projects for
# education or reserach purposes provided that you provide clear attribution to UC Berkeley,
# including a reference to the papers describing the control framework:
# [1] Ugo Rosolia and Francesco Borrelli. "Learning Model Predictive Control for Iterative Tasks. A Data-Driven
#     Control Framework." In IEEE Transactions on Automatic Control (2017).
#
# [2] Ugo Rosolia, Ashwin Carvalho, and Francesco Borrelli. "Autonomous racing using learning model predictive control."
#     In 2017 IEEE American Control Conference (ACC)
#
# [3] Maximilian Brunner, Ugo Rosolia, Jon Gonzales and Francesco Borrelli "Repetitive learning model predictive
#     control: An autonomous racing example" In 2017 IEEE Conference on Decision and Control (CDC)
#
# [4] Ugo Rosolia and Francesco Borrelli. "Learning Model Predictive Control for Iterative Tasks: A Computationally
#     Efficient Approach for Linear System." IFAC-PapersOnLine 50.1 (2017).
#
# Attibution Information: Code developed by Ugo Rosolia
# (for clarifications and suggestions please write to ugo.rosolia@berkeley.edu).
#
# Code description: Simulation of the Learning Model Predictive Controller (LMPC). The main file runs:
# 1) A PID path following controller
# 2) A Model Predictive Controller (MPC) which uses a LTI model identified from the data collected with the PID in 1)
# 3) A MPC which uses a LTV model identified from the date collected in 1)
# 4) A LMPC for racing where the safe set and value function approximation are build using the data from 1), 2) and 3)
# ----------------------------------------------------------------------------------------------------------------------

import sys
sys.path.append('ControllersObject')
sys.path.append('Utilities')
from SysModel import Simulator
from PID import PID
from dataStructures import ClosedLoopDataObj, LMPCprediction
from PathFollowingLTVMPC import PathFollowingLTV_MPC
from PathFollowingLTI_MPC import PathFollowingLTI_MPC
from trackInitialization import Map, unityTestChangeOfCoordinates
from LMPC import ControllerLMPC #, PWAControllerLMPC
from LMPC_PWA import PWAControllerLMPC
from utilities import Regression
from plot import plotTrajectory, plotClosedLoopLMPC, animation_xy, animation_states, saveGif_xyResults, Save_statesAnimation
import numpy as np
import matplotlib.pyplot as plt
import pdb
import pickle

# ======================================================================================================================
# ============================ Choose which controller to run ==========================================================
# ======================================================================================================================
RunPID     = 0; plotFlag       = 0
RunMPC     = 0; plotFlagMPC    = 0
RunMPC_tv  = 0; plotFlagMPC_tv = 0
RunLMPC    = 1; plotFlagLMPC   = 1; animation_xyFlag = 1; animation_stateFlag = 0
runPWAFlag = 1; # uncomment importing pwa_cluster in LMPC.py
testCoordChangeFlag = 0;
plotOneStepPredictionErrors = 0;

# ======================================================================================================================
# ============================ Initialize parameters for path following ================================================
# ======================================================================================================================
dt         = 1.0/10.0        # Controller discretization time
Time       = 100             # Simulation time for PID
TimeMPC    = 100             # Simulation time for path following MPC
TimeMPC_tv = 100             # Simulation time for path following LTV-MPC
vt         = 0.8             # Reference velocity for path following controllers
v0         = 0.5             # Initial velocity at lap 0
N          = 12              # Horizon length
n = 6;   d = 2               # State and Input dimension

# Path Following tuning
Q = np.diag([1.0, 1.0, 1, 1, 0.0, 100.0]) # vx, vy, wz, epsi, s, ey
R = np.diag([1.0, 10.0])                  # delta, a

map = Map('oval') #3110_big')                            # Initialize the map
simulator = Simulator(map)                # Initialize the Simulator

# ======================================================================================================================
# ==================================== Initialize parameters for LMPC ==================================================
# ======================================================================================================================
TimeLMPC   = 250              # Simulation time
Laps       = 5+2              # Total LMPC laps

# Safe Set Parameter
LMPC_Solver = "OSQP"           # Can pick CVX for cvxopt or OSQP. For OSQP uncomment line 14 in LMPC.py
SysID_Solver = "scipy"         # Can pick CVX, OSQP or scipy. For OSQP uncomment line 14 in LMPC.py
numSS_it = 2                  # Number of trajectories used at each iteration to build the safe set
numSS_Points = 32 + N         # Number of points to select from each trajectory to build the safe set
numSS_Points_PWA = 8 + N
shift = 2 * N / 3                 # Given the closed point, x_t^j, to the x(t) select the SS points from x_{t+shift}^j

# Tuning Parameters
Qslack  = 50*np.diag([10, 1, 1, 1, 10, 1])          # Cost on the slack variable for the terminal constraint
Qlane   = 0.1 * 0.5 * 10 * np.array([50, 10]) # Quadratic slack lane cost
Q_LMPC  =  0 * np.diag([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # State cost x = [vx, vy, wz, epsi, s, ey]
R_LMPC  =  1 * np.diag([1.0, 1.0])                      # Input cost u = [delta, a]
dR_LMPC =  2 * 5 * np.array([1.0, 1.0])                     # Input rate cost u (2,1)
aConstr = np.array([0.7, 2.0])

# Initialize LMPC simulator
LMPCSimulator = Simulator(map, 1, 1)

# ======================================================================================================================
# ======================================= PID path following ===========================================================
# ======================================================================================================================
print("Starting PID")
if RunPID == 1:
    ClosedLoopDataPID = ClosedLoopDataObj(dt, Time , v0)
    PIDController = PID(vt)
    simulator.Sim(ClosedLoopDataPID, PIDController)

    file_data = open(sys.path[0]+'/data/ClosedLoopDataPID.obj', 'wb')
    pickle.dump(ClosedLoopDataPID, file_data)
    file_data.close()
else:
    file_data = open(sys.path[0]+'/data/ClosedLoopDataPID.obj', 'rb')
    ClosedLoopDataPID = pickle.load(file_data)
    file_data.close()
print("===== PID terminated")

# ======================================================================================================================
# ======================================  LINEAR REGRESSION ============================================================
# ======================================================================================================================
print("Starting MPC")
lamb = 0.0000001
A, B, Error = Regression(ClosedLoopDataPID.x, ClosedLoopDataPID.u, lamb)

if RunMPC == 1:
    ClosedLoopDataLTI_MPC = ClosedLoopDataObj(dt, TimeMPC, v0)
    Controller_PathFollowingLTI_MPC = PathFollowingLTI_MPC(A, B, Q, R, N, vt, Qlane)
    simulator.Sim(ClosedLoopDataLTI_MPC, Controller_PathFollowingLTI_MPC)

    file_data = open(sys.path[0]+'/data/ClosedLoopDataLTI_MPC.obj', 'wb')
    pickle.dump(ClosedLoopDataLTI_MPC, file_data)
    file_data.close()
else:
    file_data = open(sys.path[0]+'/data/ClosedLoopDataLTI_MPC.obj', 'rb')
    ClosedLoopDataLTI_MPC = pickle.load(file_data)
    file_data.close()
print("===== MPC terminated")

# ======================================================================================================================
# ===================================  LOCAL LINEAR REGRESSION =========================================================
# ======================================================================================================================
print("Starting TV-MPC")
if RunMPC_tv == 1:
    ClosedLoopDataLTV_MPC = ClosedLoopDataObj(dt, TimeMPC_tv, v0)
    Controller_PathFollowingLTV_MPC = PathFollowingLTV_MPC(Q, R, N, vt, ClosedLoopDataPID.x, 
                                                      ClosedLoopDataPID.u, dt, map, "OSQP")
    simulator.Sim(ClosedLoopDataLTV_MPC, Controller_PathFollowingLTV_MPC)

    file_data = open(sys.path[0]+'/data/ClosedLoopDataLTV_MPC.obj', 'wb')
    pickle.dump(ClosedLoopDataLTV_MPC, file_data)
    file_data.close()
else:
    file_data = open(sys.path[0]+'/data/ClosedLoopDataLTV_MPC.obj', 'rb')
    ClosedLoopDataLTV_MPC = pickle.load(file_data)
    file_data.close()
print("===== TV-MPC terminated")
# ======================================================================================================================
# ==============================  LMPC w\ LOCAL LINEAR REGRESSION ======================================================
# ======================================================================================================================
print("Starting LMPC")
ClosedLoopLMPC = ClosedLoopDataObj(dt, TimeLMPC, v0)
LMPCSimulator = Simulator(map, 1, 1)

if runPWAFlag == 1:
    LMPCOpenLoopData = LMPCprediction(N, n, d, TimeLMPC, numSS_Points_PWA, Laps)
    LMPController = PWAControllerLMPC(2, numSS_Points_PWA, numSS_it, N, Qslack, Q_LMPC, R_LMPC, dR_LMPC, n, d, shift, dt, map, Laps, TimeLMPC, LMPC_Solver)
else:
    LMPCOpenLoopData = LMPCprediction(N, n, d, TimeLMPC, numSS_Points, Laps)
    LMPController = ControllerLMPC(numSS_Points, numSS_it, N, Qslack, Qlane, Q_LMPC, R_LMPC, dR_LMPC, 
                                    n, d, shift, dt, map, Laps, TimeLMPC, LMPC_Solver,
                                    SysID_Solver, True, 0, 0, aConstr)


 
        
LMPController.addTrajectory(ClosedLoopDataPID)
LMPController.addTrajectory(ClosedLoopDataLTV_MPC)

x0           = np.zeros((1,n))
x0_glob      = np.zeros((1,n))
x0[0,:]      = ClosedLoopLMPC.x[0,:]
x0_glob[0,:] = ClosedLoopLMPC.x_glob[0,:]

if RunLMPC == 1:
    for it in range(2, Laps):

        ClosedLoopLMPC.updateInitialConditions(x0, x0_glob)
        LMPCSimulator.Sim(ClosedLoopLMPC, LMPController, LMPCOpenLoopData)
        LMPController.addTrajectory(ClosedLoopLMPC)

        if LMPController.feasible == 0 or ClosedLoopLMPC.x[ClosedLoopLMPC.SimTime, 4] < 0.5 *  map.TrackLength:
            break
        else:
            # Reset Initial Conditions
            x0[0,:]      = ClosedLoopLMPC.x[ClosedLoopLMPC.SimTime, :] - np.array([0, 0, 0, 0, map.TrackLength, 0])
            x0_glob[0,:] = ClosedLoopLMPC.x_glob[ClosedLoopLMPC.SimTime, :]

    file_data = open(sys.path[0]+'/data/LMPController.obj', 'wb')
    pickle.dump(ClosedLoopLMPC, file_data)
    pickle.dump(LMPController, file_data)
    pickle.dump(LMPCOpenLoopData, file_data)
    file_data.close()
else:
    file_data = open(sys.path[0]+'/data/LMPController.obj', 'rb')
    ClosedLoopLMPC = pickle.load(file_data)
    LMPController  = pickle.load(file_data)
    LMPCOpenLoopData  = pickle.load(file_data)
    file_data.close()

print("===== LMPC terminated")
# ======================================================================================================================
# ========================================= PLOT TRACK/PREDICTIONS =====================================================
# ======================================================================================================================
for i in range(0, LMPController.it):
    print("Lap time at iteration ", i, " is ", LMPController.Qfun[0, i]*dt, "s")

plot_iteration = LMPController.it-1 # final iteration

print("===== Start Plotting")
if plotFlag == 1:
    plotTrajectory(map, ClosedLoopDataPID.x, ClosedLoopDataPID.x_glob, ClosedLoopDataPID.u)

if plotFlagMPC == 1:
    plotTrajectory(map, ClosedLoopDataLTI_MPC.x, ClosedLoopDataLTI_MPC.x_glob, ClosedLoopDataLTI_MPC.u)

if plotFlagMPC_tv == 1:
    plotTrajectory(map, ClosedLoopDataLTV_MPC.x, ClosedLoopDataLTV_MPC.x_glob, ClosedLoopDataLTV_MPC.u)

if plotFlagLMPC == 1:
    plotClosedLoopLMPC(LMPController, map)

if animation_xyFlag == 1:
    # animation_xy(map, LMPCOpenLoopData, LMPController, LMPController.it-2)
    animation_xy(map, LMPCOpenLoopData, LMPController, plot_iteration)
    # saveGif_xyResults(map, LMPCOpenLoopData, LMPController, 6)

if animation_stateFlag == 1:
    animation_states(map, LMPCOpenLoopData, LMPController, plot_iteration)
    # Save_statesAnimation(map, LMPCOpenLoopData, LMPController, 5)

if testCoordChangeFlag == 1:
    unityTestChangeOfCoordinates(map, ClosedLoopDataPID)
    unityTestChangeOfCoordinates(map, ClosedLoopDataLTI_MPC)
    unityTestChangeOfCoordinates(map, ClosedLoopLMPC)

if plotOneStepPredictionErrors == 1:
    it=LMPController.it-1
    print(LMPController.it)
    onestep_errors = []
    onestep_norm_errors = []
    #for i in range(1, int(LMPController.TimeSS[it])):
    for i in range(1, int(LMPCOpenLoopData.PredictedStates.shape[2])):
        if np.any(LMPController.SS[i, :, it] < 10000):
            current_state = LMPController.SS[i, :, it]
            current_pos = LMPController.SS[i, 4:6, it]
            predicted_trajectory = LMPCOpenLoopData.PredictedStates[:, :, i-1, it]
            predicted_state = predicted_trajectory[1,:]
            #print(predicted_state, current_state) # 0,10,000
            onestep_errors.append(predicted_state-current_state)
            onestep_norm_errors.append(np.linalg.norm(predicted_state-current_state))

    plt.figure();
    onestep_errors = np.array(onestep_errors)
    state_names = ['vx', 'vy', 'wz', 'epsi', 's', 'ey']
    for state in range(onestep_errors.shape[1]):
        if state == 0:
            plt.title('One Step Prediction Error')
        plt.subplot(onestep_errors.shape[1], 1, state+1)
        plt.plot(onestep_errors[:,state])
        plt.ylabel(state_names[state])

plt.show()



