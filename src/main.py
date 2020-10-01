# ----------------------------------------------------------------------------------------------------------------------
# Licensing Information: You are free to use or extend these projects for
# education or reserach purposes provided that you provide clear attribution to UC Berkeley,
# including a reference to the papers describing the control framework:
# [1] Ugo Rosolia and Francesco Borrelli. "Learning Model Predictive Control for Iterative Tasks. A Data-Driven
#     Control Framework." In IEEE Transactions on Automatic Control (2017).
#
# [2] Ugo Rosolia and Francesco Borrelli "Learning how to autonomously race a car: a predictive control approach" 
#     In 2017 IEEE Conference on Decision and Control (CDC)
#
# [3] Ugo Rosolia and Francesco Borrelli. "Learning Model Predictive Control for Iterative Tasks: A Computationally
#     IEEE Transactions on Control Systems Technology (2019).
#
# Attibution Information: Code developed by Ugo Rosolia
# (for clarifiactions and suggestions please write to ugo.rosolia@berkeley.edu).
#
# Code description: Simulation of the Learning Model Predictive Controller (LMPC). The main file runs:
# 1) A PID path following controller
# 2) A Model Predictive Controller (MPC) which uses a LTI model identified from the data collected with the PID in 1)
# 3) A MPC which uses a LTV model identified from the date collected in 1)
# 4) A LMPC for racing where the safe set and value function approximation are build using the data from 1), 2) and 3)
# ----------------------------------------------------------------------------------------------------------------------
import sys
sys.path.append('fnc/simulator')
sys.path.append('fnc/controller')
sys.path.append('fnc')
import matplotlib.pyplot as plt
from plot import plotTrajectory, plotClosedLoopLMPC, animation_xy, animation_states, saveGif_xyResults
from initControllerParameters import initMPCParams, initLMPCParams
from PredictiveControllers import MPC, LMPC, MPCParams
from PredictiveModel import PredictiveModel
from Utilities import Regression, PID
from SysModel import Simulator
from Track import Map
import numpy as np
import pickle
import pdb

def main():
    # ======================================================================================================================
    # ============================================= Initialize parameters  =================================================
    # ======================================================================================================================
    N = 14                                    # Horizon length
    n = 6;   d = 2                            # State and Input dimension
    x0 = np.array([0.5, 0, 0, 0, 0, 0])       # Initial condition
    xS = [x0, x0]
    dt = 0.1

    map = Map(0.4)                            # Initialize map
    vt = 0.8                                  # target vevlocity

    # Initialize controller parameters
    mpcParam, ltvmpcParam = initMPCParams(n, d, N, vt)
    numSS_it, numSS_Points, Laps, TimeLMPC, QterminalSlack, lmpcParameters = initLMPCParams(map, N)

    # Init simulators
    simulator     = Simulator(map)
    LMPCsimulator = Simulator(map, multiLap = False, flagLMPC = True)

    # ======================================================================================================================
    # ======================================= PID path following ===========================================================
    # ======================================================================================================================
    print("Starting PID")
    # Initialize pid and run sim
    PIDController = PID(vt)
    xPID_cl, uPID_cl, xPID_cl_glob, _ = simulator.sim(xS, PIDController)
    print("===== PID terminated")

    # ======================================================================================================================
    # ======================================  LINEAR REGRESSION ============================================================
    # ======================================================================================================================
    print("Starting MPC")
    # Estimate system dynamics
    lamb = 0.0000001
    A, B, Error = Regression(xPID_cl, uPID_cl, lamb)
    mpcParam.A = A
    mpcParam.B = B
    # Initialize MPC and run closed-loop sim
    mpc = MPC(mpcParam)
    xMPC_cl, uMPC_cl, xMPC_cl_glob, _ = simulator.sim(xS, mpc)
    print("===== MPC terminated")

    # ======================================================================================================================
    # ===================================  LOCAL LINEAR REGRESSION =========================================================
    # ======================================================================================================================
    print("Starting TV-MPC")
    # Initialized predictive model
    predictiveModel = PredictiveModel(n, d, map, 1)
    predictiveModel.addTrajectory(xPID_cl,uPID_cl)
    #Initialize TV-MPC
    ltvmpcParam.timeVarying = True 
    mpc = MPC(ltvmpcParam, predictiveModel)
    # Run closed-loop sim
    xTVMPC_cl, uTVMPC_cl, xTVMPC_cl_glob, _ = simulator.sim(xS, mpc)
    print("===== TV-MPC terminated")

    # ======================================================================================================================
    # ==============================  LMPC w\ LOCAL LINEAR REGRESSION ======================================================
    # ======================================================================================================================
    print("Starting LMPC")
    # Initialize Predictive Model for lmpc
    lmpcpredictiveModel = PredictiveModel(n, d, map, 4)
    for i in range(0,4): # add trajectories used for model learning
        lmpcpredictiveModel.addTrajectory(xPID_cl,uPID_cl)

    # Initialize Controller
    lmpcParameters.timeVarying     = True 
    lmpc = LMPC(numSS_Points, numSS_it, QterminalSlack, lmpcParameters, lmpcpredictiveModel)
    for i in range(0,4): # add trajectories for safe set
        lmpc.addTrajectory( xPID_cl, uPID_cl, xPID_cl_glob)
    
    # Run sevaral laps
    for it in range(numSS_it, Laps):
        # Simulate controller
        xLMPC, uLMPC, xLMPC_glob, xS = LMPCsimulator.sim(xS,  lmpc)
        # Add trajectory to controller
        lmpc.addTrajectory( xLMPC, uLMPC, xLMPC_glob)
        # lmpcpredictiveModel.addTrajectory(np.append(xLMPC,np.array([xS[0]]),0),np.append(uLMPC, np.zeros((1,2)),0))
        lmpcpredictiveModel.addTrajectory(xLMPC,uLMPC)
        print("Completed lap: ", it, " in ", np.round(lmpc.Qfun[it][0]*dt, 2)," seconds")
    print("===== LMPC terminated")

    # # ======================================================================================================================
    # # ========================================= PLOT TRACK =================================================================
    # # ======================================================================================================================
    for i in range(0, lmpc.it):
        print("Lap time at iteration ", i, " is ",np.round( lmpc.Qfun[i][0]*dt, 2), "s")

    print("===== Start Plotting")
    plotTrajectory(map, xPID_cl, xPID_cl_glob, uPID_cl, 'PID')
    plotTrajectory(map, xMPC_cl, xMPC_cl_glob, uMPC_cl, 'MPC')
    plotTrajectory(map, xTVMPC_cl, xTVMPC_cl_glob, uTVMPC_cl, 'TV-MPC')
    plotClosedLoopLMPC(lmpc, map)
    animation_xy(map, lmpc, Laps-1)
    plt.show()

    # animation_states(map, LMPCOpenLoopData, lmpc, Laps-2)
    # animation_states(map, LMPCOpenLoopData, lmpc, Laps-2)
    # animation_states(map, LMPCOpenLoopData, lmpc, Laps-2)
    # animation_states(map, LMPCOpenLoopData, lmpc, Laps-2)
    # saveGif_xyResults(map, LMPCOpenLoopData, lmpc, Laps-2)

if __name__== "__main__":
  main()