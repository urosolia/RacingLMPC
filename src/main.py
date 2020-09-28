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
sys.path.append('fnc')
sys.path.append('fnc/simulator')
sys.path.append('fnc/controller')
from SysModel import Simulator
from Classes import LMPCprediction
from PredictiveControllers import MPC, MPC_parameters, ControllerLMPC_child
from PredictiveModel import PredictiveModel
from Track import Map
from LMPC import ControllerLMPC
from Utilities import Regression, PID
from plot import plotTrajectory, plotClosedLoopLMPC, animation_xy, animation_states, saveGif_xyResults, Save_statesAnimation
import numpy as np
import matplotlib.pyplot as plt
import pdb
import pickle

def main():
    # ======================================================================================================================
    # ============================================= Initialize parameters  =================================================
    # ======================================================================================================================
    N = 12                                    # Horizon length
    n = 6;   d = 2                            # State and Input dimension
    x0 = np.array([0.5, 0, 0, 0, 0, 0])       # Initial condition
    xS = [x0, x0]
    dt = 0.1

    map = Map(0.4)                            # Initialize map
    simulator = Simulator(map)                # Initialize Simulator
    vt = 0.8                                  # target vevlocity
    mpcParameters, mpcParametersLTV = initMPCparameters(n, d, N,vt)

    LMPCSimulator = Simulator(map, multiLap = False, flagLMPC = True)
    LMPC_Solver, numSS_it, numSS_Points, Laps, TimeLMPC, QterminalSlack, Qslack, Q_LMPC, R_LMPC, dR_LMPC, inputConstr, lmpcParameters = initLMPCparameters(map, N)

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
    ClosedLoopDataLTI_MPC = []
    mpcParameters['A']    = A
    mpcParameters['B']    = B
    # Initialize MPC and run closed-loop sim
    mpc = MPC(mpcParameters)
    xMPC_cl, uMPC_cl, xMPC_cl_glob, _ = simulator.sim(xS, mpc)
    print("===== MPC terminated")

    # ======================================================================================================================
    # ===================================  LOCAL LINEAR REGRESSION =========================================================
    # ======================================================================================================================
    # print("Starting TV-MPC")
    # # Initialized predictive model
    # predictiveModel = PredictiveModel(n, d, map)
    # predictiveModel.xStored.append(xPID_cl)
    # predictiveModel.uStored.append(uPID_cl)
    # #Initialize TV-MPC
    # mpcParametersLTV["timeVarying"]     = True 
    # mpcParametersLTV["predictiveModel"] = predictiveModel
    # mpc = MPC(mpcParametersLTV)
    # # Run closed-loop sim
    # xTVMPC_cl, uTVMPC_cl, xTVMPC_cl_glob, _ = simulator.sim(xS, mpc)
    # print("===== TV-MPC terminated")

    # ======================================================================================================================
    # ==============================  LMPC w\ LOCAL LINEAR REGRESSION ======================================================
    # ======================================================================================================================
    print("Starting LMPC")
    LMPCOpenLoopData = LMPCprediction(N, n, d, TimeLMPC, numSS_Points, Laps)
    # Initialize Controller
    # LMPController = ControllerLMPC(numSS_Points, numSS_it, N, QterminalSlack, Qslack, Q_LMPC, R_LMPC, dR_LMPC, map, Laps, TimeLMPC, LMPC_Solver, inputConstr)
    LMPController = ControllerLMPC_child(numSS_Points, numSS_it, N, QterminalSlack, Qslack, Q_LMPC, R_LMPC, dR_LMPC, map, Laps, TimeLMPC, LMPC_Solver, inputConstr, lmpcParameters)
    LMPController.addTrajectory( xPID_cl, uPID_cl, xPID_cl_glob)
    LMPController.addTrajectory( xMPC_cl, uMPC_cl, xMPC_cl_glob)
    LMPController.addTrajectory( xPID_cl, uPID_cl, xPID_cl_glob)
    LMPController.addTrajectory( xMPC_cl, uMPC_cl, xMPC_cl_glob)

    # Run sevaral laps
    for it in range(numSS_it, Laps):
        # Simulate controller
        xLMPC, uLMPC, xLMPC_glob, xS = LMPCSimulator.sim(xS,  LMPController, LMPCprediction = LMPCOpenLoopData)
        # Add trajectory to controller
        LMPController.addTrajectory( xLMPC, uLMPC, xLMPC_glob)
        print("Completed lap: ", it, " in ", np.round(LMPController.Qfun[0, it]*dt, 2)," seconds")
    print("===== LMPC terminated")

    # ======================================================================================================================
    # ========================================= PLOT TRACK =================================================================
    # ======================================================================================================================
    for i in range(0, LMPController.it):
        print("Lap time at iteration ", i, " is ",np.round( LMPController.Qfun[0, i]*dt, 2), "s")

    print("===== Start Plotting")
    plotTrajectory(map, xPID_cl, xPID_cl_glob, uPID_cl, 'PID')
    plotTrajectory(map, xMPC_cl, xMPC_cl_glob, uMPC_cl, 'MPC')
    # plotTrajectory(map, xTVMPC_cl, xTVMPC_cl_glob, uTVMPC_cl, 'TV-MPC')
    plotClosedLoopLMPC(LMPController, map)
    animation_xy(map, LMPCOpenLoopData, LMPController, Laps-2)

    # animation_states(map, LMPCOpenLoopData, LMPController, 10)
    # saveGif_xyResults(map, LMPCOpenLoopData, LMPController, Laps-2)
    # Save_statesAnimation(map, LMPCOpenLoopData, LMPController, 5)
    plt.show()

def initMPCparameters(n, d, N, vt):
        # Path Following tuning
    Q = np.diag([1.0, 1.0, 1, 1, 0.0, 100.0]) # vx, vy, wz, epsi, s, ey
    R = np.diag([1.0, 10.0])                  # delta, a

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

    mpcParameters    = MPC_parameters(n, d, N)
    mpcParametersLTV = MPC_parameters(n, d, N)

    for mpcPram in [mpcParameters, mpcParametersLTV]:
        mpcPram['Q']      = Q
        mpcPram['Qf']     = Q
        mpcPram['R']      = R
        mpcPram['Fx']     = Fx
        mpcPram['bx']     = bx
        mpcPram['Fu']     = Fu
        mpcPram['bu']     = bu
        mpcPram['xRef']   = np.array([vt, 0, 0, 0, 0, 0])
        mpcPram['slacks'] = True
        mpcPram['Qslack'] = 1 * np.array([0, 50])
        

    return mpcParameters, mpcParametersLTV

def initLMPCparameters(map, N):
    # Path Following tuning
    Q = np.diag([1.0, 1.0, 1, 1, 0.0, 100.0]) # vx, vy, wz, epsi, s, ey
    R = np.diag([1.0, 10.0])                  # delta, a

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
    LMPC_Solver = "OSQP"           # Can pick CVX for cvxopt or OSQP. For OSQP uncomment line 14 in LMPC.py
    numSS_it = 4                  # Number of trajectories used at each iteration to build the safe set
    numSS_Points = 40             # Number of points to select from each trajectory to build the safe set

    Laps       = 35+numSS_it      # Total LMPC laps
    TimeLMPC   = 400              # Simulation time

    # Tuning Parameters
    QterminalSlack  = 20 * np.diag([10, 1, 1, 1, 10, 1])    # Cost on the slack variable for the terminal constraint
    Qslack  =  1 * np.array([0, 50])                       # Quadratic and linear slack lane cost
    Q_LMPC  =  0 * np.diag([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # State cost x = [vx, vy, wz, epsi, s, ey]
    R_LMPC  =  0 * np.diag([1.0, 1.0])                      # Input cost u = [delta, a]
    dR_LMPC =  5 * np.array([1.0, 10.0])                    # Input rate cost u

    inputConstr = np.array([[0.5, 0.5],                     # Min Steering and Max Steering
                            [10.0, 10.0]])                  # Min Acceleration and Max Acceleration
    
    lmpcParameters    = MPC_parameters(Q.shape[0], R.shape[0], N)
    lmpcParameters['Q']      = Q_LMPC
    lmpcParameters['dR']     = dR_LMPC
    lmpcParameters['R']      = R_LMPC
    lmpcParameters['Fx']     = Fx
    lmpcParameters['bx']     = bx
    lmpcParameters['Fu']     = Fu
    lmpcParameters['bu']     = bu
    lmpcParameters['slacks'] = True
    lmpcParameters['Qslack'] = Qslack

    return LMPC_Solver, numSS_it, numSS_Points, Laps, TimeLMPC, QterminalSlack, Qslack, Q_LMPC, R_LMPC, dR_LMPC, inputConstr, lmpcParameters

if __name__== "__main__":
  main()