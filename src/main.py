import sys
sys.path.append('fnc')
from SysModel import DynModel
from FTOCP import BuildMatEqConst, BuildMatCost, BuildMatIneqConst, FTOCP, GetPred
from Track import CreateTrack, Evaluate_e_ey
from SysID import LocLinReg, Regression, EstimateABC, LMPC_EstimateABC
from LMPC import LMPC, ComputeCost, LMPC_BuildMatEqConst, LMPC_BuildMatIneqConst
from InvariantSets import PropagatePoly, GenerateW, Invariance
import numpy as np
import matplotlib.pyplot as plt
from cvxopt.solvers import qp
from cvxopt import spmatrix, matrix, solvers
from scipy import linalg
import scipy
import datetime
from numpy import linalg as la
from pathos.multiprocessing import ProcessingPool as Pool
from polyhedron import Vrep, Hrep

from functools import partial

pvx = Pool(4)  # Initialize the pool for multicore
pvy = Pool(4)  # Initialize the pool for multicore
pwz = Pool(4)  # Initialize the pool for multicore

solvers.options['show_progress'] = False
# CHOOSE WHAT TO RUN
RunPID     = 0; plotFlag       = 0
RunMPC     = 0; plotFlagMPC    = 1
RunMPC_tv  = 0; plotFlagMPC_tv = 0
RunLMPC    = 1; plotFlagLMPC   = 1

# ======================================================================================================================
# ====================================== INITIALIZE PARAMETERS =========================================================
# ======================================================================================================================
dt = 1.0/10.0                      # Controller discretization time
Time = 100                          # Simulation time
Points = int(Time/dt)              # Number of points in the simulation
u = np.zeros((Points, 2))          # Initialize the input vector
x      = np.zeros((Points+1, 6))   # Initialize state vector (In curvilinear abscissas)
x_glob = np.zeros((Points+1, 6))   # Initialize the state vector in absolute reference frame
vt = 0.8

x[0,0] = 0.5; x_glob[0,0] = 0.5
x[0,4] = 0.5; x_glob[0,4] = 0.5
# Create the track. The vector PointAndTangent = [xf yf]
PointAndTangent, TrackLength = CreateTrack()
# ======================================================================================================================
# ======================================= PID path following ===========================================================
# ======================================================================================================================
if RunPID == 1:
    for i in range(0, Points):
        u[i, 0] = - 0.6 * x[i, 5] - 0.9 * x[i, 3] + np.maximum(-0.9, np.min(np.random.randn()*0.25, 0.9))
        u[i, 1] = 1.5*(vt - x[i, 0]) + np.maximum(-0.2, np.min(np.random.randn()*0.10, 0.2))
        x[i+1, :], x_glob[i+1, :] = DynModel(x[i, :], x_glob[i, :], u[i, :], np, dt, PointAndTangent)

    print "Number of laps completed: ", int(np.floor(x[-1, 4] / (PointAndTangent[-1, 3] + PointAndTangent[-1, 4])))
    np.savez('PID_PathFollowing', x=x, u=u, x_glob = x_glob)
else:
    data = np.load('PID_PathFollowing.npz')
    x      = data['x']
    u      = data['u']
    x_glob = data['x_glob']
# ======================================================================================================================
# ======================================  LINEAR REGRESSION ============================================================
# ======================================================================================================================
print "Starting MPC"
lamb = 0.0000001
SafetyFactor = 1.15
A, B, Error = Regression(x, u, lamb)

# np.savetxt('Error.csv', Error, delimiter=',', fmt='%f')
# np.savetxt('A.csv', A, delimiter=',', fmt='%f')
# np.savetxt('B.csv', B, delimiter=',', fmt='%f')
#
# W = GenerateW(Error*SafetyFactor)

#rho = 0.1
#max_r = 10
#InvariantSet = Invariance(A_cl, W, rho, max_r)

print "A matrix: \n", A, "\n B matrix: \n", B

n = 6
d = 2
N = 12

Q = np.diag([1.0, 1.0, 1, 1, 0.0, 100.0]) # vx, vy, wz, epsi, s, ey
R = np.diag([1.0, 10.0]) # delta, a

M, q    = BuildMatCost(Q, R, Q, N, linalg, np, spmatrix, vt)
F, b    = BuildMatIneqConst(N, n, np, linalg, spmatrix)
G, E, L = BuildMatEqConst(A, B, np.zeros((n,1)), N, n, d, np, spmatrix, 0)

# Initialize
TimeMPC = 100                             # Simulation time
PointsMPC = int(TimeMPC / dt)            # Number of points in the simulation
uMPC = np.zeros((PointsMPC, 2))          # Initialize the input vector
xMPC      = np.zeros((PointsMPC+1, 6))   # Initialize state vector (In curvilinear abscissas)
x_globMPC = np.zeros((PointsMPC+1, 6))   # Initialize the state vector in absolute reference frame

xMPC[0,:] = x[0,:]
x_globMPC[0,:] = x_glob[0,:]
# Time loop
if RunMPC == 1:
    for i in range(0, PointsMPC):
        x0 = xMPC[i, :]
        startTimer = datetime.datetime.now()
        Sol, feasible = FTOCP(M, q, G, L, E, F, b, x0, np, qp, matrix)
        endTimer = datetime.datetime.now(); deltaTimer = endTimer - startTimer

        xPred, uPred = GetPred(Sol, n, d, N, np)
        uMPC[i, :] = uPred[:, 0]
        xMPC[i+1, :], x_globMPC[i+1, :] = DynModel(xMPC[i, :], x_globMPC[i, :], uMPC[i, :], np, dt, PointAndTangent)

        if i <= 5:
            print("Solver time: %.4fs" % (deltaTimer.total_seconds()))
            print "Time: ", i * dt, "Current State and Input: ", xMPC[i, :], uMPC[i, :]

        if feasible == 0:
            print "Unfeasible at time ", i*dt
            print "Cur State: ", x0
            print "Pred State: ", xPred

            break

    np.savez('MPC_PathFollowing', x=xMPC, u=uMPC, x_glob=x_globMPC)
else:
    data = np.load('MPC_PathFollowing.npz')
    xMPC      = data['x']
    uMPC      = data['u']
    x_globMPC = data['x_glob']

print "===== MPC terminated"

# ======================================================================================================================
# ===================================  LOCAL LINEAR REGRESSION =========================================================
# ======================================================================================================================
# Initialize
TimeMPC_tv = 100                             # Simulation time
PointsMPC_tv = int(TimeMPC_tv / dt)            # Number of points in the simulation
uMPC_tv = np.zeros((PointsMPC_tv, 2))          # Initialize the input vector
xMPC_tv      = np.zeros((PointsMPC_tv+1, 6))   # Initialize state vector (In curvilinear abscissas)
x_globMPC_tv = np.zeros((PointsMPC_tv+1, 6))   # Initialize the state vector in absolute reference frame

xMPC_tv[0,:] = x[0,:]
x_globMPC_tv[0,:] = x_glob[0,:]
# Time loop
LinPoints = x[0:N+1,:]

if RunMPC_tv == 1:
    for i in range(0, PointsMPC_tv):
        x0 = xMPC_tv[i, :]

        startTimer = datetime.datetime.now() # Start timer for LMPC iteration
        if i == 0:
            F, b = BuildMatIneqConst(N, n, np, linalg, spmatrix)
            G, E, L = BuildMatEqConst(A, B, np.zeros((n, 1)), N, n, d, np, spmatrix, 0)
        else:
            Atv, Btv, Ctv = EstimateABC(LinPoints, N, n, d, x, u, qp, matrix, PointAndTangent, dt)
            G, E, L = BuildMatEqConst(Atv, Btv, Ctv, N, n, d, np, spmatrix, 1)
        endTimer = datetime.datetime.now(); deltaTimer_tv = endTimer - startTimer

        startTimer = datetime.datetime.now() # Start timer for LMPC iteration
        Sol, feasible = FTOCP(M, q, G, L, E, F, b, x0, np, qp, matrix)
        endTimer = datetime.datetime.now(); deltaTimer = endTimer - startTimer

        xPred, uPred = GetPred(Sol, n, d, N, np)
        LinPoints = xPred.T
        uMPC_tv[i, :] = uPred[:, 0]
        xMPC_tv[i+1, :], x_globMPC_tv[i+1, :] = DynModel(xMPC_tv[i, :], x_globMPC_tv[i, :], uMPC_tv[i, :], np, dt, PointAndTangent)

        if i <= 5:
            print("Linearization time: %.4fs Solver time: %.4fs" % (deltaTimer_tv.total_seconds(), deltaTimer.total_seconds()))
            print "Time: ", i * dt, "Current State and Input: ", xMPC_tv[i, :], uMPC_tv[i, :]
            print "xPred:\n", xPred

        if feasible == 0:
            print "Unfeasible at time ", i*dt
            break

    np.savez('MPC_tv_PathFollowing', x=xMPC_tv, u=uMPC_tv, x_glob=x_globMPC_tv)
else:
    data = np.load('MPC_tv_PathFollowing.npz')
    xMPC_tv      = data['x']
    uMPC_tv      = data['u']
    x_globMPC_tv = data['x_glob']


print "===== TV-MPC terminated"

# ======================================================================================================================
# ==============================  LMPC w\ LOCAL LINEAR REGRESSION ======================================================
# ======================================================================================================================
# Initialize
PlotIndex  = 0
PlotPred   = 0
TimeLMPC   = 400                             # Simulation time
PointsLMPC = int(TimeLMPC / dt)            # Number of points in the simulation
uLMPC = np.zeros((PointsLMPC, 2))          # Initialize the input vector
xLMPC      = np.zeros((PointsLMPC+1, 6))   # Initialize state vector (In curvilinear abscissas)
x_globLMPC = np.zeros((PointsLMPC+1, 6))   # Initialize the state vector in absolute reference frame
Laps       = 5

xLMPC[0,:] = x[0,:]
x_globLMPC[0,:] = x_glob[0,:]
# Time loop
LinPoints = xMPC_tv[0:N+1,:]

numSS_Points = 30
swifth = N-1

TimeSS = 10000*np.ones(Laps+2)
SS     = 10000*np.ones((2*xMPC_tv.shape[0], 6, Laps+2))
uSS    = 10000*np.ones((2*xMPC_tv.shape[0], 2, Laps+2))
Qfun   = 0*np.ones((2*xMPC_tv.shape[0], Laps+2)) # Need to initialize at zero as adding point on the fly

# Adding Trajectory to safe set iteration 0
TimeSS[0] = x.shape[0]
SS[0:TimeSS[0],:, 0]  = x
uSS[0:TimeSS[0]-1,:, 0]  = u
Qfun[0:TimeSS[0], 0] = ComputeCost(x, u, np, TrackLength)

# Adding Trajectory to safe set iteration 1
TimeSS[1] = xMPC_tv.shape[0]
SS[0:TimeSS[1],:, 1]  = xMPC_tv
uSS[0:TimeSS[1]-1,:, 1]  = uMPC_tv
Qfun[0:TimeSS[1], 1] = ComputeCost(xMPC_tv, uMPC_tv, np, TrackLength)

print Qfun[0:TimeSS[0], 0], Qfun[0:TimeSS[1], 1]

F_LMPC, b_LMPC = LMPC_BuildMatIneqConst(N, n, np, linalg, spmatrix, numSS_Points)

Qslack = 50*np.diag([1, 10, 10, 10, 10, 10])
Q_LMPC = 0 * np.diag([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # vx, vy, wz, epsi, s, ey
R_LMPC = 5 * np.diag([1.0, 1.0])  # delta, a

if PlotIndex == 1:
    plt.figure(105)
    xdata = []; ydata = []
    xdata0 = []; ydata0 = []
    fig0 = plt.figure()
    ax0 = fig0.add_subplot(1, 1, 1)
    Points = np.floor(10 * (PointAndTangent[-1, 3] + PointAndTangent[-1, 4]))
    Points1 = np.zeros((Points, 2))
    Points2 = np.zeros((Points, 2))
    Points0 = np.zeros((Points, 2))
    for i in range(0, int(Points)):
        Points1[i, :] = Evaluate_e_ey(i * 0.1, 1.5, PointAndTangent)
        Points2[i, :] = Evaluate_e_ey(i * 0.1, -1.5, PointAndTangent)
        Points0[i, :] = Evaluate_e_ey(i * 0.1, 0, PointAndTangent)

    plt.plot(PointAndTangent[:, 0], PointAndTangent[:, 1], 'o')
    plt.plot(Points0[:, 0], Points0[:, 1], '--')
    plt.plot(Points1[:, 0], Points1[:, 1], '-b')
    plt.plot(Points2[:, 0], Points2[:, 1], '-b')
    plt.plot(x_globMPC_tv[:, 4], x_globMPC_tv[:, 5], '-r')
    line, = ax0.plot(xdata, ydata, 'or')
    line0, = ax0.plot(xdata0, ydata0, 'sg')


if PlotPred == 1:
    plt.figure(100)
    xdata = []; ydata = []
    fig = plt.figure()

    ax1 = fig.add_subplot(5, 1, 1)
    plt.plot(xMPC_tv[0:PointsLMPC+100, 0], '-o')
    line1, = ax1.plot(xdata, ydata, 'or-')

    ax2 = fig.add_subplot(5, 1, 2)
    ax2.plot(xMPC_tv[0:PointsLMPC + 100, 1], '-o')
    line2, = ax2.plot(xdata, ydata, 'or-')

    ax3 = fig.add_subplot(5, 1, 3)
    ax3.plot(xMPC_tv[0:PointsLMPC + 100, 2], '-o')
    line3, = ax3.plot(xdata, ydata, 'or-')

    ax4 = fig.add_subplot(5, 1, 4)
    ax4.plot(xMPC_tv[0:PointsLMPC + 100, 3], '-o')
    line4, = ax4.plot(xdata, ydata, 'or-')

    ax5 = fig.add_subplot(5, 1, 5)
    ax5.plot(xMPC_tv[0:PointsLMPC + 100, 5], '-o')
    line5, = ax5.plot(xdata, ydata, 'or-')

print uSS[0:TimeSS[1]-1,:, 1].shape, uMPC_tv.shape
print SS[0:TimeSS[1],:, 1].shape, xMPC_tv.shape

print (uSS[0:TimeSS[1]-1,:, 1]== uMPC_tv).all()
print (SS[0:TimeSS[1],:, 1]== xMPC_tv).all()
print SS[0:TimeSS[1],:, 1],"\n", xMPC_tv

absTime = 0
if RunLMPC == 1:
    for it in range(2,2+Laps):
        if it <= 2:
            x_ID = xMPC_tv
            u_ID = uMPC_tv
        else:
            print xLMPC[i-2:i+2, :], xLMPC[i, :]
            SS[0:i, :, it-1]  = xLMPC[0:i, :]
            uSS[0:i-1, :, it-1] = uLMPC[0:i-1, :]
            TimeSS[it-1] = i

            x_ID = SS[0:i, :, it-1]
            u_ID = uSS[0:i-1, :, it-1]

            Qfun[0:i, it-1] = ComputeCost(xLMPC[0:i, :], xLMPC[0:i-1, :], np, TrackLength)

            print Qfun[:, it-1], Qfun[:, it-2], it
            xLMPC[0, :] = xLMPC[i, :]
            xLMPC[0, 4] = xLMPC[i, 4] - TrackLength
            xLMPC[1:,:] = 0*xLMPC[1:,:]
            Counter = i
            # name = raw_input("Please enter name: ")

        i = 0
        while (xLMPC[i, 4] < TrackLength):
            x0 = xLMPC[i, :]

            if i == 0:
                startTimer = datetime.datetime.now()  # Start timer for LMPC iteration
                G, E, L, npG, npE = LMPC_BuildMatEqConst(A, B, np.zeros((n, 1)), N, n, d, np, spmatrix, 0)
                endTimer = datetime.datetime.now(); deltaTimer_tv = endTimer - startTimer

            else:
                startTimer = datetime.datetime.now()  # Start timer for LMPC iteration

                Atv, Btv, Ctv, indexUsed_list = LMPC_EstimateABC(LinPoints, LinInput, N, n, d, SS, uSS, TimeSS, qp, matrix,
                                                                 PointAndTangent, dt, it)
                endTimer = datetime.datetime.now(); deltaTimer_tv = endTimer - startTimer
                # Atv0, Btv0, Ctv0 = EstimateABC(LinPoints, N, n, d, x_ID, u_ID, qp, matrix, PointAndTangent, dt)
                # print (Atv0[0]==Atv[0]).all()
                #
                G, E, L, npG, npE = LMPC_BuildMatEqConst(Atv, Btv, Ctv, N, n, d, np, spmatrix, 1)
                # G, E, L, npG, npE = LMPC_BuildMatEqConst(A, B, np.zeros((n, 1)), N, n, d, np, spmatrix, 0)

                if PlotIndex == 1:
                    line.set_xdata(x_globMPC_tv[indexUsed_list[0], 4])
                    line.set_ydata(x_globMPC_tv[indexUsed_list[0], 5])
                    line0.set_xdata(x_globLMPC[i,4])
                    line0.set_ydata(x_globLMPC[i,5])
                    plt.draw()
                    plt.pause(1e-17)



            Sol, feasible, deltaTimer, slack = LMPC(npG, L, npE, F_LMPC, b_LMPC, x0, np, qp, matrix, datetime, la, SS,
                                                    Qfun,  N, n, d, spmatrix, numSS_Points, Qslack, Q_LMPC, R_LMPC, it, swifth)

            xPred, uPred = GetPred(Sol, n, d, N, np)

            LinPoints = np.vstack((xPred.T[1:,:], xPred.T[-1,:]))
            LinInput = uPred.T

            uLMPC[i, :] = uPred[:, 0]
            xLMPC[i + 1, :], x_globLMPC[absTime + 1, :] = DynModel(xLMPC[i, :], x_globLMPC[absTime, :], uLMPC[i, :], np, dt,
                                                             PointAndTangent)

            print xLMPC[i + 1, 4], TrackLength, it, xLMPC[i + 1, 0], np.dot(slack, slack), slack
            if PlotPred == 1:
                line1.set_xdata(np.arange(i, i+N+1))
                line1.set_ydata(xPred[0, :])
                line2.set_xdata(np.arange(i, i + N + 1))
                line2.set_ydata(xPred[1, :])
                line3.set_xdata(np.arange(i, i + N + 1))
                line3.set_ydata(xPred[2, :])
                line4.set_xdata(np.arange(i, i + N + 1))
                line4.set_ydata(xPred[3, :])
                line5.set_xdata(np.arange(i, i + N + 1))
                line5.set_ydata(xPred[5, :])
                ax1.set_title(np.dot(slack, slack))

                plt.draw()
                plt.pause(1e-17)

            if i <= 5:
                print("Linearization time: %.4fs Solver time: %.4fs" % (deltaTimer_tv.total_seconds(), deltaTimer.total_seconds()))
                print "Time: ", i * dt, "Current State and Input: ", xLMPC[i, :], uLMPC[i, :]
                # print "LMPC xPred:\n", xPred

            if feasible == 0:
                print "Unfeasible at time ", i*dt
                break

            if it > 2:
                SS[Counter + i + 1, :, it - 1]  = xLMPC[i + 1, :] + np.array([0, 0, 0, 0, TrackLength, 0])
                uSS[Counter + i + 1, :, it - 1] = uLMPC[i, :]
            i = i + 1
            absTime = absTime + 1



    np.savez('LMPC_PathFollowing', x=xLMPC, u=uLMPC, x_glob=x_globLMPC, ss=SS, uss=uSS)
else:
    data = np.load('LMPC_PathFollowing.npz')
    xLMPC      = data['x']
    uLMPC      = data['u']
    x_globLMPC = data['x_glob']
    SS         = data['ss']
    uSS        = data['uss']

if RunLMPC == 1:
    it = 2 + Laps
    SS[0:i, :, it-1]  = xLMPC[0:i, :]
    uSS[0:i-1, :, it-1] = uLMPC[0:i-1, :]
    TimeSS[it-1] = i
    Qfun[0:i, it-1] = ComputeCost(xLMPC[0:i, :], xLMPC[0:i-1, :], np, TrackLength)

print "===== LMPC terminated"
# ======================================================================================================================
# ========================================= PLOT TRACK =================================================================
# ======================================================================================================================
for i in range(0,Laps+2):
    print "Cost at iteration ", i, " is ",Qfun[0,i]
print "===== Start Plotting"
width = 0.8
if plotFlag == 1:
    Points = np.floor(10*(PointAndTangent[-1, 3] + PointAndTangent[-1, 4]))
    Points1 = np.zeros((Points,2))
    Points2 = np.zeros((Points,2))
    Points0 = np.zeros((Points,2))
    for i in range(0,int(Points)):
        Points1[i, :] = Evaluate_e_ey(i*0.1, width, PointAndTangent)
        Points2[i, :] = Evaluate_e_ey(i*0.1,-width, PointAndTangent)
        Points0[i, :] = Evaluate_e_ey(i*0.1, 0, PointAndTangent)

    plt.figure(1)
    plt.plot(PointAndTangent[:,0], PointAndTangent[:,1], 'o')
    plt.plot(Points0[:,0], Points0[:,1], '--')
    plt.plot(Points1[:,0], Points1[:,1], '-b')
    plt.plot(Points2[:,0], Points2[:,1], '-b')
    plt.plot(x_glob[:,4], x_glob[:,5], '-r')

    plt.figure(2)
    plt.subplot(711)
    plt.plot(x[:, 4], x[:, 0],'-o')
    plt.ylabel('vx')
    plt.subplot(712)
    plt.plot(x[:, 4], x[:, 1],'-o')
    plt.ylabel('vy')
    plt.subplot(713)
    plt.plot(x[:, 4], x[:, 2],'-o')
    plt.ylabel('wz')
    plt.subplot(714)
    plt.plot(x[:, 4], x[:, 3],'-o')
    plt.ylabel('epsi')
    plt.subplot(715)
    plt.plot(x[:, 4], x[:, 5],'-o')
    plt.ylabel('ey')
    plt.subplot(716)
    plt.plot(x[0:-1, 4], u[:, 0],'-o')
    plt.ylabel('steering')
    plt.subplot(717)
    plt.plot(x[0:-1, 4], u[:, 1], '-o')
    plt.ylabel('acc')

if plotFlagMPC == 1:
    Points = np.floor(10*(PointAndTangent[-1, 3] + PointAndTangent[-1, 4]))
    Points1 = np.zeros((Points,2))
    Points2 = np.zeros((Points,2))
    Points0 = np.zeros((Points,2))
    for i in range(0,int(Points)):
        Points1[i, :] = Evaluate_e_ey(i*0.1, width, PointAndTangent)
        Points2[i, :] = Evaluate_e_ey(i*0.1,-width, PointAndTangent)
        Points0[i, :] = Evaluate_e_ey(i*0.1, 0, PointAndTangent)

    plt.figure(3)
    plt.plot(PointAndTangent[:,0], PointAndTangent[:,1], 'o')
    plt.plot(Points0[:,0], Points0[:,1], '--')
    plt.plot(Points1[:,0], Points1[:,1], '-b')
    plt.plot(Points2[:,0], Points2[:,1], '-b')
    plt.plot(x_globMPC[:,4], x_globMPC[:,5], '-r')

    plt.figure(4)
    plt.subplot(711)
    plt.plot(xMPC[:, 0], '-o')
    plt.ylabel('vx')
    plt.subplot(712)
    plt.plot(xMPC[:, 1], '-o')
    plt.ylabel('vy')
    plt.subplot(713)
    plt.plot(xMPC[:, 2],'-o')
    plt.ylabel('wz')
    plt.subplot(714)
    plt.plot(xMPC[:, 3],'-o')
    plt.ylabel('epsi')
    plt.subplot(715)
    plt.plot(xMPC[:, 5], '-o')
    plt.ylabel('ey')
    plt.subplot(716)
    plt.plot(uMPC[:, 0], '-o')
    plt.ylabel('Steering')
    plt.subplot(717)
    plt.plot(uMPC[:, 1],'-o')
    plt.ylabel('Acc')


if plotFlagMPC_tv == 1:
    Points = np.floor(10*(PointAndTangent[-1, 3] + PointAndTangent[-1, 4]))
    Points1 = np.zeros((Points,2))
    Points2 = np.zeros((Points,2))
    Points0 = np.zeros((Points,2))
    for i in range(0,int(Points)):
        Points1[i, :] = Evaluate_e_ey(i*0.1, width, PointAndTangent)
        Points2[i, :] = Evaluate_e_ey(i*0.1,-width, PointAndTangent)
        Points0[i, :] = Evaluate_e_ey(i*0.1, 0, PointAndTangent)

    plt.figure(5)
    plt.plot(PointAndTangent[:,0], PointAndTangent[:,1], 'o')
    plt.plot(Points0[:,0], Points0[:,1], '--')
    plt.plot(Points1[:,0], Points1[:,1], '-b')
    plt.plot(Points2[:,0], Points2[:,1], '-b')
    plt.plot(x_globMPC_tv[:,4], x_globMPC_tv[:,5], '-r')

    plt.figure(6)
    plt.subplot(711)
    plt.plot(xMPC_tv[:, 0], '-o')
    plt.ylabel('vx')
    plt.subplot(712)
    plt.plot(xMPC_tv[:, 1], '-o')
    plt.ylabel('vy')
    plt.subplot(713)
    plt.plot(xMPC_tv[:, 2],'-o')
    plt.ylabel('wz')
    plt.subplot(714)
    plt.plot(xMPC_tv[:, 3],'-o')
    plt.ylabel('epsi')
    plt.subplot(715)
    plt.plot(xMPC_tv[:, 5], '-o')
    plt.ylabel('ey')
    plt.subplot(716)
    plt.plot(uMPC_tv[:, 0], '-o')
    plt.ylabel('Steering')
    plt.subplot(717)
    plt.plot(uMPC_tv[:, 1],'-o')
    plt.ylabel('Acc')

if plotFlagLMPC == 1:
    Points = np.floor(10*(PointAndTangent[-1, 3] + PointAndTangent[-1, 4]))
    Points1 = np.zeros((Points,2))
    Points2 = np.zeros((Points,2))
    Points0 = np.zeros((Points,2))
    for i in range(0,int(Points)):
        Points1[i, :] = Evaluate_e_ey(i*0.1, width, PointAndTangent)
        Points2[i, :] = Evaluate_e_ey(i*0.1,-width, PointAndTangent)
        Points0[i, :] = Evaluate_e_ey(i*0.1, 0, PointAndTangent)

    plt.figure(7)
    plt.plot(PointAndTangent[:,0], PointAndTangent[:,1], 'o')
    plt.plot(Points0[:,0], Points0[:,1], '--')
    plt.plot(Points1[:,0], Points1[:,1], '-b')
    plt.plot(Points2[:,0], Points2[:,1], '-b')
    plt.plot(x_globLMPC[:,4], x_globLMPC[:,5], '-r')

    plt.figure(8)
    plt.subplot(711)
    for i in range(2, Laps + 2):
        plt.plot(SS[0:TimeSS[i], 4, i], SS[0:TimeSS[i], 0, i], '-o')
    plt.ylabel('vx')
    plt.subplot(712)
    for i in range(2, Laps + 2):
        plt.plot(SS[0:TimeSS[i], 4, i], SS[0:TimeSS[i], 1, i], '-o')
    plt.ylabel('vy')
    plt.subplot(713)
    for i in range(2, Laps + 2):
        plt.plot(SS[0:TimeSS[i], 4, i], SS[0:TimeSS[i], 2, i], '-o')
    plt.ylabel('wz')
    plt.subplot(714)
    for i in range(2, Laps + 2):
        plt.plot(SS[0:TimeSS[i], 4, i], SS[0:TimeSS[i], 3, i], '-o')
    plt.ylabel('epsi')
    plt.subplot(715)
    for i in range(2, Laps + 2):
        plt.plot(SS[0:TimeSS[i], 4, i], SS[0:TimeSS[i], 5, i], '-o')
    plt.ylabel('ey')
    plt.subplot(716)
    for i in range(2, Laps + 2):
        plt.plot(uSS[0:TimeSS[i]-1, 0, i], '-o')
    plt.ylabel('Steering')
    plt.subplot(717)
    for i in range(2, Laps + 2):
        plt.plot(uSS[0:TimeSS[i]-1, 1, i], '-o')
    plt.ylabel('Acc')

plt.show()
