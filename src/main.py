import sys
sys.path.append('fnc')
from SysModel import DynModel
from FTOCP import BuildMatEqConst, BuildMatCost, BuildMatIneqConst, FTOCP, GetPred
from Track import CreateTrack, Evaluate_e_ey
from SysID import LocLinReg, Regression, EstimateABC
from LMPC import LMPC, ComputeCost, LMPC_BuildMatEqConst, LMPC_BuildMatIneqConst
import numpy as np
import matplotlib.pyplot as plt
from cvxopt.solvers import qp
from cvxopt import spmatrix, matrix, solvers
from scipy import linalg
import datetime
from numpy import linalg as la

solvers.options['show_progress'] = False
# ======================================================================================================================
# ====================================== INITIALIZE PARAMETERS =========================================================
# ======================================================================================================================
RunPID   = 0
plotFlag = 0                       # Plot flag
dt = 1.0/50.0                      # Controller discretization time
Time = 100                          # Simulation time
Points = int(Time/dt)              # Number of points in the simulation
u = np.zeros((Points, 2))          # Initialize the input vector
x      = np.zeros((Points+1, 6))   # Initialize state vector (In curvilinear abscissas)
x_glob = np.zeros((Points+1, 6))   # Initialize the state vector in absolute reference frame
vt = 2

x[0,0] = 0.5; x_glob[0,0] = 0.5
x[0,4] = 0.5; x_glob[0,4] = 0.5
# Create the track. The vector PointAndTangent = [xf yf]
PointAndTangent = CreateTrack()

# ======================================================================================================================
# ======================================= PID path following ===========================================================
# ======================================================================================================================
if RunPID == 1:
    for i in range(0, Points):
        u[i, 0] = - 0.6 * x[i, 5] - 0.9 * x[i, 3] + np.maximum(-0.9, np.min(np.random.randn()*0.25, 0.9))
        u[i, 1] = vt - x[i, 0] + np.maximum(-0.5, np.min(np.random.randn()*0.15, 0.5))
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

A, B = Regression(x, u, lamb)
print "A matrix: \n", A, "\n B matrix: \n", B

n = 6
d = 2
N = 8


Q = np.diag([10.0, 0.0, 0.0, 1.0, 0.0, 10.0]) # vx, vy, wz, epsi, s, ey
R = np.diag([1.0, 0.1]) # delta, a

M, q    = BuildMatCost(Q, R, Q, N, linalg, np, spmatrix, vt)
F, b    = BuildMatIneqConst(N, n, np, linalg, spmatrix)
G, E, L = BuildMatEqConst(A, B, np.zeros((n,1)), N, n, d, np, spmatrix, 0)

# Initialize
RunMPC = 0
plotFlagMPC = 0                          # Plot flag
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
RunMPC_tv = 0
plotFlagMPC_tv = 0                          # Plot flag
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
RunLMPC = 1
plotFlagLMPC = 1                          # Plot flag
TimeLMPC = 1                             # Simulation time
PointsLMPC = int(TimeLMPC / dt)            # Number of points in the simulation
uLMPC = np.zeros((PointsLMPC, 2))          # Initialize the input vector
xLMPC      = np.zeros((PointsLMPC+1, 6))   # Initialize state vector (In curvilinear abscissas)
x_globLMPC = np.zeros((PointsLMPC+1, 6))   # Initialize the state vector in absolute reference frame

xLMPC[0,:] = x[0,:]
x_globLMPC[0,:] = x_glob[0,:]
# Time loop
LinPoints = x[0:N+1,:]


numSS_Points = 300

SS   = 10000*np.ones((xMPC_tv.shape[0], 6, 10))
Qfun = 10000*np.ones((xMPC_tv.shape[0], 10))

SS[:,:, 0]  = xMPC
Qfun[:, 0] = ComputeCost(xMPC_tv, uMPC_tv, np)
F_LMPC, b_LMPC = LMPC_BuildMatIneqConst(N, n, np, linalg, spmatrix, numSS_Points)

if RunLMPC == 1:
    for i in range(0, PointsLMPC):
        x0 = xLMPC[i, :]

        startTimer = datetime.datetime.now() # Start timer for LMPC iteration
        if i == 0:
            F, b = BuildMatIneqConst(N, n, np, linalg, spmatrix)
            G, E, L, npG, npE = LMPC_BuildMatEqConst(A, B, np.zeros((n, 1)), N, n, d, np, spmatrix, 0)
        else:
            Atv, Btv, Ctv = EstimateABC(LinPoints, N, n, d, xMPC_tv, uMPC_tv, qp, matrix, PointAndTangent, dt)
            G, E, L, npG, npE = LMPC_BuildMatEqConst(Atv, Btv, Ctv, N, n, d, np, spmatrix, 1)

        endTimer = datetime.datetime.now(); deltaTimer_tv = endTimer - startTimer

        Sol, feasible, deltaTimer = LMPC(npG, L, npE, F_LMPC, b_LMPC, x0, np, qp, matrix, datetime, la, SS, Qfun,  N, n, d, spmatrix, numSS_Points)

        xPred, uPred = GetPred(Sol, n, d, N, np)
        LinPoints = xPred.T
        uLMPC[i, :] = uPred[:, 0]
        xLMPC[i+1, :], x_globLMPC[i+1, :] = DynModel(xLMPC[i, :], x_globLMPC[i, :], uLMPC[i, :], np, dt, PointAndTangent)
        if i <= 5:
            print("Linearization time: %.4fs Solver time: %.4fs" % (deltaTimer_tv.total_seconds(), deltaTimer.total_seconds()))
            print "Time: ", i * dt, "Current State and Input: ", xLMPC[i, :], uLMPC[i, :]
            print "LMPC xPred:\n", xPred

        if feasible == 0:
            print "Unfeasible at time ", i*dt
            break

    np.savez('LMPC_PathFollowing', x=xLMPC, u=uLMPC, x_glob=x_globLMPC)
else:
    data = np.load('LMPC_PathFollowing.npz')
    xLMPC      = data['x']
    uLMPC      = data['u']
    x_globLMPC = data['x_glob']


print "===== LMPC terminated"

# ======================================================================================================================
# ========================================= PLOT TRACK =================================================================
# ======================================================================================================================
print "===== Start Plotting"
if plotFlag == 1:
    Points = np.floor(10*(PointAndTangent[-1, 3] + PointAndTangent[-1, 4]))
    Points1 = np.zeros((Points,2))
    Points2 = np.zeros((Points,2))
    Points0 = np.zeros((Points,2))
    for i in range(0,int(Points)):
        Points1[i, :] = Evaluate_e_ey(i*0.1, 1.5, PointAndTangent)
        Points2[i, :] = Evaluate_e_ey(i*0.1,-1.5, PointAndTangent)
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
        Points1[i, :] = Evaluate_e_ey(i*0.1, 1.5, PointAndTangent)
        Points2[i, :] = Evaluate_e_ey(i*0.1,-1.5, PointAndTangent)
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
        Points1[i, :] = Evaluate_e_ey(i*0.1, 1.5, PointAndTangent)
        Points2[i, :] = Evaluate_e_ey(i*0.1,-1.5, PointAndTangent)
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
        Points1[i, :] = Evaluate_e_ey(i*0.1, 1.5, PointAndTangent)
        Points2[i, :] = Evaluate_e_ey(i*0.1,-1.5, PointAndTangent)
        Points0[i, :] = Evaluate_e_ey(i*0.1, 0, PointAndTangent)

    plt.figure(7)
    plt.plot(PointAndTangent[:,0], PointAndTangent[:,1], 'o')
    plt.plot(Points0[:,0], Points0[:,1], '--')
    plt.plot(Points1[:,0], Points1[:,1], '-b')
    plt.plot(Points2[:,0], Points2[:,1], '-b')
    plt.plot(x_globLMPC[:,4], x_globLMPC[:,5], '-r')

    plt.figure(8)
    plt.subplot(711)
    plt.plot(xLMPC[:, 0], '-o')
    plt.ylabel('vx')
    plt.subplot(712)
    plt.plot(xLMPC[:, 1], '-o')
    plt.ylabel('vy')
    plt.subplot(713)
    plt.plot(xLMPC[:, 2],'-o')
    plt.ylabel('wz')
    plt.subplot(714)
    plt.plot(xLMPC[:, 3],'-o')
    plt.ylabel('epsi')
    plt.subplot(715)
    plt.plot(xLMPC[:, 5], '-o')
    plt.ylabel('ey')
    plt.subplot(716)
    plt.plot(uLMPC[:, 0], '-o')
    plt.ylabel('Steering')
    plt.subplot(717)
    plt.plot(uLMPC[:, 1],'-o')
    plt.ylabel('Acc')

plt.show()
