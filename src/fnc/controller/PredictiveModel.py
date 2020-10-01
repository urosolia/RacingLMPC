from cvxopt import spmatrix, matrix, solvers
from numpy import linalg as la
from cvxopt.solvers import qp
import numpy as np
import datetime
import pdb
# This class is not generic and is tailored to the autonomous racing problem.
# The only method need the LT-MPC and the LMPC is regressionAndLinearization, which given a state-action pair
# compute the matrices A,B,C such that x_{k+1} = A x_k + Bu_k + C

class PredictiveModel():
    def __init__(self,  n, d, map, trToUse):
        self.map = map
        self.n = n # state dimension
        self.d = d # input dimention
        self.xStored = []
        self.uStored = []
        self.MaxNumPoint = 7 # max number of point per lap to use 
        self.h = 5 # bandwidth of the Kernel for local linear regression
        self.lamb = 0.0 # regularization
        self.dt = 0.1
        self.scaling = np.array([[0.1, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 1.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 1.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 1.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 1.0]])

        self.stateFeatures    = [0, 1, 2]
        self.inputFeaturesVx  = [1]
        self.inputFeaturesLat = [0]
        self.usedIt = [i for i in range(trToUse)]
        self.lapTime = []
    

    def addTrajectory(self, x, u):
        if self.lapTime == [] or x.shape[0] >= self.lapTime[-1]:
            self.xStored.append(x)
            self.uStored.append(u)
            self.lapTime.append(x.shape[0])
        else:
            for i in range(0, len(self.xStored)):
                if x.shape[0] < self.lapTime[i]:
                    self.xStored.insert(i, x) 
                    self.uStored.insert(i, u) 
                    self.lapTime.insert(i, x.shape[0]) 
                    break

    def regressionAndLinearization(self, x, u):
        Ai = np.zeros((self.n, self.n))
        Bi = np.zeros((self.n, self.d))
        Ci = np.zeros(self.n)

        # Compute Index to use for each stored lap
        xuLin = np.hstack((x[self.stateFeatures], u[:]))
        self.indexSelected = []
        self.K = []
        for ii in self.usedIt:
            indexSelected_i, K_i = self.computeIndices(xuLin, ii)
            self.indexSelected.append(indexSelected_i)
            self.K.append(K_i)
        # print("xuLin: ",xuLin)
        # print("aaa indexSelected: ", self.indexSelected)

        # =========================
        # ====== Identify vx ======
        Q_vx, M_vx = self.compute_Q_M(self.inputFeaturesVx, self.usedIt)

        yIndex = 0
        b_vx = self.compute_b(yIndex, self.usedIt, M_vx)
        Ai[yIndex, self.stateFeatures], Bi[yIndex, self.inputFeaturesVx], Ci[yIndex] = self.LMPC_LocLinReg(Q_vx, b_vx, self.inputFeaturesVx)

        # =======================================
        # ====== Identify Lateral Dynamics ======
        Q_lat, M_lat = self.compute_Q_M(self.inputFeaturesLat, self.usedIt)

        yIndex = 1  # vy
        b_vy = self.compute_b(yIndex, self.usedIt, M_lat)
        Ai[yIndex, self.stateFeatures], Bi[yIndex, self.inputFeaturesLat], Ci[yIndex] = self.LMPC_LocLinReg(Q_lat, b_vy, self.inputFeaturesLat)

        yIndex = 2  # wz
        b_wz = self.compute_b(yIndex, self.usedIt, M_lat)
        Ai[yIndex, self.stateFeatures], Bi[yIndex, self.inputFeaturesLat], Ci[yIndex] = self.LMPC_LocLinReg(Q_lat, b_wz, self.inputFeaturesLat)

        # ===========================
        # ===== Linearization =======
        vx = x[0]; vy   = x[1]
        wz = x[2]; epsi = x[3]
        s  = x[4]; ey   = x[5]
        dt = self.dt

        if s < 0:
            print("s is negative, here the state: \n", x)

        startTimer = datetime.datetime.now()  # Start timer for LMPC iteration
        cur = self.map.curvature(s)
        cur = self.map.curvature(s)
        den = 1 - cur * ey

        # ===========================
        # ===== Linearize epsi ======
        # epsi_{k+1} = epsi + dt * ( wz - (vx * np.cos(epsi) - vy * np.sin(epsi)) / (1 - cur * ey) * cur )
        depsi_vx   = -dt * np.cos(epsi) / den * cur
        depsi_vy   = dt * np.sin(epsi) / den * cur
        depsi_wz   = dt
        depsi_epsi = 1 - dt * (-vx * np.sin(epsi) - vy * np.cos(epsi)) / den * cur
        depsi_s    = 0  # Because cur = constant
        depsi_ey   = dt * (vx * np.cos(epsi) - vy * np.sin(epsi)) / (den ** 2) * cur * (-cur)

        Ai[3, :] = [depsi_vx, depsi_vy, depsi_wz, depsi_epsi, depsi_s, depsi_ey]
        Ci[3]    = epsi + dt * (wz - (vx * np.cos(epsi) - vy * np.sin(epsi)) / (1 - cur * ey) * cur) - np.dot(Ai[3, :], x)
        # ===========================
        # ===== Linearize s =========
        # s_{k+1} = s    + dt * ( (vx * np.cos(epsi) - vy * np.sin(epsi)) / (1 - cur * ey) )
        ds_vx   = dt * (np.cos(epsi) / den)
        ds_vy   = -dt * (np.sin(epsi) / den)
        ds_wz   = 0
        ds_epsi = dt * (-vx * np.sin(epsi) - vy * np.cos(epsi)) / den
        ds_s    = 1  # + Ts * (Vx * cos(epsi) - Vy * sin(epsi)) / (1 - ey * rho) ^ 2 * (-ey * drho);
        ds_ey   = -dt * (vx * np.cos(epsi) - vy * np.sin(epsi)) / (den ** 2) * (-cur)

        Ai[4, :] = [ds_vx, ds_vy, ds_wz, ds_epsi, ds_s, ds_ey]
        Ci[4]    = s + dt * ((vx * np.cos(epsi) - vy * np.sin(epsi)) / (1 - cur * ey)) - np.dot(Ai[4, :], x)

        # ===========================
        # ===== Linearize ey ========
        # ey_{k+1} = ey + dt * (vx * np.sin(epsi) + vy * np.cos(epsi))
        dey_vx   = dt * np.sin(epsi)
        dey_vy   = dt * np.cos(epsi)
        dey_wz   = 0
        dey_epsi = dt * (vx * np.cos(epsi) - vy * np.sin(epsi))
        dey_s    = 0
        dey_ey   = 1

        Ai[5, :] = [dey_vx, dey_vy, dey_wz, dey_epsi, dey_s, dey_ey]
        Ci[5]    = ey + dt * (vx * np.sin(epsi) + vy * np.cos(epsi)) - np.dot(Ai[5, :], x)

        endTimer = datetime.datetime.now(); deltaTimer_tv = endTimer - startTimer

        return Ai, Bi, Ci

    def compute_Q_M(self, inputFeatures, usedIt):
        Counter = 0
        X0   = np.empty((0,len(self.stateFeatures)+len(inputFeatures)))
        Ktot = np.empty((0))

        for it in usedIt:
            X0 = np.append( X0, np.hstack((self.xStored[it][np.ix_(self.indexSelected[Counter], self.stateFeatures)],self.uStored[it][np.ix_(self.indexSelected[Counter], inputFeatures)])), axis=0 )
            Ktot    = np.append(Ktot, self.K[Counter])
            Counter += 1

        M = np.hstack( (X0, np.ones((X0.shape[0], 1))) )
        Q0 = np.dot(np.dot(M.T, np.diag(Ktot)), M)
        Q = matrix(Q0 + self.lamb * np.eye(Q0.shape[0]))

        return Q, M

    def compute_b(self, yIndex, usedIt, M):
        Counter = 0
        y = np.empty((0))
        Ktot = np.empty((0))

        for it in usedIt:
            y       = np.append(y, np.squeeze(self.xStored[it][self.indexSelected[Counter] + 1, yIndex]))
            Ktot    = np.append(Ktot, self.K[Counter])
            Counter += 1

        b = matrix(-np.dot(np.dot(M.T, np.diag(Ktot)), y))
        return b

    def LMPC_LocLinReg(self, Q, b, inputFeatures):
        # Solve QP
        res_cons = qp(Q, b) # This is ordered as [A B C]
        # Unpack results
        result = np.squeeze(np.array(res_cons['x']))
        A = result[0:len(self.stateFeatures)]
        B = result[len(self.stateFeatures):(len(self.stateFeatures)+len(inputFeatures))]
        C = result[-1]
        return A, B, C

    def computeIndices(self, x, it):
        oneVec = np.ones( (self.xStored[it].shape[0]-1, 1) )
        xVec = (np.dot( np.array([x]).T, oneVec.T )).T
        DataMatrix = np.hstack((self.xStored[it][0:-1, self.stateFeatures], self.uStored[it][0:-1, :]))

        diff  = np.dot(( DataMatrix - xVec ), self.scaling)
        norm = la.norm(diff, 1, axis=1)
        indexTot =  np.squeeze(np.where(norm < self.h))
        if (indexTot.shape[0] >= self.MaxNumPoint):
            index = np.argsort(norm)[0:self.MaxNumPoint]
        else:
            index = indexTot

        K  = ( 1 - ( norm[index] / self.h )**2 ) * 3/4
        # if norm.shape[0]<500:
        #     print("norm: ", norm, norm.shape)

        return index, K