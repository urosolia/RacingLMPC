from scipy import linalg
import numpy as np
from cvxopt.solvers import qp
from cvxopt import spmatrix, matrix, solvers
from Utilities import Curvature
import datetime

solvers.options['show_progress'] = False

class PathFollowingLTV_MPC:
    """Create the Path Following LMPC controller with LTV model
    Attributes:
        solve: given x0 computes the control action
    """
    def __init__(self, Q, R, N, vt, n, d, x, u, dt, map, inputConstr):
        """Initialization
        Q, R: weights to build the cost function h(x,u) = ||x||_Q + ||u||_R
        N: horizon length
        vt: target velocity
        n, d: state and input dimension
        x, u: date used in the system identification strategy
        dt: discretization time
        map: map
        """
        self.A    = []
        self.B    = []
        self.C    = []
        self.N    = N
        self.n    = n
        self.d    = d
        self.vt   = vt
        self.Q    = Q
        self.R    = R
        self.Datax = x
        self.Datau = u
        self.LinPoints = x[0:N+1,:]
        self.dt = dt
        self.map = map
        self.inputConstr = inputConstr

        self.M, self.q = _buildMatCost(self)
        self.F, self.b = _buildMatIneqConst(self)
        self.G = []
        self.E = []

    def solve(self, x0):
        """Computes control action
        Arguments:
            x0: current state position
        """

        startTimer = datetime.datetime.now()
        self.A, self.B, self. C = _EstimateABC(self)
        self.G, self.E, self.L = _buildMatEqConst(self)
        endTimer = datetime.datetime.now(); deltaTimer = endTimer - startTimer
        self.linearizationTime = deltaTimer

        M = self.M
        q = self.q
        G = self.G
        L = self.L
        E = self.E
        F = self.F
        b = self.b
        n = self.n
        N = self.N
        d = self.d

        startTimer = datetime.datetime.now()
        sol = qp(M, matrix(q), F, matrix(b), G, E * matrix(x0) + L)
        endTimer = datetime.datetime.now(); deltaTimer = endTimer - startTimer
        self.solverTime = deltaTimer
        if sol['status'] == 'optimal':
            self.feasible = 1
        else:
            self.feasible = 0

        self.xPred = np.squeeze(np.transpose(np.reshape((np.squeeze(sol['x'])[np.arange(n * (N + 1))]), (N + 1, n))))
        self.uPred = np.squeeze(np.transpose(np.reshape((np.squeeze(sol['x'])[n * (N + 1) + np.arange(d * N)]), (N, d))))
        self.LinPoints = np.concatenate( (self.xPred.T[1:,:], np.array([self.xPred.T[-1,:]])), axis=0 )
        self.xPred = self.xPred.T
        self.uPred = self.uPred.T

# ======================================================================================================================
# ======================================================================================================================
# =============================== Internal functions for MPC reformulation to QP =======================================
# ======================================================================================================================
# ======================================================================================================================
def _buildMatIneqConst(Controller):
    N = Controller.N
    n = Controller.n
    inputConstr = Controller.inputConstr

    # Buil the matrices for the state constraint in each region. In the region i we want Fx[i]x <= bx[b]
    Fx = np.array([[1., 0., 0., 0., 0., 0.],
                   [0., 0., 0., 0., 0., 1.],
                   [0., 0., 0., 0., 0., -1.]])

    bx = np.array([[10],  # vx max
                   [2.],  # max ey
                   [2.]]) # max ey

    # Buil the matrices for the input constraint in each region. In the region i we want Fx[i]x <= bx[b]
    Fu = np.array([[1., 0.],
                   [-1., 0.],
                   [0., 1.],
                   [0., -1.]])

    bu = np.array([[inputConstr[0,0]],  # -Min Steering
                   [inputConstr[0,1]],  # Max Steering
                   [inputConstr[1,0]],  # -Min Acceleration
                   [inputConstr[1,1]]])  # Max Acceleration

    # Now stuck the constraint matrices to express them in the form Fz<=b. Note that z collects states and inputs
    # Let's start by computing the submatrix of F relates with the state
    rep_a = [Fx] * (N)
    Mat = linalg.block_diag(*rep_a)
    NoTerminalConstr = np.zeros((np.shape(Mat)[0], n))  # No need to constraint also the terminal point
    Fxtot = np.hstack((Mat, NoTerminalConstr))
    bxtot = np.tile(np.squeeze(bx), N)

    # Let's start by computing the submatrix of F relates with the input
    rep_b = [Fu] * (N)
    Futot = linalg.block_diag(*rep_b)
    butot = np.tile(np.squeeze(bu), N)

    # Let's stack all together
    rFxtot, cFxtot = np.shape(Fxtot)
    rFutot, cFutot = np.shape(Futot)
    Dummy1 = np.hstack((Fxtot, np.zeros((rFxtot, cFutot))))
    Dummy2 = np.hstack((np.zeros((rFutot, cFxtot)), Futot))
    F = np.vstack((Dummy1, Dummy2))
    b = np.hstack((bxtot, butot))

    F_sparse = spmatrix(F[np.nonzero(F)], np.nonzero(F)[0].astype(int), np.nonzero(F)[1].astype(int), F.shape)
    return F_sparse, b

def _buildMatCost(Controller):
    Q  = Controller.Q
    R  = Controller.R
    P  = Controller.Q
    N  = Controller.N
    vt = Controller.vt
    b = [Q] * (N)
    Mx = linalg.block_diag(*b)

    c = [R] * (N)
    Mu = linalg.block_diag(*c)

    M0 = linalg.block_diag(Mx, P, Mu)
    xtrack = np.array([vt, 0, 0, 0, 0, 0])
    q = - 2 * np.dot(np.append(np.tile(xtrack, N + 1), np.zeros(R.shape[0] * N)), M0)
    M = 2 * M0  # Need to multiply by two because CVX considers 1/2 in front of quadratic cost

    M_sparse = spmatrix(M[np.nonzero(M)], np.nonzero(M)[0].astype(int), np.nonzero(M)[1].astype(int), M.shape)
    return M_sparse, q

def _buildMatEqConst(Controller):
    # Buil matrices for optimization (Convention from Chapter 15.2 Borrelli, Bemporad and Morari MPC book)
    # We are going to build our optimization vector z \in \mathbb{R}^((N+1) \dot n \dot N \dot d), note that this vector
    # stucks the predicted trajectory x_{k|t} \forall k = t, \ldots, t+N+1 over the horizon and
    # the predicte input u_{k|t} \forall k = t, \ldots, t+N over the horizon
    A = Controller.A
    B = Controller.B
    C = Controller.C
    N = Controller.N
    n = Controller.n
    d = Controller.d

    Gx = np.eye(n * (N + 1))
    Gu = np.zeros((n * (N + 1), d * (N)))

    E = np.zeros((n * (N + 1), n))
    E[np.arange(n)] = np.eye(n)

    L = np.zeros((n * (N + 1), 1))

    for i in range(0, N):
        ind1 = n + i * n + np.arange(n)
        ind2x = i * n + np.arange(n)
        ind2u = i * d + np.arange(d)

        Gx[np.ix_(ind1, ind2x)] = -A[i]
        Gu[np.ix_(ind1, ind2u)] = -B[i]
        L[ind1, :] = C[i]

    G = np.hstack((Gx, Gu))

    G_sparse = spmatrix(G[np.nonzero(G)], np.nonzero(G)[0].astype(int), np.nonzero(G)[1].astype(int), G.shape)
    E_sparse = spmatrix(E[np.nonzero(E)], np.nonzero(E)[0].astype(int), np.nonzero(E)[1].astype(int), E.shape)
    L_sparse = spmatrix(L[np.nonzero(L)], np.nonzero(L)[0].astype(int), np.nonzero(L)[1].astype(int), L.shape)

    return G_sparse, E_sparse, L_sparse

def _EstimateABC(Controller):
    LinPoints = Controller.LinPoints
    N = Controller.N
    n = Controller.n
    d = Controller.d
    x = Controller.Datax
    u = Controller.Datau
    PointAndTangent = Controller.map.PointAndTangent
    dt = Controller.dt

    Atv = []; Btv = []; Ctv = []

    for i in range(0, N + 1):
        MaxNumPoint = 500 # Need to reason on how these points are selected
        x0 = LinPoints[i, :]


        Ai = np.zeros((n, n))
        Bi = np.zeros((n, d))
        Ci = np.zeros((n, 1))

        # =========================
        # ====== Identify vx ======
        h = 2
        stateFeatures = [0, 1, 2]
        inputFeatures = [1]
        lamb = 0.0
        yIndex = 0
        scaling = np.array([[1.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0],
                            [0.0, 0.0, 1.0]])

        Ai[yIndex, stateFeatures], Bi[yIndex, inputFeatures], Ci[yIndex], _ = LocLinReg(h, x, u, x0, yIndex, stateFeatures,
                                                                             inputFeatures, scaling, qp, matrix, lamb, MaxNumPoint)

        # =========================
        # ====== Identify vy ======
        h = 2
        stateFeatures = [0, 1, 2]
        inputFeatures = [0] # May want to add acceleration here
        lamb = 0.0
        yIndex = 1
        # scaling = np.array([[1.0, 0.0, 0.0],
        #                     [0.0, 1.0, 0.0],
        #                     [0.0, 0.0, 1.0]])
        scaling = np.eye(len(stateFeatures))

        Ai[yIndex, stateFeatures], Bi[yIndex, inputFeatures], Ci[yIndex], _ = LocLinReg(h, x, u, x0, yIndex, stateFeatures,
                                                                             inputFeatures, scaling, qp, matrix, lamb, MaxNumPoint)

        # =========================
        # ====== Identify wz ======
        h = 2
        stateFeatures = [0, 1, 2]
        inputFeatures = [0] # May want to add acceleration here
        lamb = 0.0
        yIndex = 2
        # scaling = np.array([[1.0, 0.0, 0.0],
        #                     [0.0, 1.0, 0.0],
        #                     [0.0, 0.0, 1.0]])
        scaling = np.eye(len(stateFeatures))

        Ai[yIndex, stateFeatures], Bi[yIndex, inputFeatures], Ci[yIndex], _ = LocLinReg(h, x, u, x0, yIndex, stateFeatures,
                                                                             inputFeatures, scaling, qp, matrix, lamb, MaxNumPoint)

        # ===========================
        # ===== Linearization =======
        vx = x0[0]; vy   = x0[1]
        wz = x0[2]; epsi = x0[3]
        s  = x0[4]; ey   = x0[5]

        if s<0:
            print("s is negative, here the state: \n", LinPoints)

        cur = Curvature(s, PointAndTangent)
        den = 1 - cur *ey
        # ===========================
        # ===== Linearize epsi ======
        # epsi_{k+1} = epsi + dt * ( wz - (vx * np.cos(epsi) - vy * np.sin(epsi)) / (1 - cur * ey) * cur )
        depsi_vx   =     -dt * np.cos(epsi) / den * cur
        depsi_vy   =      dt * np.sin(epsi) / den * cur
        depsi_wz   =      dt
        depsi_epsi =  1 - dt * ( -vx * np.sin(epsi) - vy * np.cos(epsi) ) / den * cur
        depsi_s    =      0                                                                      # Because cur = constant
        depsi_ey   =      dt * (vx * np.cos(epsi) - vy * np.sin(epsi)) / (den**2) * cur * (-cur)

        Ai[3, :] = [depsi_vx, depsi_vy, depsi_wz, depsi_epsi, depsi_s, depsi_ey]
        Ci[3] = epsi + dt * (wz - (vx * np.cos(epsi) - vy * np.sin(epsi)) / (1 - cur * ey) * cur) - np.dot(Ai[3, :], x0)
        # ===========================
        # ===== Linearize s =========
        # s_{k+1} = s    + dt * ( (vx * np.cos(epsi) - vy * np.sin(epsi)) / (1 - cur * ey) )
        ds_vx = dt * (np.cos(epsi) / den)
        ds_vy = -dt * (np.sin(epsi) / den)
        ds_wz = 0
        ds_epsi = dt * (-vx * np.sin(epsi) - vy * np.cos(epsi)) / den
        ds_s = 1  # + Ts * (Vx * cos(epsi) - Vy * sin(epsi)) / (1 - ey * rho) ^ 2 * (-ey * drho);
        ds_ey = -dt * (vx * np.cos(epsi) - vy * np.sin(epsi)) / (den * 2) * (-cur)

        Ai[4, :] = [ds_vx, ds_vy, ds_wz, ds_epsi, ds_s, ds_ey]
        Ci[4] = s + dt * ((vx * np.cos(epsi) - vy * np.sin(epsi)) / (1 - cur * ey)) - np.dot(Ai[4, :], x0)

        # ===========================
        # ===== Linearize ey ========
        # ey_{k+1} = ey + dt * (vx * np.sin(epsi) + vy * np.cos(epsi))
        dey_vx = dt * np.sin(epsi)
        dey_vy = dt * np.cos(epsi)
        dey_wz = 0
        dey_epsi = dt * (vx * np.cos(epsi) - vy * np.sin(epsi))
        dey_s = 0
        dey_ey = 1

        Ai[5, :] = [dey_vx, dey_vy, dey_wz, dey_epsi, dey_s, dey_ey]
        Ci[5] = ey + dt * (vx * np.sin(epsi) + vy * np.cos(epsi)) - np.dot(Ai[5, :], x0)

        Atv.append(Ai)
        Btv.append(Bi)
        Ctv.append(Ci)

    return Atv, Btv, Ctv

def LocLinReg(h, x, u, x0, yIndex, stateFeatures, inputFeatures, scaling, qp, matrix, lamb, MaxNumPoint):
    import numpy as np
    from numpy import linalg as la
    # What to learn a model such that: x_{k+1} = A x_k  + B u_k + C
    oneVec = np.ones( (x.shape[0]-1, 1) )
    x0Vec = (np.dot( np.array([x0[stateFeatures]]).T, oneVec.T )).T
    diff  = np.dot(( x[0:-1, stateFeatures] - x0Vec ), scaling)
    # print 'x0Vec \n',x0Vec
    norm = la.norm(diff, 1, axis=1)
    indexTot =  np.squeeze(np.where(norm < h))
    if (indexTot.shape[0] >= MaxNumPoint):
        index = np.argsort(norm)[0:MaxNumPoint]
        # MinNorm = np.argmin(norm)
        # if MinNorm+MaxNumPoint >= indexTot.shape[0]:
        #     index = indexTot[indexTot.shape[0]-MaxNumPoint:indexTot.shape[0]]
        # else:
        #     index = indexTot[MinNorm:MinNorm+MaxNumPoint]
    else:
        index = indexTot

    K  = ( 1 - ( norm[index] / h )**2 ) * 3/4
    # K = np.ones(len(index))
    X0 = np.hstack( ( x[np.ix_(index, stateFeatures)], u[np.ix_(index, inputFeatures)] ) )
    M = np.hstack( ( X0, np.ones((X0.shape[0],1)) ) )

    y = x[np.ix_(index+1, [yIndex])]
    b = matrix( -np.dot( np.dot(M.T, np.diag(K)), y) )

    Q0 = np.dot( np.dot(M.T, np.diag(K)), M )
    Q  = matrix( Q0 + lamb * np.eye(Q0.shape[0]) )

    res_cons = qp(Q, b) # This is ordered as [A B C]
    Result = np.squeeze(np.array(res_cons['x']))
    A = Result[0:len(stateFeatures)]
    B = Result[len(stateFeatures):(len(stateFeatures)+len(inputFeatures))]
    C = Result[-1]

    return A, B, C, index