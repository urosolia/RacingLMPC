from scipy import linalg
import numpy as np
from cvxopt.solvers import qp
from cvxopt import spmatrix, matrix, solvers
import datetime, pdb

solvers.options['show_progress'] = False

class PathFollowingLTI_MPC:
    """Create the Path Following LMPC controller with LTI model
    Attributes:
        solve: given x0 computes the control action
    """
    def __init__(self, A, B, Q, R, N, vt, inputConstr):
        """Initialization
        A, B: Liner Time Invariant (LTI) system dynamics
        Q, R: weights to build the cost function h(x,u) = ||x||_Q + ||u||_R
        N: horizon length
        vt: target velocity
        """
        self.A = A         # n x n matrix
        self.B = B         # n x d matrix
        self.n = A.shape[0]   # number of states
        self.d = B.shape[1]   # number of inputs
        self.N = N
        self.Q = Q
        self.R = R
        self.vt = vt        # target velocity
        startTimer = datetime.datetime.now()
        endTimer = datetime.datetime.now(); deltaTimer = endTimer - startTimer
        self.solverTime = deltaTimer
        self.linearizationTime = deltaTimer
        self.inputConstr = inputConstr

        self.M, self.q = _buildMatCost(self)
        self.F, self.b = _buildMatIneqConst(self)
        self.G, self.E = _buildMatEqConst(self)

    def solve(self, x0):
        M = self.M
        F = self.F
        G = self.G
        E = self.E
        q = self.q
        b = self.b
        n, d = self.n, self.d
        N = self.N

        startTimer = datetime.datetime.now()
        sol = qp(M, matrix(q), F, matrix(b), G, E * matrix(x0))
        endTimer = datetime.datetime.now(); deltaTimer = endTimer - startTimer
        self.solverTime = deltaTimer
        if sol['status'] == 'optimal':
            self.feasible = 1
        else:
            self.feasible = 0

        self.xPred = np.squeeze(np.transpose(np.reshape((np.squeeze(sol['x'])[np.arange(n * (N + 1))]), (N + 1, n)))).T
        self.uPred = np.squeeze(np.transpose(np.reshape((np.squeeze(sol['x'])[n * (N + 1) + np.arange(d * N)]), (N, d)))).T



# ======================================================================================================================
# ======================================================================================================================
# =============================== Internal functions for MPC reformulation to QP =======================================
# ======================================================================================================================
# ======================================================================================================================

def _buildMatEqConst(Controller):
    # Buil matrices for optimization (Convention from Chapter 15.2 Borrelli, Bemporad and Morari MPC book)
    # We are going to build our optimization vector z \in \mathbb{R}^((N+1) \dot n \dot N \dot d), note that this vector
    # stucks the predicted trajectory x_{k|t} \forall k = t, \ldots, t+N+1 over the horizon and
    # the predicte input u_{k|t} \forall k = t, \ldots, t+N over the horizon
    n = Controller.n
    N = Controller.N
    d = Controller.d
    A,B = Controller.A, Controller.B

    Gx = np.eye(n * (N + 1))
    Gu = np.zeros((n * (N + 1), d * (N)))

    E = np.zeros((n * (N + 1), n))
    E[np.arange(n)] = np.eye(n)

    for i in range(0, N):
        ind1 = n + i * n + np.arange(n)
        ind2x = i * n + np.arange(n)
        ind2u = i * d + np.arange(d)

        Gx[np.ix_(ind1, ind2x)] = -A
        Gu[np.ix_(ind1, ind2u)] = -B

    G = np.hstack((Gx, Gu))

    G_sparse = spmatrix(G[np.nonzero(G)], np.nonzero(G)[0].astype(int), np.nonzero(G)[1].astype(int), G.shape)
    E_sparse = spmatrix(E[np.nonzero(E)], np.nonzero(E)[0].astype(int), np.nonzero(E)[1].astype(int), E.shape)

    return G_sparse, E_sparse

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
                   [2.]]), # max ey

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
    N = Controller.N
    Q = Controller.Q
    R = Controller.R
    vt = Controller.vt
    P = Controller.Q
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