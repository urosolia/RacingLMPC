from scipy import linalg, sparse
import numpy as np
from cvxopt.solvers import qp
from cvxopt import spmatrix, matrix, solvers
import datetime
import pdb
from numpy import hstack, inf, ones
from scipy.sparse import vstack
from osqp import OSQP

solvers.options['show_progress'] = False

class PathFollowingLTI_MPC:
    """Create the Path Following LMPC controller with LTI model
    Attributes:
        solve: given x0 computes the control action
    """
    def __init__(self, A, B, Q, R, N, vt, Qlane):
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
        self.Qlane = Qlane
        startTimer = datetime.datetime.now()
        endTimer = datetime.datetime.now(); deltaTimer = endTimer - startTimer
        self.solverTime = deltaTimer
        self.linearizationTime = deltaTimer

        self.M, self.q = _buildMatCost(self)
        self.F, self.b = _buildMatIneqConst(self)
        self.G, self.E = _buildMatEqConst(self)

    def solve(self, x0):
        """Solve the finite time optimal control problem
        x0: current state
        """
        M = self.M
        F = self.F
        G = self.G
        E = self.E
        q = self.q
        b = self.b
        n, d = self.n, self.d
        N = self.N

        startTimer = datetime.datetime.now()
#        sol = qp(M, matrix(q), F, matrix(b), G, E * matrix(x0))
        res_cons, feasible = osqp_solve_qp(sparse.csr_matrix(M), q, sparse.csr_matrix(F), b, sparse.csr_matrix(G), np.dot(E,x0))
        endTimer = datetime.datetime.now(); deltaTimer = endTimer - startTimer
        self.solverTime = deltaTimer

        Solution = res_cons.x

        self.feasible = feasible
        self.xPred = np.squeeze(np.transpose(np.reshape(Solution[np.arange(n * (N + 1))], (N + 1, n)))).T
        self.uPred = np.squeeze(np.transpose(np.reshape(Solution[n * (N + 1) + np.arange(d * N)], (N, d)))).T

    def oneStepPrediction(self, x, u, UpdateModel = 0):
        """Propagate the model one step foreward
        Arguments:
            x: current state
            u: current input
        """
        startTimer = datetime.datetime.now()
        endTimer = datetime.datetime.now(); deltaTimer = endTimer - startTimer

        x_next = np.dot(self.A, x) + np.dot(self.B, u)
        return x_next, deltaTimer

# ======================================================================================================================
# ======================================================================================================================
# =============================== Internal functions for MPC reformulation to QP =======================================
# ======================================================================================================================
# ======================================================================================================================
def osqp_solve_qp(P, q, G=None, h=None, A=None, b=None, initvals=None):
    """
    Solve a Quadratic Program defined as:
        minimize
            (1/2) * x.T * P * x + q.T * x
        subject to
            G * x <= h
            A * x == b
    using OSQP <https://github.com/oxfordcontrol/osqp>.
    Parameters
    ----------
    P : scipy.sparse.csc_matrix Symmetric quadratic-cost matrix.
    q : numpy.array Quadratic cost vector.
    G : scipy.sparse.csc_matrix Linear inequality constraint matrix.
    h : numpy.array Linear inequality constraint vector.
    A : scipy.sparse.csc_matrix, optional Linear equality constraint matrix.
    b : numpy.array, optional Linear equality constraint vector.
    initvals : numpy.array, optional Warm-start guess vector.
    Returns
    -------
    x : array, shape=(n,)
        Solution to the QP, if found, otherwise ``None``.
    Note
    ----
    OSQP requires `P` to be symmetric, and won't check for errors otherwise.
    Check out for this point if you e.g. `get nan values
    <https://github.com/oxfordcontrol/osqp/issues/10>`_ in your solutions.
    """
    osqp = OSQP()
    if G is not None:
        l = -inf * ones(len(h))
        if A is not None:
            qp_A = vstack([G, A]).tocsc()
            qp_l = hstack([l, b])
            qp_u = hstack([h, b])
        else:  # no equality constraint
            qp_A = G
            qp_l = l
            qp_u = h
        osqp.setup(P=P, q=q, A=qp_A, l=qp_l, u=qp_u, verbose=False, polish=True)
    else:
        osqp.setup(P=P, q=q, A=None, l=None, u=None, verbose=False)
    if initvals is not None:
        osqp.warm_start(x=initvals)
    res = osqp.solve()
    if res.info.status_val != osqp.constant('OSQP_SOLVED'):
        print("OSQP exited with status '%s'" % res.info.status)
    feasible = 0
    if res.info.status_val == osqp.constant('OSQP_SOLVED') or res.info.status_val == osqp.constant('OSQP_SOLVED_INACCURATE') or  res.info.status_val == osqp.constant('OSQP_MAX_ITER_REACHED'):
        feasible = 1
    return res, feasible

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

    G_hard = np.hstack((Gx, Gu))
    SlackLane = np.zeros((G_hard.shape[0], 2*N))
    G = np.hstack((G_hard, SlackLane))

    return G, E

def _buildMatIneqConst(Controller):
    N = Controller.N
    n = Controller.n
    # Buil the matrices for the state constraint in each region. In the region i we want Fx[i]x <= bx[b]
    Fx = np.array([[0., 0., 0., 0., 0., 1.],
                   [0., 0., 0., 0., 0., -1.]])

    bx = np.array([[2.],  # max ey
                   [2.]])  # max ey

    # Buil the matrices for the input constraint in each region. In the region i we want Fx[i]x <= bx[b]
    Fu = np.array([[1., 0.],
                   [-1., 0.],
                   [0., 1.],
                   [0., -1.]])

    bu = np.array([[0.5],  # Max Steering
                   [0.5],  # Max Steering
                   [1.],  # Max Acceleration
                   [1.]])  # Max Acceleration

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
    F_hard = np.vstack((Dummy1, Dummy2))
    b = np.hstack((bxtot, butot))

    LaneSlack = np.zeros((F_hard.shape[0], 2*N))
    colIndexPositive = []
    rowIndexPositive = []
    colIndexNegative = []
    rowIndexNegative = []
    for i in range(0, N):
        colIndexPositive.append( i*2 + 0 )
        colIndexNegative.append( i*2 + 1 )

        rowIndexPositive.append(i*Fx.shape[0] + 0) # Slack on second element of Fx
        rowIndexNegative.append(i*Fx.shape[0] + 1) # Slack on third element of Fx
    
    LaneSlack[rowIndexPositive, colIndexPositive] =  1.0
    LaneSlack[rowIndexNegative, rowIndexNegative] = -1.0

    F = np.hstack((F_hard, LaneSlack))

    return F, b

def _buildMatCost(Controller):
    N = Controller.N
    Q = Controller.Q
    R = Controller.R
    vt = Controller.vt
    P = Controller.Q
    b = [Q] * (N)
    Mx = linalg.block_diag(*b)
    Qlane = Controller.Qlane

    dR = np.array([2 * 2 * 2 * 40, 2* 10])
    c = [R + 2 * np.diag(dR)] * (N) # Need to add dR for the derivative input cost

    Mu = linalg.block_diag(*c)
    # Need to condider that the last input appears just once in the difference
    Mu[Mu.shape[0] - 1, Mu.shape[1] - 1] = Mu[Mu.shape[0] - 1, Mu.shape[1] - 1] - dR[1]
    Mu[Mu.shape[0] - 2, Mu.shape[1] - 2] = Mu[Mu.shape[0] - 2, Mu.shape[1] - 2] - dR[0]

    # Derivative Input Cost
    OffDiaf = -np.tile(dR, N-1)
    np.fill_diagonal(Mu[2:], OffDiaf)
    np.fill_diagonal(Mu[:, 2:], OffDiaf)


    quadLaneSlack = Qlane[0] * np.eye(2*N)
    M00 = linalg.block_diag(Mx, P, Mu)
    M0  = linalg.block_diag(M00, quadLaneSlack)

    xtrack = np.array([vt, 0, 0, 0, 0, 0])
    q_hard = - 2 * np.dot(np.append(np.tile(xtrack, N + 1), np.zeros(R.shape[0] * N)), M00)
    
    linLaneSlack = Qlane[1] * np.ones(2*N)
    q = np.append(q_hard, linLaneSlack)

    M = 2 * M0  # Need to multiply by two because CVX considers 1/2 in front of quadratic cost

    return M, q