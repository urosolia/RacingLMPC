import pdb
import numpy as np
from cvxopt import spmatrix, matrix, solvers
from numpy import linalg as la
from scipy import linalg
from scipy import sparse
from cvxopt.solvers import qp
import datetime
from Utilities import Curvature
from numpy import hstack, inf, ones
from scipy.sparse import vstack
from osqp import OSQP

solvers.options['show_progress'] = False

# MPC paramters struct
def MPC_parameters(n, d, N):
    mpcPrameters = {
      "n": n, # dimension state space
      "d": d, # dimension input space
      "N": N, # horizon length
      "A": np.zeros((n,n)), # prediction matrices. Single matrix for LTI and list for LTV
      "B": np.zeros((n,d)), # prediction matrices. Single matrix for LTI and list for LTV
      "Q": np.zeros((n,n)), # quadratic state cost
      "Qf": np.zeros((n,n)), # quadratic state cost final
      "R": np.zeros((d,d)), # quadratic input cost
      "dR": np.zeros(d), # rate cost. It has to be a vector and penalized the rate for each input component
      "Qslack": np.zeros(2), # it has to be a vector. Qslack = [linearSlackCost, quadraticSlackCost]
      "Fx": [], # State constraint Fx * x <= bx
      "bx": [],
      "Fu": [], # State constraint Fu * u <= bu
      "bu": [],
      "xRef": np.zeros(n), # reference point # TO DO: add LTV option
      "slacks": False, # Flase = no slack variable for state constraints, True = Use slack for input constraints
      "timeVarying": False # False = LTI model, true = LTV model (TO DO)
    }
    return mpcPrameters

############################################################################################
####################################### MPC CLASS ##########################################
############################################################################################
class MPC():
    def __init__(self,  mpcPrameters):
        """Initialization
        Arguments:
            N: horizon length
            Q,R: weight to define cost function h(x,u) = ||x||_Q + ||u||_R
            dR: weight to define the input rate cost h(x,u) = ||x_{k+1}-x_k||_dR
            n,d: state and input dimensiton
            map: map
            Solver: solver used in the reformulation of the LMPC as QP
        """
        self.N      = mpcPrameters['N']
        self.Qslack = mpcPrameters['Qslack']
        self.Q      = mpcPrameters['Q']
        self.Qf     = mpcPrameters['Qf']
        self.R      = mpcPrameters['R']
        self.dR     = mpcPrameters['dR']
        self.n      = mpcPrameters['n']
        self.d      = mpcPrameters['d']
        self.A      = mpcPrameters['A']
        self.B      = mpcPrameters['B']
        self.Fx     = mpcPrameters['Fx']
        self.Fu     = mpcPrameters['Fu']
        self.bx     = mpcPrameters['bx']
        self.bu     = mpcPrameters['bu']
        self.xRef   = mpcPrameters['xRef']

        self.slacks      = mpcPrameters['slacks']
        self.timeVarying = mpcPrameters['timeVarying']
        
        self.OldInput = np.zeros((1,2))



        # Build matrices for inequality constraints
        self.buildIneqConstr()
        self.buildEqConstr()
        self.buildCost()

        self.xPred = []

        # initialize time
        startTimer = datetime.datetime.now()
        endTimer = datetime.datetime.now(); deltaTimer = endTimer - startTimer
        self.solverTime = deltaTimer
        self.linearizationTime = deltaTimer

    def solve(self, x0):
        """Computes control action
        Arguments:
            x0: current state
        """
        # Solve QP
        startTimer = datetime.datetime.now()
        self.osqp_solve_qp(self.H, self.q, self.F, self.b, self.G, np.add(np.dot(self.E,x0),self.L[:,0]))
        self.unpackSolution()
        endTimer = datetime.datetime.now(); deltaTimer = endTimer - startTimer
        self.solverTime = deltaTimer

        self.OldInput = self.uPred[:,0]

    def unpackSolution(self):
        self.xPred = np.squeeze(np.transpose(np.reshape((self.Solution[np.arange(self.n*(self.N+1))]),(self.N+1,self.n)))).T
        self.uPred = np.squeeze(np.transpose(np.reshape((self.Solution[self.n*(self.N+1)+np.arange(self.d*self.N)]),(self.N, self.d)))).T
        
    def buildIneqConstr(self):
        # The inequality constraint is Fz<=b
        # Let's start by computing the submatrix of F relates with the state
        rep_a = [self.Fx] * (self.N)
        Mat = linalg.block_diag(*rep_a)
        NoTerminalConstr = np.zeros((np.shape(Mat)[0], self.n))  # The last state is unconstrained. There is a specific function add the terminal constraints (so that more complicated terminal constrains can be handled)
        Fxtot = np.hstack((Mat, NoTerminalConstr))
        bxtot = np.tile(np.squeeze(self.bx), self.N)

        # Let's start by computing the submatrix of F relates with the input
        rep_b = [self.Fu] * (self.N)
        Futot = linalg.block_diag(*rep_b)
        butot = np.tile(np.squeeze(self.bu), self.N)

        # Let's stack all together
        F_hard = linalg.block_diag(Fxtot, Futot)

        # Add slack if need
        if self.slacks == True:
            nc = self.F.shape[0]
            # Fist add add slack to existing constraints
            addSlack = np.zeros((F_hard.shape[0], nc*(N+1)))
            addSlack[0:nc*(N+1), 0:nc*(N+1)] = -np.eye(nc*(N+1))
            # Now constraint slacks >= 0
            I = - np.eye(nc*N); Zeros = np.zeros((nc*N, F_hard.shape[1]))
            Positivity = np.hstack((Zeros, I))

            # Let's stack all together
            self.F = sparse.csc_matrix(np.vstack(( np.hstack((F_hard, addSlack)) , Positivity)))
            self.b = np.hstack((bxtot, butot, np.zeros(nc*(N+1))))
        else:
            self.F = sparse.csc_matrix(F_hard)
            self.b = np.hstack((bxtot, butot))

    def buildEqConstr(self):
        # Buil matrices for optimization (Convention from Chapter 15.2 Borrelli, Bemporad and Morari MPC book)
        # The equality constraint is: G*z = E * x(t) + L
        Gx = np.eye(self.n * (self.N + 1))
        Gu = np.zeros((self.n * (self.N + 1), self.d * (self.N)))

        E = np.zeros((self.n * (self.N + 1), self.n))
        E[np.arange(self.n)] = np.eye(self.n)

        L = np.zeros((self.n * (self.N + 1), 1))

        for i in range(0, self.N):
            if self.timeVarying == True:
                Gx[(self.n + i*self.n):(self.n + i*self.n + self.n), (i*self.n):(i*self.n + self.n)] = -self.A[i]
                Gu[(self.n + i*self.n):(self.n + i*self.n + self.n), (i*self.d):(i*self.d + self.d)] = -self.B[i]
                L[(self.n + i*self.n):(self.n + i*self.n + self.n)]                                  =  self.C[i]
            else:
                Gx[(self.n + i*self.n):(self.n + i*self.n + self.n), (i*self.n):(i*self.n + self.n)] = -self.A
                Gu[(self.n + i*self.n):(self.n + i*self.n + self.n), (i*self.d):(i*self.d + self.d)] = -self.B

        self.G = sparse.csc_matrix(np.hstack((Gx, Gu)))
        self.E = E
        self.L = L

    def buildCost(self):
        # The cost is: (1/2) * z' H z + q' z
        listQ = [self.Q] * (self.N)
        Hx = linalg.block_diag(*listQ)

        listTotR = [self.R + 2 * np.diag(self.dR)] * (self.N) # Need to add dR for the derivative input cost
        Hu = linalg.block_diag(*listTotR)
        # Need to condider that the last input appears just once in the difference
        for i in range(0, self.d):
            Hu[ i - self.d, i - self.d] = Hu[ i - self.d, i - self.d] - self.dR[i]

        # Derivative Input Cost
        OffDiaf = -np.tile(self.dR, self.N-1)
        np.fill_diagonal(Hu[self.d:], OffDiaf)
        np.fill_diagonal(Hu[:, self.d:], OffDiaf)
        
        # Cost linear term for state and input
        q = - 2 * np.dot(np.append(np.tile(self.xRef, self.N + 1), np.zeros(self.R.shape[0] * self.N)), linalg.block_diag(Hx, self.Qf, Hu))
        # Derivative Input (need to consider input at previous time step)
        q[self.n*(self.N+1):self.n*(self.N+1)+self.d] = -2 * np.dot( self.OldInput, np.diag(self.dR) )
        if self.slacks == True: 
            quadSlack = self.Qslack[0] * np.eye(self.Fx.shape[0]*self.N)
            linSlack = self.Qslack[1] * np.ones( self.Fx.shape[0]*self.N )
            self.H = sparse.csc_matrix(linalg.block_diag(Hx, self.Qf, Hu, *quadSlack))
            self.q = np.append(q, linLaneSlack)
        else: 
            self.H = sparse.csc_matrix(linalg.block_diag(Hx, self.Qf, Hu))
            self.q = q 
 
        self.H = 2 * self.H  #  Need to multiply by two because CVX considers 1/2 in front of quadratic cost

    def osqp_solve_qp(self, P, q, G= None, h=None, A=None, b=None, initvals=None):
        """ 
        Solve a Quadratic Prog ram defined as:
        minimize
            (1/2) * x.T * P * x + q.T * x
        subject to
            G * x <= h
            A * x == b
        using OSQP <https://github.com/oxfordcontrol/osqp>.
        """  
        self.osqp = OSQP()
        qp_A = vstack([G, A]).tocsc()
        l = -inf * ones(len(h))
        qp_l = hstack([l, b])
        qp_u = hstack([h, b])

        self.osqp.setup(P=P, q=q, A=qp_A, l=qp_l, u=qp_u, verbose=False, polish=True)
        if initvals is not None:
            self.osqp.warm_start(x=initvals)
        res = self.osqp.solve()
        if res.info.status_val == 1:
            self.feasible = 1
        else:
            self.feasible = 0
        self.Solution = res.x

