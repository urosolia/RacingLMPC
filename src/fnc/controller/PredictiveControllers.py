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
      "timeVarying": False, # False = LTI model, true = LTV model (TO DO)
      "predictiveModel": [] # This object is used to update the LTV prediction model (used if timeVarying == True)
    }
    return mpcPrameters

############################################################################################
####################################### MPC CLASS ##########################################
############################################################################################
class MPC():
    def __init__(self,  mpcPrameters):
        """Initialization
        Arguments:
            mpcPrameters: struct containing MPC parameters
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

        self.slacks          = mpcPrameters['slacks']
        self.timeVarying     = mpcPrameters['timeVarying']
        self.predictiveModel = mpcPrameters['predictiveModel'] 

        if self.timeVarying == True:
            self.xLin = self.predictiveModel.xStored[-1][0:self.N+1,:]
            self.uLin = self.predictiveModel.uStored[-1][0:self.N,:]
            self.computeLTVdynamics()
        
        self.OldInput = np.zeros((1,2)) # TO DO fix size

        # Build matrices for inequality constraints
        self.buildIneqConstr()
        self.buildCost()
        self.buildEqConstr()

        self.xPred = []

        # initialize time
        startTimer = datetime.datetime.now()
        endTimer = datetime.datetime.now(); deltaTimer = endTimer - startTimer
        self.solverTime = deltaTimer
        self.linearizationTime = deltaTimer

    def computeLTVdynamics(self):
        self.A = []; self.B = []; self.C =[]
        for i in range(0, self.N):
            Ai, Bi, Ci = self.predictiveModel.regressionAndLinearization(self.xLin[i], self.uLin[i])
            self.A.append(Ai); self.B.append(Bi); self.C.append(Ci)

    def addTerminalComponents(self):
        # TO DO: ....
        self.H_FTOCP = sparse.csc_matrix(self.H)
        self.q_FTOCP = self.q
        self.F_FTOCP = sparse.csc_matrix(self.F)
        self.b_FTOCP = self.b
        self.G_FTOCP = sparse.csc_matrix(self.G)
        self.E_FTOCP = self.E
        self.L_FTOCP = self.L
        return []

    def solve(self, x0):
        """Computes control action
        Arguments:
            x0: current state
        """
        # If LTV active --> identify system model
        if self.timeVarying == True:
            self.computeLTVdynamics()
            self.buildEqConstr()
        
        self.addTerminalComponents()
        # Solve QP
        startTimer = datetime.datetime.now()
        self.osqp_solve_qp(self.H_FTOCP, self.q_FTOCP, self.F_FTOCP, self.b_FTOCP, self.G_FTOCP, np.add(np.dot(self.E_FTOCP,x0),self.L_FTOCP[:,0]))
        self.unpackSolution()
        endTimer = datetime.datetime.now(); deltaTimer = endTimer - startTimer
        self.solverTime = deltaTimer

        # If LTV active --> compute state-input linearization trajectory
        self.feasibleStateInput()
        if self.timeVarying == True:
            self.xLin = np.vstack((self.xPred[1:, :], self.zt))
            self.uLin = np.vstack((self.uPred[1:, :], self.zt_u))

        # update applied input
        self.OldInput = self.uPred[:,0]

    def feasibleStateInput(self):
        self.zt   = self.xPred[-1,:]
        self.zt_u = self.uPred[-1,:]

    def unpackSolution(self):
        # Extract predicted state and predicted input trajectories
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
            nc_x = self.Fx.shape[0] # add slack only for state constraints
            # Fist add add slack to existing constraints
            addSlack = np.zeros((F_hard.shape[0], nc_x*self.N))
            addSlack[0:nc_x*(self.N), 0:nc_x*(self.N)] = -np.eye(nc_x*(self.N))
            # Now constraint slacks >= 0
            I = - np.eye(nc_x*self.N); Zeros = np.zeros((nc_x*self.N, F_hard.shape[1]))
            Positivity = np.hstack((Zeros, I))

            # Let's stack all together
            self.F = np.vstack(( np.hstack((F_hard, addSlack)) , Positivity))
            self.b = np.hstack((bxtot, butot, np.zeros(nc_x*self.N)))
        else:
            self.F = F_hard
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

        if self.slacks == True:
            self.G = np.hstack( (Gx, Gu, np.zeros( ( Gx.shape[0], self.Fx.shape[0]*self.N) ) ) ) 
        else:
            self.G = np.hstack((Gx, Gu))
    
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
            linSlack  = self.Qslack[1] * np.ones(self.Fx.shape[0]*self.N )
            self.H = linalg.block_diag(Hx, self.Qf, Hu, quadSlack)
            self.q = np.append(q, linSlack)
        else: 
            self.H = linalg.block_diag(Hx, self.Qf, Hu)
            self.q = q 
 
        self.H = 2 * self.H  #  Need to multiply by two because CVX considers 1/2 in front of quadratic cost

    def osqp_solve_qp(self, P, q, G= None, h=None, A=None, b=None, initvals=None):
        """ 
        Solve a Quadratic Program defined as:
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

############## Below LMPC class which is a child of the MPC super class
class ControllerLMPC_child(MPC):
    """Create the LMPC
    Attributes:
        solve: given x0 computes the control action
        addTrajectory: given a ClosedLoopData object adds the trajectory to SS, Qfun, uSS and updates the iteration index
        addPoint: this function allows to add the closed loop data at iteration j to the SS of iteration (j-1)
        update: this function can be used to set SS, Qfun, uSS and the iteration index.
    """

    def __init__(self, numSS_Points, numSS_it, N, QterminalSlack, Qslack, Q, R, dR, map, Laps, TimeLMPC, Solver, inputConstraints, mpcPrameters, dt = 0.1):
        """Initialization
        Arguments:
            numSS_Points: number of points selected from the previous trajectories to build SS
            numSS_it: number of previois trajectories selected to build SS
            N: horizon length
            Q,R: weight to define cost function h(x,u) = ||x||_Q + ||u||_R
            dR: weight to define the input rate cost h(x,u) = ||x_{k+1}-x_k||_dR
            n,d: state and input dimensiton
            shift: given the closest point x_t^j to x(t) the controller start selecting the point for SS from x_{t+shift}^j
            map: map
            Laps: maximum number of laps the controller can run (used to avoid dynamic allocation)
            TimeLMPC: maximum time [s] that an lap can last (used to avoid dynamic allocation)
            Solver: solver used in the reformulation of the LMPC as QP
        """
        super().__init__(mpcPrameters)
        self.numSS_Points = numSS_Points
        self.numSS_it     = numSS_it
        self.N = N
        self.Qslack = Qslack
        self.QterminalSlack = QterminalSlack
        self.Q = Q
        self.R = R
        self.dR = dR
        self.n = Q.shape[1]
        self.d = R.shape[1]
        self.dt = dt
        self.map = map
        self.Solver = Solver
        self.LapTime = 0
        self.itUsedSysID = 1
        self.inputConstraints = inputConstraints

        self.OldInput = np.zeros((1,2))

        # Initialize the following quantities to avoid dynamic allocation
        NumPoints = int(TimeLMPC / dt) + 1
        self.LapCounter = 10000 * np.ones(Laps).astype(int)        # Time at which each j-th iteration is completed
        self.TimeSS     = 10000 * np.ones(Laps).astype(int)        # Time at which each j-th iteration is completed
        self.SS         = 10000 * np.ones((NumPoints, 6, Laps))    # Sampled Safe SS
        self.uSS        = 10000 * np.ones((NumPoints, 2, Laps))    # Input associated with the points in SS
        self.Qfun       =     0 * np.ones((NumPoints, Laps))       # Qfun: cost-to-go from each point in SS
        self.SS_glob    = 10000 * np.ones((NumPoints, 6, Laps))    # SS in global (X-Y) used for plotting

        self.zVector = np.array([0.0, 0.0, 0.0, 0.0, 10.0, 0.0])


        # Initialize the controller iteration
        self.it      = 0

        # Build matrices for inequality constraints

        self.buildIneqConstr()

        self.buildCost()

        self.addSafeSetIneqConstr()

        self.xPred = []

    def addSafeSetIneqConstr(self):
        # Add positiviti constraints for lambda_{SafeSet}. Note that no constraint is enforced on slack_{SafeSet} ---> add np.hstack(-np.eye(self.numSS_Points), np.zeros(self.n)) 
        self.F_FTOCP = sparse.csc_matrix( linalg.block_diag( self.F, np.hstack((-np.eye(self.numSS_Points), np.zeros((self.numSS_Points, self.n)))) ) )
        self.b_FTOCP = np.append(self.b, np.zeros(self.numSS_Points))
    
    def addSafeSetEqConstr(self):
        # Add constrains for x, u, slack
        xTermCons = np.zeros((self.n, self.G.shape[1]))
        xTermCons[:, self.N * self.n:(self.N + 1) * self.n] = np.eye(self.n)
        G_x_u_slack = np.vstack((self.G, xTermCons))
        # Constraint for lambda_{SaFeSet, slack_{safeset}} to enforce safe set
        G_lambda_slackSafeSet = np.vstack( (np.zeros((self.G.shape[0], self.SS_PointSelectedTot.shape[1] + self.n)), np.hstack((-self.SS_PointSelectedTot, np.eye(self.n)))) )
        # Constraints on lambda = 1
        G_lambda = np.append(np.append(np.zeros(self.G.shape[1]), np.ones(self.SS_PointSelectedTot.shape[1])), np.zeros(self.n))
        # Put all together
        self.G_FTOCP = sparse.csc_matrix(np.vstack((np.hstack((G_x_u_slack, G_lambda_slackSafeSet)), G_lambda)))
        self.E_FTOCP = np.vstack((self.E, np.zeros((self.n+1,self.n)))) # adding n for terminal constraint and 1 for lambda = 1
        self.L_FTOCP = np.append(np.append(self.L, np.zeros(self.n)), 1)

    def addSafeSetCost(self):
        # need to multiply the quadratic term as cost is (1/2) z'*Q*z
        self.H_FTOCP = sparse.csc_matrix(linalg.block_diag(self.H, np.zeros((self.SS_PointSelectedTot.shape[1], self.SS_PointSelectedTot.shape[1])), 2*self.QterminalSlack) )
        self.q_FTOCP = np.append(np.append(self.q, self.Qfun_SelectedTot), np.zeros(self.n))

    def solve(self, x0, uOld = np.zeros([0, 0])):
        """Computes control action
        Arguments:
            x0: current state position
        """
        # vector z = [ x_{0:N+1}, u_{0:N}, slack_{0:N}, lambda_{SafeSet}, slack_{SafeSet}]

        n = self.n;        d = self.d
        F = self.F;        b = self.b
        SS = self.SS;      Qfun = self.Qfun
        uSS = self.uSS;    TimeSS = self.TimeSS
        Q = self.Q;        R = self.R
        dR =self.dR;       OldInput = self.OldInput
        N = self.N; dt = self.dt
        it           = self.it
        numSS_Points = self.numSS_Points
        Qslack       = self.Qslack

        LinPoints = self.LinPoints
        LinInput  = self.LinInput
        map = self.map

        # Select Points from SS
        if (self.zVector[4]-x0[4] > map.TrackLength/2):
            self.zVector[4] = np.max([self.zVector[4] - map.TrackLength,0])
            self.LinPoints[4,-1] = self.LinPoints[4,-1]- map.TrackLength


        sortedLapTime = np.argsort(self.Qfun[0, 0:it])

        SS_PointSelectedTot = np.empty((n, 0))
        Succ_SS_PointSelectedTot = np.empty((n, 0))
        Succ_uSS_PointSelectedTot = np.empty((d, 0))
        Qfun_SelectedTot = np.empty((0))
        for jj in sortedLapTime[0:self.numSS_it]:
            SS_PointSelected, uSS_PointSelected, Qfun_Selected = _SelectPoints(self, jj, self.zVector, numSS_Points / self.numSS_it + 1)
            Succ_SS_PointSelectedTot =  np.append(Succ_SS_PointSelectedTot, SS_PointSelected[:,1:], axis=1)
            Succ_uSS_PointSelectedTot =  np.append(Succ_uSS_PointSelectedTot, uSS_PointSelected[:,1:], axis=1)
            SS_PointSelectedTot      = np.append(SS_PointSelectedTot, SS_PointSelected[:,0:-1], axis=1)
            Qfun_SelectedTot         = np.append(Qfun_SelectedTot, Qfun_Selected[0:-1], axis=0)

        self.SS_PointSelectedTot = SS_PointSelectedTot
        self.Qfun_SelectedTot = Qfun_SelectedTot

        startTimer = datetime.datetime.now()
        # print("======= Start")
        # print("========= Start")
        # Atv, Btv, Ctv, indexUsed_list = _LMPC_EstimateABC(self, sortedLapTime)
        endTimer = datetime.datetime.now(); deltaTimer = endTimer - startTimer
        # L, npG, npE = _LMPC_BuildMatEqConst(self, Atv, Btv, Ctv, N, n, d)
        self.linearizationTime = deltaTimer

        # self.buildIneqConstr()
        # self.F = self.F.todense()
        self.computeLTVdynamics()
        self.buildEqConstr()
        # if (self.A[0] != Atv[0]).any():
        #     pdb.set_trace()
        # print("========= End")

        self.buildCost() # Need to build cost because of the input rate
        # print("======= End")
        # pdb.set_trace()
        # self.addSafeSetIneqConstr()
        # Build Terminal cost and Constraint
        # self.G = npG
        # self.L = L
        # self.E = npE
        self.addSafeSetEqConstr()
        self.addSafeSetCost()
        # G, E = _LMPC_TermConstr(self, npG, npE, N, n, d, SS_PointSelectedTot)
        # M, q = _LMPC_BuildMatCost(self, Qfun_SelectedTot, numSS_Points, N, Qslack, Q, R, dR, OldInput)

        # Solve QP
        startTimer = datetime.datetime.now()
        # res_cons, feasible = osqp_solve_qp(sparse.csr_matrix(M), q, sparse.csr_matrix(F), b, sparse.csr_matrix(G), np.add(np.dot(E,x0),L[:,0]))
        res_cons, feasible = osqp_solve_qp(self.H_FTOCP, self.q_FTOCP, self.F_FTOCP, self.b_FTOCP, self.G_FTOCP, np.add(np.dot(self.E_FTOCP,x0),self.L_FTOCP))
        Solution = res_cons.x

        self.feasible = feasible

        endTimer = datetime.datetime.now(); deltaTimer = endTimer - startTimer
        self.solverTime = deltaTimer

        # Extract solution and set linerizations points
        xPred, uPred, lambd, slack = _LMPC_GetPred(Solution, n, d, N, np, self)

        self.zVector = np.dot(Succ_SS_PointSelectedTot, lambd)
        self.uVector = np.dot(Succ_uSS_PointSelectedTot, lambd)

        self.xPred = xPred.T
        if self.N == 1:
            self.uPred    = np.array([[uPred[0], uPred[1]]])
            self.LinInput =  np.array([[uPred[0], uPred[1]]])
        else:
            self.uPred = uPred.T
            self.LinInput = np.vstack((uPred.T[1:, :], self.uVector))

        self.LinPoints = np.vstack((xPred.T[1:,:], self.zVector))

        self.xLin = self.LinPoints
        self.uLin = self.LinInput 

        self.OldInput = uPred.T[0,:]

    def addTrajectory(self, x, u, x_glob):
        """update iteration index and construct SS, uSS and Qfun
        Arguments:
            ClosedLoopData: ClosedLoopData object
        """
        it = self.it
        self.TimeSS[it] = x.shape[0]
        self.LapCounter[it] = x.shape[0]
        self.SS[0:(self.TimeSS[it]), :, it] = x
        self.SS_glob[0:(self.TimeSS[it]), :, it] = x_glob
        self.uSS[0:self.TimeSS[it], :, it]      = u
        self.Qfun[0:(self.TimeSS[it]), it]  = _ComputeCost(x,u, self.map.TrackLength)

        for i in np.arange(0, self.Qfun.shape[0]):
            if self.Qfun[i, it] == 0:
                self.Qfun[i, it] = self.Qfun[i - 1, it] - 1

        if self.it == 0:
            self.LinPoints = self.SS[1:self.N + 2, :, it]
            self.LinInput  = self.uSS[1:self.N + 1, :, it]
            self.xLin = self.LinPoints
            self.uLin = self.LinInput

        self.it = self.it + 1

    def addPoint(self, x, u):
        """at iteration j add the current point to SS, uSS and Qfun of the previous iteration
        Arguments:
            x: current state
            u: current input
            i: at the j-th iteration i is the time at which (x,u) are recorded
        """
        Counter = self.TimeSS[self.it - 1]
        self.SS[Counter, :, self.it - 1] = x + np.array([0, 0, 0, 0, self.map.TrackLength, 0])
        self.uSS[Counter, :, self.it - 1] = u
        
        # The above two lines are needed as the once the predicted trajectory has crossed the finish line the goal is
        # to reach the end of the lap which is about to start
        if self.Qfun[Counter, self.it - 1] == 0:
            self.Qfun[Counter, self.it - 1] = self.Qfun[Counter, self.it - 1] - 1
        self.TimeSS[self.it - 1] = self.TimeSS[self.it - 1] + 1

# ======================================================================================================================
# ======================================================================================================================
# =============================== Internal functions for LMPC reformulation to QP ======================================
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
    osqp2 = OSQP()
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
        osqp2.setup(P=P, q=q, A=qp_A, l=qp_l, u=qp_u, verbose=False, polish=True)
    else:
        osqp2.setup(P=P, q=q, A=None, l=None, u=None, verbose=False)
    if initvals is not None:
        osqp2.warm_start(x=initvals)
    res = osqp2.solve()

    feasible = 0
    if res.info.status_val == 1:
        feasible = 1
    return res, feasible


def _SelectPoints(LMPC, it, x0, numSS_Points):
    SS  = LMPC.SS
    uSS = LMPC.uSS
    TimeSS = LMPC.TimeSS
    SS_glob = LMPC.SS_glob
    Qfun = LMPC.Qfun
    xPred = LMPC.xPred
    map = LMPC.map
    TrackLength = map.TrackLength
    currIt = LMPC.it

    x = SS[:, 0:(TimeSS[it]-1), it]
    u = uSS[:, 0:(TimeSS[it]-1), it]
    x_glob = SS_glob[:, :, it]
    oneVec = np.ones((x.shape[0], 1))
    x0Vec = (np.dot(np.array([x0]).T, oneVec.T)).T
    diff = x - x0Vec
    norm = la.norm(diff, 1, axis=1)
    MinNorm = np.argmin(norm)


    if (MinNorm - numSS_Points/2 >= 0):
        # print("one")
        # print(numSS_Points, MinNorm)
        indexSSandQfun = range(-int(numSS_Points/2) + MinNorm, int(numSS_Points/2) + MinNorm + 1)
        # print(indexSSandQfun)
        # pdb.set_trace()
    else:
        # print("two")
        # print(numSS_Points)
        # pdb.set_trace()
        indexSSandQfun = range(MinNorm, MinNorm + int(numSS_Points))

    SS_Points  = x[indexSSandQfun, :].T
    SSu_Points = u[indexSSandQfun, :].T
    SS_glob_Points = x_glob[indexSSandQfun, :].T
    Sel_Qfun = Qfun[indexSSandQfun, it]

    # Modify the cost if the predicion has crossed the finisch line
    if xPred == []:
        Sel_Qfun = Qfun[indexSSandQfun, it]
    elif (np.all((xPred[:, 4] > TrackLength) == False)):
        Sel_Qfun = Qfun[indexSSandQfun, it]
    elif it < currIt - 1:
        Sel_Qfun = Qfun[indexSSandQfun, it] + Qfun[0, it + 1]
    else:
        sPred = xPred[:, 4]
        predCurrLap = LMPC.N - sum(sPred > TrackLength)
        currLapTime = LMPC.LapTime
        Sel_Qfun = Qfun[indexSSandQfun, it] + currLapTime + predCurrLap

    return SS_Points, SSu_Points, Sel_Qfun

def _ComputeCost(x, u, TrackLength):
    Cost = 10000 * np.ones((x.shape[0]))  # The cost has the same elements of the vector x --> time +1
    # Now compute the cost moving backwards in a Dynamic Programming (DP) fashion.
    # We start from the last element of the vector x and we sum the running cost
    for i in range(0, x.shape[0]):
        if (i == 0):  # Note that for i = 0 --> pick the latest element of the vector x
            Cost[x.shape[0] - 1 - i] = 0
        elif x[x.shape[0] - 1 - i, 4]< TrackLength:
            Cost[x.shape[0] - 1 - i] = Cost[x.shape[0] - 1 - i + 1] + 1
        else:
            Cost[x.shape[0] - 1 - i] = 0

    return Cost



def _LMPC_GetPred(Solution,n,d,N, np, LMPC):
    stateIdx = n*(N+1)
    inputIdx = stateIdx + d*N
    slackIdx = inputIdx + LMPC.Fx.shape[0]*N
    lambdIdx = slackIdx + LMPC.SS_PointSelectedTot.shape[1]
    sTermIdx = lambdIdx + n

    xPred = np.squeeze(np.transpose(np.reshape((Solution[np.arange(n*(N+1))]),(N+1,n))))
    uPred = np.squeeze(np.transpose(np.reshape((Solution[n*(N+1)+np.arange(d*N)]),(N, d))))
    slack = Solution[inputIdx:slackIdx]
    lambd = Solution[slackIdx:lambdIdx]
    slackTerminal = Solution[lambdIdx:]

    # print np.sum(np.abs(laneSlack))
    # if np.sum(np.abs(laneSlack)) > 0.5:
    #     pdb.set_trace()

    return xPred, uPred, lambd, slack

def _LMPC_BuildMatEqConst(LMPC, A, B, C, N, n, d):
    # Buil matrices for optimization (Convention from Chapter 15.2 Borrelli, Bemporad and Morari MPC book)
    # We are going to build our optimization vector z \in \mathbb{R}^((N+1) \dot n \dot N \dot d), note that this vector
    # stucks the predicted trajectory x_{k|t} \forall k = t, \ldots, t+N+1 over the horizon and
    # the predicte input u_{k|t} \forall k = t, \ldots, t+N over the horizon
    Gx = np.eye(n * (N + 1))
    Gu = np.zeros((n * (N + 1), d * (N)))

    E = np.zeros((n * (N + 1), n))
    E[np.arange(n)] = np.eye(n)

    L = np.zeros((n * (N + 1), 1)) # n+1 for the terminal constraint

    for i in range(0, N):
        ind1 = n + i * n + np.arange(n)
        ind2x = i * n + np.arange(n)
        ind2u = i * d + np.arange(d)

        Gx[np.ix_(ind1, ind2x)] = -A[i]
        Gu[np.ix_(ind1, ind2u)] = -B[i]
        L[ind1, :]              =  C[i]

    G = np.hstack( (Gx, Gu, np.zeros( ( Gx.shape[0], LMPC.Fx.shape[0]*LMPC.N) ) ) )

    L_return = L

    return L_return, G, E
# ======================================================================================================================
# ======================================================================================================================
# ========================= Internal functions for Local Regression and Linearization ==================================
# ======================================================================================================================
# ======================================================================================================================
def _LMPC_EstimateABC(ControllerLMPC, sortedLapTime):
    LinPoints       = ControllerLMPC.LinPoints
    LinInput        = ControllerLMPC.LinInput
    N               = ControllerLMPC.N
    n               = ControllerLMPC.n
    d               = ControllerLMPC.d
    TimeSS          = ControllerLMPC.TimeSS
    LapCounter      = ControllerLMPC.LapCounter
    PointAndTangent = ControllerLMPC.map.PointAndTangent
    dt              = ControllerLMPC.dt
    it              = ControllerLMPC.it
    SS              = ControllerLMPC.SS
    uSS             = ControllerLMPC.uSS

    ParallelComputation = 0
    Atv = []; Btv = []; Ctv = []; indexUsed_list = []

    usedIt = sortedLapTime[0:ControllerLMPC.itUsedSysID] # range(ControllerLMPC.it-ControllerLMPC.itUsedSysID, ControllerLMPC.it)
    MaxNumPoint = 40  # Need to reason on how these points are selected


    for i in range(0, N):
       Ai, Bi, Ci, indexSelected = RegressionAndLinearization(LinPoints, LinInput, usedIt, SS, uSS, TimeSS,
                                                           MaxNumPoint, qp, n, d, matrix, PointAndTangent, dt, i)
       Atv.append(Ai)
       Btv.append(Bi)
       Ctv.append(Ci)
       indexUsed_list.append(indexSelected)

    return Atv, Btv, Ctv, indexUsed_list

def RegressionAndLinearization(LinPoints, LinInput, usedIt, SS, uSS, LapCounter, MaxNumPoint, qp, n, d, matrix, PointAndTangent, dt, i):


    x0 = LinPoints[i, :]

    Ai = np.zeros((n, n))
    Bi = np.zeros((n, d))
    Ci = np.zeros((n, 1))

    # Compute Index to use
    h = 5
    lamb = 0.0
    stateFeatures = [0, 1, 2]
    ConsiderInput = 1

    if ConsiderInput == 1:
        scaling = np.array([[0.1, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 1.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 1.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 1.0]])
        xLin = np.hstack((LinPoints[i, stateFeatures], LinInput[i, :]))
    else:
        scaling = np.array([[1.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0],
                            [0.0, 0.0, 1.0]])
        xLin = LinPoints[i, stateFeatures]

    indexSelected = []
    K = []
    for ii in usedIt:
        # print("it: ", ii)
        indexSelected_i, K_i = ComputeIndex(h, SS, uSS, LapCounter, ii, xLin, stateFeatures, scaling, MaxNumPoint,
                                            ConsiderInput)
        indexSelected.append(indexSelected_i)
        K.append(K_i)
    # print("xLin: ", xLin)
    # print("indexSelected: ", indexSelected)
    # =========================
    # ====== Identify vx ======
    inputFeatures = [1]
    Q_vx, M_vx = Compute_Q_M(SS, uSS, indexSelected, stateFeatures, inputFeatures, usedIt, np, matrix, lamb, K)

    yIndex = 0
    b = Compute_b(SS, yIndex, usedIt, matrix, M_vx, indexSelected, K, np)
    Ai[yIndex, stateFeatures], Bi[yIndex, inputFeatures], Ci[yIndex] = LMPC_LocLinReg(Q_vx, b, stateFeatures,
                                                                                      inputFeatures, qp)

    # =======================================
    # ====== Identify Lateral Dynamics ======
    inputFeatures = [0]
    Q_lat, M_lat = Compute_Q_M(SS, uSS, indexSelected, stateFeatures, inputFeatures, usedIt, np, matrix, lamb, K)

    yIndex = 1  # vy
    b = Compute_b(SS, yIndex, usedIt, matrix, M_lat, indexSelected, K, np)
    Ai[yIndex, stateFeatures], Bi[yIndex, inputFeatures], Ci[yIndex] = LMPC_LocLinReg(Q_lat, b, stateFeatures,
                                                                                      inputFeatures, qp)

    yIndex = 2  # wz
    b = Compute_b(SS, yIndex, usedIt, matrix, M_lat, indexSelected, K, np)
    Ai[yIndex, stateFeatures], Bi[yIndex, inputFeatures], Ci[yIndex] = LMPC_LocLinReg(Q_lat, b, stateFeatures,
                                                                                      inputFeatures, qp)

    # ===========================
    # ===== Linearization =======
    vx = x0[0]; vy   = x0[1]
    wz = x0[2]; epsi = x0[3]
    s  = x0[4]; ey   = x0[5]

    if s < 0:
        print("s is negative, here the state: \n", LinPoints)

    startTimer = datetime.datetime.now()  # Start timer for LMPC iteration
    cur = Curvature(s, PointAndTangent)
    cur = Curvature(s, PointAndTangent)
    den = 1 - cur * ey
    # ===========================
    # ===== Linearize epsi ======
    # epsi_{k+1} = epsi + dt * ( wz - (vx * np.cos(epsi) - vy * np.sin(epsi)) / (1 - cur * ey) * cur )
    depsi_vx = -dt * np.cos(epsi) / den * cur
    depsi_vy = dt * np.sin(epsi) / den * cur
    depsi_wz = dt
    depsi_epsi = 1 - dt * (-vx * np.sin(epsi) - vy * np.cos(epsi)) / den * cur
    depsi_s = 0  # Because cur = constant
    depsi_ey = dt * (vx * np.cos(epsi) - vy * np.sin(epsi)) / (den ** 2) * cur * (-cur)

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

    endTimer = datetime.datetime.now(); deltaTimer_tv = endTimer - startTimer

    return Ai, Bi, Ci, indexSelected

def Compute_Q_M(SS, uSS, indexSelected, stateFeatures, inputFeatures, usedIt, np, matrix, lamb, K):
    Counter = 0
    it = 1
    X0   = np.empty((0,len(stateFeatures)+len(inputFeatures)))
    Ktot = np.empty((0))

    for it in usedIt:
        X0 = np.append( X0, np.hstack((np.squeeze(SS[np.ix_(indexSelected[Counter], stateFeatures, [it])]),
                            np.squeeze(uSS[np.ix_(indexSelected[Counter], inputFeatures, [it])], axis=2))), axis=0)
        Ktot = np.append(Ktot, K[Counter])
        Counter = Counter + 1

    M = np.hstack((X0, np.ones((X0.shape[0], 1))))
    Q0 = np.dot(np.dot(M.T, np.diag(Ktot)), M)
    Q = matrix(Q0 + lamb * np.eye(Q0.shape[0]))


    return Q, M

def Compute_b(SS, yIndex, usedIt, matrix, M, indexSelected, K, np):
    Counter = 0
    y = np.empty((0))
    Ktot = np.empty((0))

    for it in usedIt:
        y = np.append(y, np.squeeze(SS[np.ix_(indexSelected[Counter] + 1, [yIndex], [it])]))
        Ktot = np.append(Ktot, K[Counter])
        Counter = Counter + 1

    b = matrix(-np.dot(np.dot(M.T, np.diag(Ktot)), y))

    return b

def LMPC_LocLinReg(Q, b, stateFeatures, inputFeatures, qp):
    import numpy as np
    from numpy import linalg as la
    import datetime

    startTimer = datetime.datetime.now()  # Start timer for LMPC iteration
    res_cons = qp(Q, b) # This is ordered as [A B C]

    endTimer = datetime.datetime.now(); deltaTimer_tv = endTimer - startTimer
    Result = np.squeeze(np.array(res_cons['x']))
    A = Result[0:len(stateFeatures)]
    B = Result[len(stateFeatures):(len(stateFeatures)+len(inputFeatures))]
    C = Result[-1]

    return A, B, C

def ComputeIndex(h, SS, uSS, LapCounter, it, x0, stateFeatures, scaling, MaxNumPoint, ConsiderInput):
    import numpy as np
    from numpy import linalg as la
    import datetime



    startTimer = datetime.datetime.now()  # Start timer for LMPC iteration

    # What to learn a model such that: x_{k+1} = A x_k  + B u_k + C
    oneVec = np.ones( (SS[0:LapCounter[it], :, it].shape[0]-1, 1) )

    x0Vec = (np.dot( np.array([x0]).T, oneVec.T )).T

    if ConsiderInput == 1:
        DataMatrix = np.hstack((SS[0:LapCounter[it]-1, stateFeatures, it], uSS[0:LapCounter[it]-1, :, it]))
    else:
        DataMatrix = SS[0:LapCounter[it]-1, stateFeatures, it]

    # np.savetxt('A.csv', SS[0:TimeSS[it]-1, stateFeatures, it], delimiter=',', fmt='%f')
    # np.savetxt('B.csv', SS[0:TimeSS[it], :, it][0:-1, stateFeatures], delimiter=',', fmt='%f')
    # np.savetxt('SS.csv', SS[0:TimeSS[it], :, it], delimiter=',', fmt='%f')

    diff  = np.dot(( DataMatrix - x0Vec ), scaling)
    # print 'x0Vec \n',x0Vec
    norm = la.norm(diff, 1, axis=1)
    indexTot =  np.squeeze(np.where(norm < h))
    # print indexTot.shape, np.argmin(norm), norm, x0
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
    if norm.shape[0] < 500:
        print(LapCounter[it])
        print("norm: ", norm, norm.shape)
    # K = np.ones(len(index))

    return index, K