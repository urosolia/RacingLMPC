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

    def addTerminalComponents(self, x0):
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
            self.buildCost()
            self.buildEqConstr()
        
        self.addTerminalComponents(x0)
        # Solve QP
        startTimer = datetime.datetime.now()
        self.osqp_solve_qp(self.H_FTOCP, self.q_FTOCP, self.F_FTOCP, self.b_FTOCP, self.G_FTOCP, np.add(np.dot(self.E_FTOCP,x0),self.L_FTOCP))
        self.unpackSolution()
        endTimer = datetime.datetime.now(); deltaTimer = endTimer - startTimer
        self.solverTime = deltaTimer

        # If LTV active --> compute state-input linearization trajectory
        self.feasibleStateInput()
        if self.timeVarying == True:
            self.xLin = np.vstack((self.xPred[1:, :], self.zt))
            self.uLin = np.vstack((self.uPred[1:, :], self.zt_u))

        # update applied input
        self.OldInput = self.uPred[0,:]
        # print(self.uPred)
        # if self.feasible == 0:
        #     pdb.set_trace()

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

        L = np.zeros(self.n * (self.N + 1))

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
class LMPC(MPC):
    """Create the LMPC
    Attributes:
        solve: given x0 computes the control action
        addTrajectory: given a ClosedLoopData object adds the trajectory to SS, Qfun, uSS and updates the iteration index
        addPoint: this function allows to add the closed loop data at iteration j to the SS of iteration (j-1)
        update: this function can be used to set SS, Qfun, uSS and the iteration index.
    """

    def __init__(self, numSS_Points, numSS_it, QterminalSlack, TimeLMPC, Laps, mpcPrameters, dt = 0.1):
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
        self.QterminalSlack = QterminalSlack
        self.LapTime = 0

        self.OldInput = np.zeros((1,2))

        # Initialize the following quantities to avoid dynamic allocation
        NumPoints = int(TimeLMPC / dt) + 1
        self.LapCounter = 10000 * np.ones(Laps).astype(int)        # Time at which each j-th iteration is completed
        self.TimeSS     = 10000 * np.ones(Laps).astype(int)        # Time at which each j-th iteration is completed
        self.SS         = 10000 * np.ones((NumPoints, 6, Laps))    # Sampled Safe SS
        self.uSS        = 10000 * np.ones((NumPoints, 2, Laps))    # Input associated with the points in SS
        self.Qfun       =     0 * np.ones((NumPoints, Laps))       # Qfun: cost-to-go from each point in SS
        self.SS_glob    = 10000 * np.ones((NumPoints, 6, Laps))    # SS in global (X-Y) used for plotting

        self.zt = np.array([0.0, 0.0, 0.0, 0.0, 10.0, 0.0])

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

   
    def unpackSolution(self):
        stateIdx = self.n*(self.N+1)
        inputIdx = stateIdx + self.d*self.N
        slackIdx = inputIdx + self.Fx.shape[0]*self.N
        lambdIdx = slackIdx + self.SS_PointSelectedTot.shape[1]
        sTermIdx = lambdIdx + self.n

        self.xPred = np.squeeze(np.transpose(np.reshape((self.Solution[np.arange(self.n*(self.N+1))]),(self.N+1,self.n)))).T
        self.uPred = np.squeeze(np.transpose(np.reshape((self.Solution[self.n*(self.N+1)+np.arange(self.d*self.N)]),(self.N, self.d)))).T
        self.slack = self.Solution[inputIdx:slackIdx]
        self.lambd = self.Solution[slackIdx:lambdIdx]
        self.slackTerminal = self.Solution[lambdIdx:]


    def feasibleStateInput(self):
        self.zt = np.dot(self.Succ_SS_PointSelectedTot, self.lambd)
        self.zt_u = np.dot(self.Succ_uSS_PointSelectedTot, self.lambd)

    def addTerminalComponents(self,x0):
        # TO DO: add description
        # Select Points from SS
        if (self.zt[4]-x0[4] > self.predictiveModel.map.TrackLength/2):
            self.zt[4] = np.max([self.zt[4] - self.predictiveModel.map.TrackLength,0])
            self.xLin[4,-1] = self.xLin[4,-1]- self.predictiveModel.map.TrackLength

        sortedLapTime = np.argsort(self.Qfun[0, 0:self.it])

        SS_PointSelectedTot = np.empty((self.n, 0))
        Succ_SS_PointSelectedTot = np.empty((self.n, 0))
        Succ_uSS_PointSelectedTot = np.empty((self.d, 0))
        Qfun_SelectedTot = np.empty((0))
        for jj in sortedLapTime[0:self.numSS_it]:
            SS_PointSelected, uSS_PointSelected, Qfun_Selected = self.selectPoints(jj, self.zt, self.numSS_Points / self.numSS_it + 1)
            Succ_SS_PointSelectedTot =  np.append(Succ_SS_PointSelectedTot, SS_PointSelected[:,1:], axis=1)
            Succ_uSS_PointSelectedTot =  np.append(Succ_uSS_PointSelectedTot, uSS_PointSelected[:,1:], axis=1)
            SS_PointSelectedTot      = np.append(SS_PointSelectedTot, SS_PointSelected[:,0:-1], axis=1)
            Qfun_SelectedTot         = np.append(Qfun_SelectedTot, Qfun_Selected[0:-1], axis=0)

        self.Succ_SS_PointSelectedTot = Succ_SS_PointSelectedTot
        self.Succ_uSS_PointSelectedTot = Succ_uSS_PointSelectedTot
        self.SS_PointSelectedTot = SS_PointSelectedTot
        self.Qfun_SelectedTot = Qfun_SelectedTot
        
        self.addSafeSetEqConstr()
        self.addSafeSetCost()

    def addTrajectory(self, x, u, x_glob):
        """update iteration index and construct SS, uSS and Qfun
        Arguments:
            ClosedLoopData: ClosedLoopData object
        """
        self.TimeSS[self.it] = x.shape[0]
        self.LapCounter[self.it] = x.shape[0]
        self.SS[0:(self.TimeSS[self.it]), :, self.it] = x
        self.SS_glob[0:(self.TimeSS[self.it]), :, self.it] = x_glob
        self.uSS[0:self.TimeSS[self.it], :, self.it]      = u
        self.Qfun[0:(self.TimeSS[self.it]), self.it]  = self.computeCost(x,u)

        for i in np.arange(0, self.Qfun.shape[0]):
            if self.Qfun[i, self.it] == 0:
                self.Qfun[i, self.it] = self.Qfun[i - 1, self.it] - 1

        if self.it == 0:
            self.xLin = self.SS[1:self.N + 2, :, self.it]
            self.uLin  = self.uSS[1:self.N + 1, :, self.it]

        self.it = self.it + 1

    def computeCost(self, x, u):
        Cost = 10000 * np.ones((x.shape[0]))  # The cost has the same elements of the vector x --> time +1
        # Now compute the cost moving backwards in a Dynamic Programming (DP) fashion.
        # We start from the last element of the vector x and we sum the running cost
        for i in range(0, x.shape[0]):
            if (i == 0):  # Note that for i = 0 --> pick the latest element of the vector x
                Cost[x.shape[0] - 1 - i] = 0
            elif x[x.shape[0] - 1 - i, 4]< self.predictiveModel.map.TrackLength:
                Cost[x.shape[0] - 1 - i] = Cost[x.shape[0] - 1 - i + 1] + 1
            else:
                Cost[x.shape[0] - 1 - i] = 0

        return Cost

    def addPoint(self, x, u):
        """at iteration j add the current point to SS, uSS and Qfun of the previous iteration
        Arguments:
            x: current state
            u: current input
            i: at the j-th iteration i is the time at which (x,u) are recorded
        """
        Counter = self.TimeSS[self.it - 1]
        self.SS[Counter, :, self.it - 1] = x + np.array([0, 0, 0, 0, self.predictiveModel.map.TrackLength, 0])
        self.uSS[Counter, :, self.it - 1] = u
        
        # The above two lines are needed as the once the predicted trajectory has crossed the finish line the goal is
        # to reach the end of the lap which is about to start
        if self.Qfun[Counter, self.it - 1] == 0:
            self.Qfun[Counter, self.it - 1] = self.Qfun[Counter, self.it - 1] - 1
        self.TimeSS[self.it - 1] = self.TimeSS[self.it - 1] + 1

    def selectPoints(self, it, x0, numPoints):
        x = self.SS[:, 0:(self.TimeSS[it]-1), it]
        u = self.uSS[:, 0:(self.TimeSS[it]-1), it]
        oneVec = np.ones((x.shape[0], 1))
        x0Vec = (np.dot(np.array([x0]).T, oneVec.T)).T
        diff = x - x0Vec
        norm = la.norm(diff, 1, axis=1)
        MinNorm = np.argmin(norm)

        if (MinNorm - numPoints/2 >= 0):
            indexSSandQfun = range(-int(numPoints/2) + MinNorm, int(numPoints/2) + MinNorm + 1)
        else:
            indexSSandQfun = range(MinNorm, MinNorm + int(numPoints))

        SS_Points  = x[indexSSandQfun, :].T
        SSu_Points = u[indexSSandQfun, :].T
        Sel_Qfun = self.Qfun[indexSSandQfun, it]

        # Modify the cost if the predicion has crossed the finisch line
        if self.xPred == []:
            Sel_Qfun = self.Qfun[indexSSandQfun, it]
        elif (np.all((self.xPred[:, 4] > self.predictiveModel.map.TrackLength) == False)):
            Sel_Qfun = self.Qfun[indexSSandQfun, it]
        elif it < self.it - 1:
            Sel_Qfun = self.Qfun[indexSSandQfun, it] + self.Qfun[0, it + 1]
        else:
            sPred = self.xPred[:, 4]
            predCurrLap = self.N - sum(sPred > self.predictiveModel.map.TrackLength)
            currLapTime = self.LapTime
            Sel_Qfun = self.Qfun[indexSSandQfun, it] + currLapTime + predCurrLap

        return SS_Points, SSu_Points, Sel_Qfun
