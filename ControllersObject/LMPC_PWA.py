import pdb
import numpy as np
from cvxopt import spmatrix, matrix, solvers
from numpy import linalg as la
from scipy import linalg
from scipy import sparse
from cvxopt.solvers import qp
import datetime
from functools import partial
# from pathos.multiprocessing import ProcessingPool as Pool
from utilities import Curvature
from numpy import hstack, inf, ones
from scipy.sparse import vstack
from osqp import OSQP
import os

from abc import ABCMeta, abstractmethod
import sys
import pwa_cluster as pwac
import pdb


solvers.options['show_progress'] = False

class AbstractControllerLMPC:
    __metaclass__ = ABCMeta
    """Create the LMPC
    Attributes:
        solve: given x0 computes the control action
        addTrajectory: given a ClosedLoopData object adds the trajectory to SS, Qfun, uSS and updates the iteration index
        addPoint: this function allows to add the closed loop data at iteration j to the SS of iteration (j-1)
        update: this function can be used to set SS, Qfun, uSS and the iteration index.
    """

    def __init__(self, numSS_Points, numSS_it, N, Qslack, Q, R, dR, n, d, shift, dt, track_map, Laps, 
                 TimeLMPC, Solver):
        
        """Initialization
        Arguments:
            numSS_Points: number of points selected from the previous trajectories to build SS
            numSS_it: number of previous trajectories selected to build SS
            N: horizon length
            Q,R: weight to define cost function h(x,u) = ||x||_Q + ||u||_R
            dR: weight to define the input rate cost h(x,u) = ||x_{k+1}-x_k||_dR
            n,d: state and input dimension
            shift: given the closest point x_t^j to x(t) the controller start selecting the point for SS from x_{t+shift}^j
            track_map: track_map
            Laps: maximum number of laps the controller can run (used to avoid dynamic allocation)
            TimeLMPC: maximum time [s] that an lap can last (used to avoid dynamic allocation)
            Solver: solver used in the reformulation of the LMPC as QP
        """
        n_parallel=4
        self.numSS_Points = numSS_Points
        self.numSS_it     = numSS_it
        self.N = N
        self.Qslack = Qslack
        self.Q = Q
        self.R = R
        self.dR = dR
        self.n = n
        self.d = d
        self.shift = int(shift)
        self.dt = dt
        self.track_map = track_map
        self.map = self.track_map
        self.Solver = Solver            
        self.clustering = None # TODO: this is a lazy check, better way
        self.OldInput = np.zeros((1,d))
        startTimer = datetime.datetime.now()
        deltaTimer = datetime.datetime.now() - startTimer
        self.solverTime = deltaTimer
        self.linearizationTime = deltaTimer


        # Initialize the following quantities to avoid dynamic allocation
        # TODO: is there a more graceful way to do this in python?
        # changing things to nan is better, but numerical values
        # seem to be implicitly used in computations
        NumPoints = int(TimeLMPC / dt) + 1
        self.TimeSS  = 10000 * np.ones(Laps).astype(int)        # Time at which each j-th iteration is completed
        self.LapCounter  = -10000 * np.ones(Laps).astype(int)        # Time at which each j-th iteration is completed
        self.SS      = 10000 * np.ones((NumPoints, n, Laps))    # Sampled Safe SS
        self.uSS     = 10000 * np.ones((NumPoints, d, Laps))    # Input associated with the points in SS
        self.Qfun    =     0 * np.ones((NumPoints, Laps))       # Qfun: cost-to-go from each point in SS
        # TODO replace with after-the-fact mapping?
        self.SS_glob = 10000 * np.ones((NumPoints, n, Laps))    # SS in global (X-Y) used for plotting

        # TODO this is only used for PWA controller
        self.SSind = None

        # Initialize the controller iteration
        self.it      = 0

        # Initialize pool for parallel computing used in the internal function _LMPC_EstimateABC
        # self.p = Pool(n_parallel)

    def solve(self, x0, uOld=np.zeros([0, 0])):
        """Computes control action
        Arguments:
            x0: current state position
        """

        # Select Points from Safe Set
        # a subset of nearby points are chosen from past iterations
        self._selectSS(x0)

        # Get the matrices for defining the QP
        qp_params = self._getQP(x0)

        # setting up to loop over QPs
        best_cost = np.inf; best_solution = np.empty((0,0))
        self.feasible = 0; best_ind = 0
        best_status = ''

        startTimer = datetime.datetime.now()
        # TODO make this parallel
        for i, qp_param in enumerate(qp_params):
            startTimer_i = datetime.datetime.now()
            L, G, E, M, q, F, b = qp_param
            # Solve QP
            try:
                if self.Solver == "CVX":
                    res_cons = qp(convert_sparse_cvx(M), matrix(q), convert_sparse_cvx(F), 
                                  matrix(b), convert_sparse_cvx(G), 
                                  convert_sparse_cvx(E) * matrix(x0) + convert_sparse_cvx(L))
                    if res_cons['status'] == 'optimal':
                        feasible = 1
                        cost = res_cons['primal objective']
                    else:
                        feasible = 0
                        cost = np.inf
                    Solution = np.squeeze(res_cons['x'])  
                    status = ''   
                elif self.Solver == "OSQP":
                    # Adaptation for QSQP from https://github.com/alexbuyval/RacingLMPC/
                    res_cons, feasible = osqp_solve_qp(sparse.csr_matrix(M), q, sparse.csr_matrix(F), b, sparse.csr_matrix(G), np.add(np.dot(E,x0),L[:,0]))
                    Solution = res_cons.x
                    cost = res_cons.info.obj_val if feasible else np.inf
                    status = res_cons.info.status_val
                if self.clustering is not None: # TODO: better way to integrate cost
                    cost = cost + self.Qfun_SelectedTot[i]
            except ValueError as e:
                print("caught", e)
                cost = np.inf
                feasible = 0
            deltaTimer_i = datetime.datetime.now() - startTimer_i

            # if np.any(self.SS_PointSelectedTot[4,i]>self.map.TrackLength):
            # #if x0[4] > 15:
            #     print(self.Select_Regs[i])
            #     # print(self.SS_PointSelectedTot[:,i])
            #     pdb.set_trace()

            if cost < best_cost:
                best_cost = cost
                best_solution = Solution
                best_ind = i
                self.feasible = feasible
                best_status = status

        deltaTimer = datetime.datetime.now() - startTimer
        self.solverTime = deltaTimer

        if self.feasible == 0:
            return 

        if best_status != OSQP().constant('OSQP_SOLVED'):
            print('nonoptimal', best_status, x0)
            
        # TODO: incorporate this into selectSS()
        if self.SSind is not None:
            self.SSind += best_ind + 1 

        # Extract solution and set linearization points
        xPred, uPred, lam, slack = LMPC_GetPred(best_solution, self.n, self.d, self.N)
        
        self.xPred = xPred.T
        # TODO more elegant way for various dimensions
        if self.N == 1:
            self.uPred    = np.array([[uPred[0], uPred[1]]])
            self.LinInput =  np.array([[uPred[0], uPred[1]]])
        else:
            self.uPred = uPred.T
            self.LinInput = np.vstack((uPred.T[1:, :], uPred.T[-1, :]))
        self.OldInput = uPred.T[0,:]
        
        # TODO: make this more general
        self.LinPoints = np.vstack((xPred.T[1:,:], xPred.T[-1,:]))
        
        

    def addTrajectory(self, ClosedLoopData):
        """update iteration index and construct SS, uSS and Qfun
        Arguments:
            ClosedLoopData: ClosedLoopData object
        """
        it = self.it

        end_it = ClosedLoopData.SimTime
        self.TimeSS[it] = end_it
        self.LapCounter[it] = ClosedLoopData.SimTime
        self.SS[0:(end_it + 1), :, it] = ClosedLoopData.x[0:(end_it + 1), :]
        self.SS_glob[0:(end_it + 1), :, it] = ClosedLoopData.x_glob[0:(end_it + 1), :]
        self.uSS[0:end_it, :, it]      = ClosedLoopData.u[0:(end_it), :]
        self.Qfun[0:(end_it + 1), it]  = ComputeCost(ClosedLoopData.x[0:(end_it + 1), :],
                                                              ClosedLoopData.u[0:(end_it), :], self.track_map.TrackLength)
        for i in np.arange(0, self.Qfun.shape[0]):
            if self.Qfun[i, it] == 0:
                self.Qfun[i, it] = self.Qfun[i - 1, it] - 1
        if self.it == 0:
            # TODO: made this more general
            self.LinPoints = self.SS[1:self.N + 2, :, it]
            self.LinInput  = self.uSS[1:self.N + 1, :, it]

        self.it = self.it + 1

    def addPoint(self, x, x_glob, u, i):
        """at iteration j add the current point to SS, uSS and Qfun of the previous iteration
        Arguments:
            x: current state
            u: current input
            i: at the j-th iteration i is the time at which (x,u) are recorded
        """
        self.TimeSS[self.it - 1] = self.TimeSS[self.it - 1] + 1
        Counter = self.TimeSS[self.it - 1]
        self.SS[Counter, :, self.it - 1] = x + np.array([0, 0, 0, 0, self.map.TrackLength, 0])
        self.SS_glob[Counter, :, self.it - 1] = x_glob
        self.uSS[Counter, :, self.it - 1] = u
        if self.Qfun[Counter, self.it - 1] == 0:
            self.Qfun[Counter, self.it - 1] = self.Qfun[Counter + i - 1, self.it - 1] - 1        
                
        
    def update(self, SS, uSS, Qfun, TimeSS, it, LinPoints, LinInput):
        """update controller parameters. This function is useful to transfer information among LMPC controller
           with different tuning
        Arguments:
            SS: sampled safe set
            uSS: input associated with the points in SS
            Qfun: Qfun: cost-to-go from each point in SS
            TimeSS: time at which each j-th iteration is completed
            it: current iteration
            LinPoints: points used in the linearization and system identification procedure
            LinInput: inputs associated with the points used in the linearization and system identification procedure
        """
        self.SS  = SS
        self.uSS = uSS
        self.Qfun  = Qfun
        self.TimeSS  = TimeSS
        self.it = it

        self.LinPoints = LinPoints
        self.LinInput  = LinInput

class PWAControllerLMPC(AbstractControllerLMPC):
    """
    Piecewise affine controller

    """

    def __init__(self, n_clusters, numSS_Points, numSS_it, N, Qslack, Q, R, dR, 
                 n, d, shift, dt, track_map, Laps, TimeLMPC, Solver):
        self.n_clusters = n_clusters
        stem = '/home/sarah/barc/workspace/src/barc/src/RacingLMPC/'
        self.load_path = stem + 'notebooks/pwa_model_oval_10.npz'
        self.region_update = False
        self.load_use_data = True
        self.clustering = None
        # python 2/3 compatibility
        super(PWAControllerLMPC, self).__init__(numSS_Points, numSS_it, N, Qslack, Q, R, dR, 
                                              n, d, shift, dt, track_map, Laps, TimeLMPC, Solver)
        self.SS_regions = np.nan * np.ones((self.SS.shape[0],self.SS.shape[2]))
        # np.zeros((self.SS.shape[0],self.SS.shape[2]))


        self.affine = True #  False # False
        dim0 = n+d+1 if self.affine else n+d
        # mask = [A B d].T
        sparse_mask = np.ones([n, dim0])
        sparse_mask[0:3, 3:6] = 0
        sparse_mask[0,6] = 0
        sparse_mask[1:3,7] = 0
        sparse_mask[3:,6:8] = 0
        # sparse_mask[1,0] = 0.0; sparse_mask[1,2] = 0.0
        self.sparse_mask = sparse_mask.T
        
    def addTrajectory(self, ClosedLoopData):
        super(PWAControllerLMPC, self).addTrajectory(ClosedLoopData)
        print(self.it)
        if self.it > 1: 
            it = np.argmin(self.Qfun[0, 1:self.it])+1
            print(it, self.Qfun[0, :self.it])

        # if PWA model initialized, need to update with new data
        # assigning new data to regions
        if self.clustering is not None:
            print("ADDING TRAJECTORY PWA")
            self.SSind = None
            self._estimate_pwa(verbose=False, addTrajectory=True)

    def addPoint(self, x, x_glob, u, i):
        # overriding abstract class
        super(PWAControllerLMPC, self).addPoint(x, x_glob, u, i)

        # update list of SS regions
        Counter = self.TimeSS[self.it - 1]
        self.SS_regions[Counter, self.it - 1] = int(self.clustering.get_region(x 
                                      + np.array([0, 0, 0, 0, self.map.TrackLength, 0])))


    def _selectSS(self, x0):
        
        # choosing safe set to be the fastest one
        # it = self.it -1
        it = np.argmin(self.Qfun[0, 1:self.it])+1

        if True: 
            # TODO: problem behavior with self.N when moving too fast
            self.SSind = max(self.N-self.shift, closest_idx(self.SS[:,:,it], x0))
        self._estimate_pwa(verbose=False)
        select_reg_0 = self.SS_regions[self.SSind-self.N+self.shift:(self.SSind+self.shift+1), it]
        
        # temporary for debugging
        if np.any(np.isnan(select_reg_0)): pdb.set_trace()

        SS_PointSelectedTot      = []
        SS_glob_PointSelectedTot      = []
        Qfun_SelectedTot         = []
        Select_Regs = []
        # TODO should we use numSS_it?
        for i in range(self.numSS_Points):
            term_idx = self.SSind+self.shift+1+i
            terminal_point = self.SS[term_idx,:, it]
            terminal_point_glob = self.SS_glob[term_idx,:, it]
            terminal_cost = self.Qfun[term_idx, it]
            region = self.SS_regions[term_idx, it]

            if terminal_point[0] == 10000: # or region == select_reg_0[-1]:
                select_reg = select_reg_0
            else:
                select_reg = self.SS_regions[(term_idx-self.N):term_idx, it]
            if np.any(np.isnan(select_reg)): 
                # TODO what's happening in this case?
                print(select_reg)
                print(self.SS[(term_idx-self.N):term_idx, it])
                print(terminal_point)
                pdb.set_trace()
                # print("hm why nan")
                select_reg = select_reg_0
            # print(select_reg)
            # remove select_reg[0] = self.SS_regions[self.SSind, it] # TODO is this a hack...
            SS_PointSelectedTot.append(terminal_point)
            SS_glob_PointSelectedTot.append(terminal_point_glob)
            Qfun_SelectedTot.append(terminal_cost)
            Select_Regs.append(select_reg)

        self.SS_PointSelectedTot = np.array(SS_PointSelectedTot).T
        self.SS_glob_PointSelectedTot = np.array(SS_glob_PointSelectedTot).T
        self.Qfun_SelectedTot    = np.array(Qfun_SelectedTot)
        self.Select_Regs = Select_Regs

    def _getQP(self, x0):

        # get dynamics
        As, Bs, ds = pwac.get_PWA_models(self.clustering.thetas, self.n, self.d)
        
        
        qp_mat_list = []
        for i in range(self.numSS_Points):
            terminal_point = self.SS_PointSelectedTot[:,i]
            terminal_cost = self.Qfun_SelectedTot[i]
            select_reg = self.Select_Regs[i]
            
            # TODO when there is no need to recompute matrices
            # L, G, E1, E2, stackedF, stackedb, M, q = full_mat_list[-1]
            # Lterm = L + np.expand_dims(E2.dot(terminal_point),1)
            # qp_mat_list.append((Lterm, G, E1, M, q, stackedF, stackedb))

            # equality constraints from dynamics 
            L, G, E1, E2 = BuildMatEqConst_PWA(As, Bs, ds, self.N, select_reg)
            Lterm = L + np.expand_dims(E2.dot(terminal_point),1)
            # inequality constraints from regions
            F_region, b_region = self.clustering.get_region_matrices()
            stackedF, stackedb = LMPC_BuildMatIneqConst(self.N, F_region=F_region, 
                                                            b_region=b_region, SelectReg=select_reg)
            # stackedF, stackedb = LMPC_BuildMatIneqConst(self.N)
            M, q = LMPC_BuildMatCost(self.N, self.Qslack, self.Q, self.R, self.dR, self.OldInput)

            qp_mat_list.append((Lterm, G, E1, M, q, stackedF, stackedb))
        return qp_mat_list

    def _estimate_pwa(self, verbose=True, addTrajectory=False):
        startTimer = datetime.datetime.now()
        # when the clustering object is not initialized
        if self.clustering is None:
            # recording all SS data
            zs = []; ys = []
            for it in range(self.it):
                states = self.SS[:int(self.LapCounter[it]+1), :, it]
                inputs = self.uSS[:int(self.LapCounter[it]+1), :, it]
                zs.append(np.hstack([states[:-1,:], inputs[:-1,:]]))
                ys.append(states[1:,:])
            zs = np.squeeze(np.vstack(zs)); ys = np.squeeze(np.vstack(ys))

            # create clustering object with possibly previous data
            if self.load_path is not None:
                # loading previous data and model
                data = np.load(self.load_path)
                affine=self.affine; sparse_mask=self.sparse_mask 
                if self.load_use_data:
                    self.clustering = pwac.ClusterPWA.from_labels(data['zs'], data['ys'], 
                                   data['labels'], z_cutoff=self.n, affine=affine, sparse_mask=sparse_mask)
                    cluster_ind = len(self.clustering.cluster_labels)
                    self.clustering.region_fns = data['region_fns']
                    # adding new SS data
                    self.clustering.add_data_update(zs, ys, verbose=verbose, full_update=self.region_update)
                else:
                    # labelling new SS data
                    cluster_labels = []
                    for z,y in zip(zs,ys):
                        dot_pdt = [w.T.dot(np.hstack([z[0:self.n], [1]])) for w in data['region_fns']]
                        idx = np.argmax(dot_pdt)
                        cluster_labels.append(idx)
                    cluster_labels = np.array(cluster_labels)
                    self.clustering = pwac.ClusterPWA.from_labels(zs, ys, 
                                   cluster_labels, z_cutoff=self.n, affine=self.affine, sparse_mask=self.sparse_mask)
                    self.clustering.region_fns = data['region_fns']

                if self.region_update: self.clustering.determine_polytopic_regions(verbose=verbose)
                
            else:
                # use greedy method to fit PWA model
                self.clustering = pwac.ClusterPWA.from_num_clusters(zs, ys, 
                                    self.n_clusters, z_cutoff=self.n,
                                    affine=self.affine, sparse_mask=self.sparse_mask)
                self.clustering.fit_clusters(verbose=verbose)
                self.clustering.determine_polytopic_regions(verbose=verbose)
                
            # label the regions of the points in the safe set
            for it in range(self.it):
                for i in range(self.LapCounter[it]+1):
                    self.SS_regions[i, it] = self.clustering.get_region(self.SS[i,:,it])
            if verbose: print_PWA_models(pwac.get_PWA_models(self.clustering.thetas, self.n, self.d))
        # if cluster already initialized, adding new lap data
        elif addTrajectory:
            print('updating PWA model with new data')
            zs = []; ys = []
            it = self.it-1
            states = self.SS[:int(self.LapCounter[it]+1), :, it]
            inputs = self.uSS[:int(self.LapCounter[it]+1), :, it]
            zs.append(np.hstack([states[:-1], inputs[:-1]]))
            ys.append(states[1:])
            zs = np.squeeze(np.array(zs)); ys = np.squeeze(np.array(ys))
            
            self.clustering.add_data_update(zs, ys, verbose=verbose, full_update=self.region_update)
            # TODO this method takes a long time to run with full_update
            if self.region_update: self.clustering.determine_polytopic_regions(verbose=verbose)
            if verbose: print_PWA_models(pwac.get_PWA_models(self.clustering.thetas, self.n, self.d))

            # label the regions of the points in the safe set
            # TODO
            for i in range(self.LapCounter[it]+1):
                self.SS_regions[i, it] = self.clustering.get_region(self.SS[i,:,it])

            np.savez('cluster_labels'+str(self.it), labels=self.clustering.cluster_labels,
                                       region_fns=self.clustering.region_fns,
                                       thetas=self.clustering.thetas)
        # print('count in each region', np.bincount(self.clustering.cluster_labels.astype(int)))
        deltaTimer = datetime.datetime.now() - startTimer
        self.linearizationTime = deltaTimer # TODO generalize

    def oneStepPrediction(self, x, u, UpdateModel=0):
        """Propagate the model one step foreward
        Arguments:
            x: current state
            u: current input
            UpdateModel: (unused)
        """
        startTimer = datetime.datetime.now()

        if self.clustering is None:
            x_next = x
        else:
            x_next = self.clustering.get_prediction(np.hstack([x, u]))

        endTimer = datetime.datetime.now(); deltaTimer = endTimer - startTimer

        return x_next, deltaTimer


class ControllerLMPC(AbstractControllerLMPC):
    """Create the LMPC
    Attributes:
        solve: given x0 computes the control action
        addTrajectory: given a ClosedLoopData object adds the trajectory to SS, Qfun, uSS and updates the iteration index
        addPoint: this function allows to add the closed loop data at iteration j to the SS of iteration (j-1)
        update: this function can be used to set SS, Qfun, uSS and the iteration index.
    """
    def __init__(self, numSS_Points, numSS_it, N, Qslack, Q, R, dR, 
                 n, d, shift, dt, track_map, Laps, TimeLMPC, Solver):
        # Build matrices for inequality constraints
        self.stackedF, self.stackedb = LMPC_BuildMatIneqConst(N, numSS_Points)
       
        super(ControllerLMPC, self).__init__(numSS_Points, numSS_it, N, Qslack, Q, R, dR, 
                                              n, d, shift, dt, track_map, Laps, TimeLMPC, Solver)
    
    def _selectSS(self, x0):
        SS_PointSelectedTot      = np.empty((self.n, 0))
        Qfun_SelectedTot         = np.empty((0))
        for jj in range(0, self.numSS_it):
            SS_PointSelected, Qfun_Selected = SelectPoints(self.SS, self.Qfun, self.it - jj - 1, x0, self.numSS_Points / self.numSS_it, self.shift)
            SS_PointSelectedTot =  np.append(SS_PointSelectedTot, SS_PointSelected, axis=1)
            Qfun_SelectedTot    =  np.append(Qfun_SelectedTot, Qfun_Selected, axis=0)

        self.SS_PointSelectedTot = SS_PointSelectedTot
        self.Qfun_SelectedTot    = Qfun_SelectedTot

    def _getQP(self, x0):
        # Run System ID
        startTimer = datetime.datetime.now()
        Atv, Btv, Ctv, _ = self._EstimateABC()
        deltaTimer = datetime.datetime.now() - startTimer
        L, G, E = BuildMatEqConst_TV(Atv, Btv, Ctv, self.SS_PointSelectedTot)
        self.linearizationTime = deltaTimer

        M, q = LMPC_BuildMatCost(self.N, self.Qslack, self.Q, self.R, self.dR, self.OldInput,
                                 self.Qfun_SelectedTot, self.numSS_Points)
        # confirming cost matrix correctness
        # M_old, q_old = LMPC_BuildMatCost_old(self.N, self.Qfun_SelectedTot, self.numSS_Points, 
        #                              self.Qslack, self.Q, self.R, self.dR, self.OldInput)

        # print(np.linalg.norm(M-M_old))
        # print(np.linalg.norm(q-q_old))

        return [(L, G, E, M, q, self.stackedF, self.stackedb)]

    def _EstimateABC(self):
        LinPoints       = self.LinPoints
        LinInput        = self.LinInput
        N               = self.N
        n               = self.n
        d               = self.d
        SS              = self.SS
        uSS             = self.uSS
        TimeSS          = self.TimeSS
        PointAndTangent = self.track_map.PointAndTangent
        dt              = self.dt
        it              = self.it
        p               = self.p

        ParallelComputation = 0 # TODO
        Atv = []; Btv = []; Ctv = []; indexUsed_list = []

        usedIt = range(it-2,it)
        MaxNumPoint = 40  # TODO Need to reason on how these points are selected

        if ParallelComputation == 1:
            # Parallel Implementation
            Fun = partial(RegressionAndLinearization, LinPoints, LinInput, usedIt, SS, uSS, TimeSS,
                           MaxNumPoint, qp, n, d, matrix, PointAndTangent, dt)

            index = np.arange(0, N)  # Create the index vector

            Res = p.map(Fun, index)  # Run the process in parallel
            ParallelResutl = np.asarray(Res)

        for i in range(0, N):
            if ParallelComputation == 0:
               Ai, Bi, Ci, indexSelected = RegressionAndLinearization(LinPoints, LinInput, usedIt, SS, uSS, TimeSS,
                                                                   MaxNumPoint, qp, n, d, matrix, PointAndTangent, dt, i)
               Atv.append(Ai); Btv.append(Bi); Ctv.append(Ci)
               indexUsed_list.append(indexSelected)
            else:
               Atv.append(ParallelResutl[i][0])
               Btv.append(ParallelResutl[i][1])
               Ctv.append(ParallelResutl[i][2])
               indexUsed_list.append(ParallelResutl[i][3])

        return Atv, Btv, Ctv, indexUsed_list

# ======================================================================================================================
# ======================================================================================================================
# =============================== Utility functions for LMPC Safe Set ==================================================
# ======================================================================================================================
# ======================================================================================================================

def SelectPoints(SS, Qfun, it, x0, numSS_Points, shift):
    # selects the closest point in the safe set to x0
    # returns a subset of the safe set which contains a range of points ahead of this point
    x = SS[:, :, it]
    MinNorm = closest_idx(x, x0)

    if (MinNorm + shift >= 0):
        # TODO: what if shift + MinNorm + numSS_Points is greater than the points in the safe set?
        SS_Points = x[int(shift + MinNorm):int(shift + MinNorm + numSS_Points), :].T
        Sel_Qfun = Qfun[int(shift + MinNorm):int(shift + MinNorm + numSS_Points), it]
    else:
        SS_Points = x[int(MinNorm):int(MinNorm + numSS_Points), :].T
        Sel_Qfun = Qfun[int(MinNorm):int(MinNorm + numSS_Points), it]

    return SS_Points, Sel_Qfun

def closest_idx(x, x0, verbose=False):
    oneVec = np.ones((x.shape[0], 1))
    x0Vec = (np.dot(np.array([x0]).T, oneVec.T)).T
    diff = x - x0Vec
    norm = la.norm(diff, 1, axis=1)
    if verbose: print(norm)
    return np.argmin(norm)

def ComputeCost(x, u, TrackLength):
    Cost = 10000 * np.ones((x.shape[0]))  # The cost has the same elements of the vector x --> time +1
    # Now compute the cost moving backwards in a Dynamic Programming (DP) fashion.
    # We start from the last element of the vector x and we sum the running cost
    # TODO why this form of cost?
    for i in range(0, x.shape[0]):
        if (i == 0):  # Note that for i = 0 --> pick the latest element of the vector x
            Cost[x.shape[0] - 1 - i] = 0
        elif x[x.shape[0] - 1 - i, 4]< TrackLength:
            Cost[x.shape[0] - 1 - i] = Cost[x.shape[0] - 1 - i + 1] + 1
        else:
            Cost[x.shape[0] - 1 - i] = 0

    return Cost



# ======================================================================================================================
# ======================================================================================================================
# =============================== Utility functions for LMPC reformulation to QP =======================================
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
    # osqp.setup()
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
        osqp.setup(P=P, q=q, A=qp_A, l=qp_l, u=qp_u, verbose=False, polish=True, time_limit=0.007)
    else:
        osqp.setup(P=P, q=q, A=None, l=None, u=None, verbose=False, time_limit=0.007)
    if initvals is not None:
        osqp.warm_start(x=initvals)
    res = osqp.solve()
    if res.info.status_val != osqp.constant('OSQP_SOLVED'):
        # print("OSQP exited with status '%s'" % res.info.status)
        feasible = 0
    if res.info.status_val == osqp.constant('OSQP_SOLVED') or res.info.status_val == osqp.constant('OSQP_SOLVED_INACCURATE') or  res.info.status_val == osqp.constant('OSQP_MAX_ITER_REACHED'):
        feasible = 1
    return res, feasible

def convert_sparse_cvx(A):
    return spmatrix(A[np.nonzero(A)], np.nonzero(A)[0].astype(int), np.nonzero(A)[1].astype(int), A.shape)

def LMPC_GetPred(Solution,n,d,N):
    # logic to decompose the QP solution
    xPred = np.squeeze(np.transpose(np.reshape((Solution[np.arange(n*(N+1))]),(N+1,n))))
    uPred = np.squeeze(np.transpose(np.reshape((Solution[n*(N+1)+np.arange(d*N)]),(N, d))))
    if Solution.shape[0] > n*(N+1)+d*N+n: # extra variables are SS lambdas
        lambd = Solution[n*(N+1)+d*N:Solution.shape[0]-n]
    else:
        lambd = None
    slack = Solution[Solution.shape[0]-n:]
    return xPred, uPred, lambd, slack

def LMPC_BuildMatCost(N, Qslack, Q, R, dR, uOld, Sel_Qfun=[], numSS_Points=0):
    '''
    Builds costs matrices M,q for the cost
    1/2 z^T M z + q^T M

    Sel_Qfun, numSS_Points are optional arguments to be included
    if the QP includes the convex hull of SS points.
    '''
    # TODO, additional cost for LMPC? affine term.
    n = Q.shape[0]
    d = R.shape[0]
    vt = 2
    Qlane = 0.1 * 0.5 * 10 * np.array([50, 10])
    quadLaneSlack = Qlane[0] * np.eye(2*N)
    linLaneSlack = Qlane[1] * np.ones(2*N)

    Mx, P, Mu = MPC_MatCost(N, Q, R, dR)
    M = 2 * linalg.block_diag(Mx, P, Mu, 
                   np.zeros((numSS_Points, numSS_Points)), Qslack, quadLaneSlack)
    
    xtrack = np.array([vt, 0, 0, 0, 0, 0])
    q00 = - 2 * np.dot(np.tile(xtrack, N + 1), linalg.block_diag(Mx, P) )
    q0 = np.append(q00, np.zeros(d * N))
    # Derivative Input
    q0[n*(N+1):n*(N+1)+2] = -2 * np.dot( uOld, np.diag(dR) )
    q = np.append(np.append(np.append(q0, Sel_Qfun), np.zeros(n)), linLaneSlack)
    return M, q

def LMPC_BuildMatCost_old(N, Sel_Qfun, numSS_Points, Qslack, Q, R, dR, uOld):
    n = Q.shape[0]
    P = Q
    vt = 2

    b = [Q] * (N)
    Mx = linalg.block_diag(*b)

    c = [R + 2*np.diag(dR)] * (N)

    Mu = linalg.block_diag(*c)
    # Need to condider that the last input appears just onece in the difference
    Mu[Mu.shape[0] - 1, Mu.shape[1] - 1] = Mu[Mu.shape[0] - 1, Mu.shape[1] - 1] - dR[1]
    Mu[Mu.shape[0] - 2, Mu.shape[1] - 2] = Mu[Mu.shape[0] - 2, Mu.shape[1] - 2] - dR[0]

    # Derivative Input Cost
    OffDiaf = -np.tile(dR, N-1)
    np.fill_diagonal(Mu[2:], OffDiaf)
    np.fill_diagonal(Mu[:, 2:], OffDiaf)
    # np.savetxt('Mu.csv', Mu, delimiter=',', fmt='%f')

    # Add lane slack variable Cost
    Qlane = 0.1 * 0.5 * 10 * np.array([50, 10])
    quadLaneSlack = Qlane[0] * np.eye(2*LMPC.N)

    M00 = linalg.block_diag(Mx, P, Mu)
    M0 = linalg.block_diag(M00, np.zeros((numSS_Points, numSS_Points)), Qslack, quadLaneSlack)
    xtrack = np.array([vt, 0, 0, 0, 0, 0])
    q0 = - 2 * np.dot(np.append(np.tile(xtrack, N + 1), np.zeros(R.shape[0] * N)), M00)

    # Derivative Input
    q0[n*(N+1):n*(N+1)+2] = -2 * np.dot( uOld, np.diag(dR) )


    # Add lane slack variable Cost
    linLaneSlack = Qlane[1] * np.ones(2*LMPC.N)
    # np.savetxt('q0.csv', q0, delimiter=',', fmt='%f')
    q = np.append(np.append(q0, Sel_Qfun), np.zeros(Q.shape[0]), linLaneSlack)

    # np.savetxt('q.csv', q, delimiter=',', fmt='%f')

    M = 2 * M0  # Need to multiply by two because CVX considers 1/2 in front of quadratic cost

    return M, q

def MPC_MatCost(N, Q, R, dR):
    '''
    builds repeated cost matrices for finite horizon MPC
    TODO allow for time-varying Q,R
    '''
    n = Q.shape[0]
    P = Q # TODO this should be final cost related to xf? Sel_Qfun

    b = [Q] * (N)
    Mx = linalg.block_diag(*b)

    c = [R + 2*np.diag(dR)] * (N)

    Mu = linalg.block_diag(*c)
    # Need to consider that the last input appears just once in the difference
    Mu[Mu.shape[0] - 1, Mu.shape[1] - 1] = Mu[Mu.shape[0] - 1, Mu.shape[1] - 1] - dR[1]
    Mu[Mu.shape[0] - 2, Mu.shape[1] - 2] = Mu[Mu.shape[0] - 2, Mu.shape[1] - 2] - dR[0]

    # Derivative Input Cost
    OffDiag = -np.tile(dR, N-1)
    np.fill_diagonal(Mu[2:], OffDiag)
    np.fill_diagonal(Mu[:, 2:], OffDiag)
    return Mx, P, Mu

def LMPC_BuildMatIneqConst(N, numSS_Points=0, SelectReg=None, 
                           F_region=[], b_region=[]):
    '''
    Builds constraints F,b for 
    F z <= b

    numSS_Points is optional argument to be included
    if the QP includes the convex hull of SS points.

    SelectReg, F_region, b_region are optional 
    arguments if QP is for PWA
    '''
    Fx, bx, Fu, bu =  getRacingFb() 
    n = Fx.shape[1] 

    # Now stuck the constraint matrices to express them in the form Fz<=b. Note that z collects states and inputs
    # Let's start by computing the submatrix of F relates with the state
    if SelectReg is None:
        rep_a = [Fx] * (N)
        MatFx = linalg.block_diag(*rep_a)
        bxtot = np.tile(np.squeeze(bx), N)
    else: 
        MatFx = np.empty((0, 0))
        bxtot  = np.empty(0)
        for i in range(1, N): # No need to constraint (initial or terminal point --> go 1 up to N
            Fxreg = np.vstack([Fx, F_region[int(SelectReg[i])]])
            bxreg = np.vstack([bx, np.expand_dims(b_region[int(SelectReg[i])], 1)])
            MatFx = linalg.block_diag(MatFx, Fxreg)
            bxtot  = np.append(bxtot, bxreg)

    NoTerminalConstr = np.zeros((np.shape(MatFx)[0], n))  # No need to constraint initial/terminal point
    Fxtot = np.hstack((NoTerminalConstr, MatFx, NoTerminalConstr))

    # Let's start by computing the submatrix of F relates with the input
    rep_b = [Fu] * (N)
    Futot = linalg.block_diag(*rep_b)
    butot = np.tile(np.squeeze(bu), N)

    # Let's stack all together
    rFxtot, cFxtot = np.shape(Fxtot)
    rFutot, cFutot = np.shape(Futot)
    Dummy1 = np.hstack((Fxtot, np.zeros((rFxtot, cFutot))))
    Dummy2 = np.hstack((np.zeros((rFutot, cFxtot)), Futot))
    FDummy = np.vstack((Dummy1, Dummy2))

    I = -np.eye(numSS_Points)
    FDummy2 = linalg.block_diag(FDummy, I)
    
    Fslack = np.zeros((FDummy2.shape[0], n))
    F_hard = np.hstack((FDummy2, Fslack))
    b_hard = np.hstack((bxtot, butot, np.zeros(numSS_Points)))

    # Add slack Variables on Lane Boundaries
    LaneSlack = np.zeros((F_hard.shape[0], 2*N))
    colIndexPositive = []
    rowIndexPositive = []
    colIndexNegative = []
    rowIndexNegative = []
    for i in range(0, N):
        colIndexPositive.append( i*2 + 0 )
        colIndexNegative.append( i*2 + 1 )

        rowIndexPositive.append(i*Fx.shape[0] + 0) # Slack on first element of Fx
        rowIndexNegative.append(i*Fx.shape[0] + 1) # Slack on second element of Fx
    
    LaneSlack[rowIndexPositive, colIndexPositive] = -1.0
    LaneSlack[rowIndexNegative, rowIndexNegative] = -1.0

    F_1 = np.hstack((F_hard, LaneSlack))

    I = - np.eye(2*N)
    Zeros = np.zeros((2*N, F_hard.shape[1]))
    Positivity = np.hstack((Zeros, I))

    F = np.vstack((F_1, Positivity))

    # np.savetxt('F.csv', F, delimiter=',', fmt='%f')
    # pdb.set_trace()

    # b vector
    b = np.hstack((b_hard, np.zeros(2*N)))

    return F, b


def getRacingFb():
    # Build the matrices for the state constraint in each region. In the region i we want Fx[i]x <= bx[b]
    Fx = np.array([[0., 0., 0., 0., 0., 1.],
                   [0., 0., 0., 0., 0., -1.]])

    bx = np.array([[0.8],  # max ey
                   [0.8]])  # max ey

    # Build the matrices for the input constraint in each region. In the region i we want Fx[i]x <= bx[b]
    Fu = np.array([[1., 0.],
                   [-1., 0.],
                   [0., 1.],
                   [0., -1.]])

    bu = np.array([[0.25],  # Max Steering
                   [0.25],  # Max Steering
                   [2.],  # Max Acceleration
                   [0.7]])  # Max Acceleration
    return Fx, bx, Fu, bu


def BuildMatEqConst_TV(A, B, C, SS_Points):
    '''
    Builds equality constraints L, G, E for 
    G z = E * x0 + L
    '''
    N = len(A)
    n, d = B[0].shape

    Afun = lambda i: A[i]
    Bfun = lambda i: B[i]
    Cfun = lambda i: C[i]

    Gx, Gu, L1, E = dynamicsEqConstr(Afun, Bfun, Cfun, N)

    G = np.hstack((Gx, Gu))

    # for terminal constraint lambda
    L = np.vstack((L1, [1]))

    # extra row to constraint the terminal point to be equal to a point in SS
    TermCons = np.zeros((n, (N + 1) * n + N * d))
    TermCons[:, N * n:(N + 1) * n] = np.eye(n)
    G_enlarged = np.vstack((G, TermCons))

    G_lambda = np.zeros(( G_enlarged.shape[0], SS_Points.shape[1] + n))
    G_lambda[G_enlarged.shape[0] - n:G_enlarged.shape[0], :] = np.hstack((-SS_Points, np.eye(n)))
    G_LMPC0 = np.hstack((G_enlarged, G_lambda))
    G_ConHull = np.zeros((1, G_LMPC0.shape[1]))
    G_ConHull[-1, G_ConHull.shape[1]-SS_Points.shape[1]-n:G_ConHull.shape[1]-n] = np.ones((1,SS_Points.shape[1]))

    G_LMPC = np.vstack((G_LMPC0, G_ConHull))

    E_LMPC = np.vstack((E, np.zeros((n + 1, n))))

    return L, G_LMPC, E_LMPC

def BuildMatEqConst_PWA(As, Bs, ds, N, SelectedRegions):
    '''
    Builds equality constraints L, G, E for 
    G z = E * x0 + L
    '''
    n, d = Bs[0].shape
    Afun = lambda i: As[int(SelectedRegions[i])]
    Bfun = lambda i: Bs[int(SelectedRegions[i])]
    Cfun = lambda i: ds[int(SelectedRegions[i])].reshape(n,1)

    Gx, Gu, L, E = dynamicsEqConstr(Afun, Bfun, Cfun, N)

    
    G0 = np.hstack((Gx, Gu, np.zeros([n*(N+1),n]))) # zeros for slack
    Gterm = np.zeros([n, G0.shape[1]])  # constraint on x_{N+1}
    Gterm[:,n*N:n*(N+1)] = np.eye(n) # constraint on x_{N+1}
    Gterm[:,-n:] = np.eye(n) # slack on x_{N+1}
    G_hard = np.vstack([G0,Gterm])

    # Add columns related with lane slack
    SlackLane = np.zeros((G_hard.shape[0], 2*N))
    G = np.hstack((G_hard, SlackLane))


    E1 = np.zeros((n * (N + 1) + n, n))  # + n for terminal constraint
    E1[np.arange(n)] = np.eye(n) # x0 constraint

    E2 = np.zeros((n * (N + 1) + n, n))  # + n for terminal constraint
    E2[-n:] = np.eye(n) # terminal point constraint


    return L, G, E1, E2

def dynamicsEqConstr(Afun, Bfun, Cfun, N):
    '''
    Returns equality constraints of the form
    diagstack(Gx, Gu) [x;u] = E * x0 + L
    for possibly time-varying dynamics returned
    by indexing functions Afun, Bfun, Cfun
    '''
    n, d = Bfun(0).shape

    Gx = np.eye(n * (N + 1))
    Gu = np.zeros((n * (N + 1), d * (N)))

    L = np.zeros((n * (N + 1) + n, 1)) # + n for terminal constraint

    for i in range(0, N):
        ind1 = n + i * n + np.arange(n)
        ind2x = i * n + np.arange(n)
        ind2u = i * d + np.arange(d)

        Gx[np.ix_(ind1, ind2x)] = -Afun(i)
        Gu[np.ix_(ind1, ind2u)] = -Bfun(i)
        L[ind1, :]   =  Cfun(i)

    E = np.zeros((n * (N + 1), n))
    E[np.arange(n)] = np.eye(n)

    return Gx, Gu, L, E


# ======================================================================================================================
# ======================================================================================================================
# ========================= Utility functions for Local Regression and Linearization ===================================
# ======================================================================================================================
# ======================================================================================================================

# TODO: this function is in progress
def PWA_model_from_LTV(self):
    LinPoints       = self.LinPoints
    LinInput        = self.LinInput
    N               = self.N
    n               = self.n
    d               = self.d
    SS              = self.SS
    uSS             = self.uSS
    TimeSS          = self.TimeSS
    PointAndTangent = self.track_map.PointAndTangent
    dt              = self.dt
    it              = self.it
    p               = self.p

    ParallelComputation = 0 # TODO
    Atv = []; Btv = []; Ctv = []; indexUsed_list = []

    usedIt = range(it-2,it)
    MaxNumPoint = 40  # TODO Need to reason on how these points are selected


    for i in range(0, N):
       Ai, Bi, Ci, indexSelected = RegressionAndLinearization(LinPoints, LinInput, usedIt, SS, uSS, TimeSS,
                                                           MaxNumPoint, qp, n, d, matrix, PointAndTangent, dt, i)
       Atv.append(Ai); Btv.append(Bi); Ctv.append(Ci)
       indexUsed_list.append(indexSelected)

    return Atv, Btv, Ctv, indexUsed_list

def RegressionAndLinearization(LinPoints, LinInput, usedIt, SS, uSS, TimeSS, MaxNumPoint, qp, n, d, matrix, PointAndTangent, dt, i):
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
    for i in usedIt:
        indexSelected_i, K_i = ComputeIndex(h, SS, uSS, TimeSS, i, xLin, stateFeatures, scaling, MaxNumPoint,
                                            ConsiderInput)
        indexSelected.append(indexSelected_i)
        K.append(K_i)

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
    Ci[3] = epsi + dt * ( wz - (vx * np.cos(epsi) - vy * np.sin(epsi)) / (1 - cur * ey) * cur ) - np.dot(Ai[3, :], x0)

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
    Ci[4] = s    + dt * ( (vx * np.cos(epsi) - vy * np.sin(epsi)) / (1 - cur * ey) ) - np.dot(Ai[4, :], x0)

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

    deltaTimer_tv = datetime.datetime.now() - startTimer

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

    # K = np.ones(len(index))

    startTimer = datetime.datetime.now()  # Start timer for LMPC iteration
    res_cons = qp(Q, b) # This is ordered as [A B C]

    deltaTimer_tv = datetime.datetime.now() - startTimer

    # print("Non removable time: ", deltaTimer_tv.total_seconds())
    Result = np.squeeze(np.array(res_cons['x']))
    A = Result[0:len(stateFeatures)]
    B = Result[len(stateFeatures):(len(stateFeatures)+len(inputFeatures))]
    C = Result[-1]

    return A, B, C

def ComputeIndex(h, SS, uSS, TimeSS, it, x0, stateFeatures, scaling, MaxNumPoint, ConsiderInput):

    # 40 smallest k \| [x(t); u(t)] - [x_k; u_k] \|

    startTimer = datetime.datetime.now()  # Start timer for LMPC iteration

    # What to learn a model such that: x_{k+1} = A x_k  + B u_k + C
    oneVec = np.ones( (SS[0:TimeSS[it], :, it].shape[0]-1, 1) )

    x0Vec = (np.dot( np.array([x0]).T, oneVec.T )).T

    if ConsiderInput == 1:
        DataMatrix = np.hstack((SS[0:TimeSS[it]-1, stateFeatures, it], uSS[0:TimeSS[it]-1, :, it]))
    else:
        DataMatrix = SS[0:TimeSS[it]-1, stateFeatures, it]

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
    # K = np.ones(len(index))

    return index, K

def print_PWA_models(models):
    As, Bs, ds = models
    for A,B,d in zip(As, Bs, ds):
        spacer = np.nan * np.ones([A.shape[0], 1])
        stacked = np.hstack([A, spacer, B, spacer, d[:,np.newaxis]])
        print(np.array_str(stacked, precision=2, suppress_small=True))