def LMPC(npG, L, npE, F, b, x0, np, qp, matrix, datetime, la, SS, Qfun, N, n, d, spmatrix, numSS_Points, Qslack, Q_LMPC, R_LMPC, it):

    SS_Points1, Sel_Qfun1 = SelectPoints(SS, Qfun, it-1, x0, numSS_Points/2, np, la)
    SS_Points2, Sel_Qfun2 = SelectPoints(SS, Qfun, it-2, x0, numSS_Points/2, np, la)

    SS_Points = np.hstack((SS_Points1, SS_Points2))

    Sel_Qfun = np.hstack((Sel_Qfun1, Sel_Qfun2))


    G, E = LMPC_TermConstr(npG, npE, N, n, d, np, spmatrix, SS_Points)

    M, q = LMPC_BuildMatCost(Sel_Qfun, numSS_Points, N, np, spmatrix, Qslack, Q_LMPC, R_LMPC)

    startTimer = datetime.datetime.now()
    Sol, feasible = LMPC_FTOCP(M, q, G, L, E, F, b, x0, np, qp, matrix)
    xPred, uPred, lambdPred, slack = LMPC_GetPred(Sol, n,d,N, np)

    # print SS_Points.shape, lambdPred.shape
    # print "Term Costr \n", np.dot(SS_Points, lambdPred),"\n", xPred[:,-1], "\n", xPred.T
    # print "Here lambda ", lambdPred, lambdPred.shape, np.dot(np.ones(lambdPred.shape[0]), lambdPred)
    # print "Check Slack", slack, np.dot(SS_Points, lambdPred)-xPred[:,-1]

    # name = raw_input("Please enter name: ")


    endTimer = datetime.datetime.now();
    deltaTimer = endTimer - startTimer

    return Sol, feasible, deltaTimer, slack

def SelectPoints(SS, Qfun, it, x0, numSS_Points, np, la):
    x = SS[:, :, it]

    oneVec = np.ones((x.shape[0], 1))
    x0Vec = (np.dot(np.array([x0]).T, oneVec.T)).T
    diff = x - x0Vec
    norm = la.norm(diff, 1, axis=1)
    MinNorm = np.argmin(norm)

    SS_Points = x[MinNorm:MinNorm + numSS_Points, :].T
    Sel_Qfun = Qfun[MinNorm:MinNorm + numSS_Points, it]
    return SS_Points, Sel_Qfun

def ComputeCost(x, u, np, TrackLength):
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


def LMPC_TermConstr(G, E, N ,n ,d ,np, spmatrix, SS_Points):
    # Update the matrices for the Equality constraint in the LMPC. Now we need an extra row to constraint the terminal point to be equal to a point in SS
    # The equality constraint has now the form: G_LMPC*z = E_LMPC*x0 + TermPoint.
    # Note that the vector TermPoint is updated to constraint the predicted trajectory into a point in SS. This is done in the FTOCP_LMPC function

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

    # np.savetxt('G.csv', G_LMPC, delimiter=',', fmt='%f')
    # np.savetxt('E.csv', E_LMPC, delimiter=',', fmt='%f')

    G_LMPC_sparse = spmatrix(G_LMPC[np.nonzero(G_LMPC)], np.nonzero(G_LMPC)[0], np.nonzero(G_LMPC)[1], G_LMPC.shape)
    E_LMPC_sparse = spmatrix(E_LMPC[np.nonzero(E_LMPC)], np.nonzero(E_LMPC)[0], np.nonzero(E_LMPC)[1], E_LMPC.shape)

    return G_LMPC_sparse, E_LMPC_sparse

def LMPC_BuildMatCost(Sel_Qfun, numSS_Points, N, np, spmatrix, Qslack, Q, R):
    from scipy import linalg

    P = Q
    vt = 2


    b = [Q] * (N)
    Mx = linalg.block_diag(*b)

    c = [R] * (N)
    Mu = linalg.block_diag(*c)

    M00 = linalg.block_diag(Mx, P, Mu)
    M0  = linalg.block_diag(M00, np.zeros((numSS_Points, numSS_Points)), Qslack)
    xtrack = np.array([vt, 0, 0, 0, 0, 0])
    q0 = - 2 * np.dot(np.append(np.tile(xtrack, N+1), np.zeros(R.shape[0]*N)), M00)
    # print q0.shape, Sel_Qfun.shape, Q.shape[0], np.zeros(Q.shape[0]).shape
    q  = np.append(np.append(q0, Sel_Qfun), np.zeros(Q.shape[0]))

    # np.savetxt('q.csv', q, delimiter=',', fmt='%f')

    M = 2 * M0  # Need to multiply by two because CVX considers 1/2 in front of quadratic cost

    M_sparse = spmatrix(M[np.nonzero(M)], np.nonzero(M)[0], np.nonzero(M)[1], M.shape)
    return M_sparse, q

def LMPC_FTOCP(M, q, G, L, E, F, b, x0, np, qp, matrix):
    from numpy.linalg import matrix_rank
    G_cvx = F
    A_cvx = G

    # print A_cvx.size, G_cvx.size
    # print M.size,  matrix(q).size, F.size, matrix(b).size, G.size, E.size, L.size

    res_cons = qp(M, matrix(q), F, matrix(b), G, E * matrix(x0) + L)
    if res_cons['status'] == 'optimal':
        feasible = 1
    else:
        feasible = 0

    return np.squeeze(res_cons['x']), feasible


def LMPC_BuildMatEqConst(A, B, C, N, n, d, np, spmatrix, TimeVarying):
    # Buil matrices for optimization (Convention from Chapter 15.2 Borrelli, Bemporad and Morari MPC book)
    # We are going to build our optimization vector z \in \mathbb{R}^((N+1) \dot n \dot N \dot d), note that this vector
    # stucks the predicted trajectory x_{k|t} \forall k = t, \ldots, t+N+1 over the horizon and
    # the predicte input u_{k|t} \forall k = t, \ldots, t+N over the horizon
    Gx = np.eye(n * (N + 1))
    Gu = np.zeros((n * (N + 1), d * (N)))

    E = np.zeros((n * (N + 1), n))
    E[np.arange(n)] = np.eye(n)

    L = np.zeros((n * (N + 1) + n + 1, 1)) # n+1 for the terminal constraint
    L[-1] = 1 # Summmation of lamba must add up to 1

    for i in range(0, N):
        ind1 = n + i * n + np.arange(n)
        ind2x = i * n + np.arange(n)
        ind2u = i * d + np.arange(d)

        if TimeVarying == 0:
            Gx[np.ix_(ind1, ind2x)] = -A
            Gu[np.ix_(ind1, ind2u)] = -B
            L[ind1, :]              =  C
        else:
            Gx[np.ix_(ind1, ind2x)] = -A[i]
            Gu[np.ix_(ind1, ind2u)] = -B[i]
            L[ind1, :]              =  C[i]

    G = np.hstack((Gx, Gu))


    G_sparse = spmatrix(G[np.nonzero(G)], np.nonzero(G)[0], np.nonzero(G)[1], G.shape)
    E_sparse = spmatrix(E[np.nonzero(E)], np.nonzero(E)[0], np.nonzero(E)[1], E.shape)
    L_sparse = spmatrix(L[np.nonzero(L)], np.nonzero(L)[0], np.nonzero(L)[1], L.shape)

    return G_sparse, E_sparse, L_sparse, G, E


def LMPC_BuildMatIneqConst(N, n, np, linalg, spmatrix, numSS_Points):
    # Buil the matrices for the state constraint in each region. In the region i we want Fx[i]x <= bx[b]
    Fx = np.array([[ 1., 0., 0., 0., 0., 0.],
                   [ 0., 0., 0., 0., 0., 1.],
                   [ 0., 0., 0., 0., 0.,-1.]])

    bx = np.array([[ 10.], # vx max
                   [ 1.], # max ey
                   [ 1.]])# max ey

    # Buil the matrices for the input constraint in each region. In the region i we want Fx[i]x <= bx[b]
    Fu = np.array([[ 1., 0.],
                   [-1., 0.],
                   [ 0., 1.],
                   [ 0.,-1.]])

    bu = np.array([[ 0.5],  # Max Steering
                   [ 0.5],  # Max Steering
                   [ 1.],  # Max Acceleration
                   [ 1.]]) # Max Acceleration

    # Now stuck the constraint matrices to express them in the form Fz<=b. Note that z collects states and inputs
    # Let's start by computing the submatrix of F relates with the state
    rep_a = [Fx] * (N)
    Mat = linalg.block_diag(*rep_a)
    NoTerminalConstr = np.zeros((np.shape(Mat)[0],n)) # No need to constraint also the terminal point
    Fxtot = np.hstack((Mat, NoTerminalConstr))
    bxtot = np.tile(np.squeeze(bx), N)

    # Let's start by computing the submatrix of F relates with the input
    rep_b = [Fu] * (N)
    Futot = linalg.block_diag(*rep_b)
    butot = np.tile(np.squeeze(bu), N)

    # Let's stack all together
    rFxtot, cFxtot = np.shape(Fxtot)
    rFutot, cFutot = np.shape(Futot)
    Dummy1 = np.hstack( (Fxtot                    , np.zeros((rFxtot,cFutot))))
    Dummy2 = np.hstack( (np.zeros((rFutot,cFxtot)), Futot))

    FDummy = np.vstack( ( Dummy1, Dummy2) )
    I = -np.eye(numSS_Points)

    FDummy2 = linalg.block_diag( FDummy, I )

    Fslack = np.zeros((FDummy2.shape[0], n))

    F = np.hstack((FDummy2, Fslack))

    # np.savetxt('F.csv', F, delimiter=',', fmt='%f')

    b = np.hstack((bxtot, butot, np.zeros(numSS_Points)))

    F_sparse = spmatrix(F[np.nonzero(F)], np.nonzero(F)[0], np.nonzero(F)[1], F.shape)
    return F_sparse, b

def LMPC_GetPred(Solution,n,d,N, np):
    xPred = np.squeeze(np.transpose(np.reshape((Solution[np.arange(n*(N+1))]),(N+1,n))))
    uPred = np.squeeze(np.transpose(np.reshape((Solution[n*(N+1)+np.arange(d*N)]),(N, d))))
    lambd = Solution[n*(N+1)+d*N:Solution.shape[0]-n]
    slack = Solution[Solution.shape[0]-n:]
    return xPred, uPred, lambd, slack