def LMPC(npG, L, npE, E_LMPC, F, b, z0, x0, np, qp, matrix, datetime, la, zSS, xSS, Qfun, N, n, d, spmatrix, numSS_Points, Qslack, Q_LMPC, R_LMPC, it, swifth):

    zSS_Points1, _ = SelectPoints(zSS, Qfun, it-1, z0, numSS_Points/2, np, la, swifth)
    zSS_Points2, _ = SelectPoints(zSS, Qfun, it-2, z0, numSS_Points/2, np, la, swifth)

    zSS_Points = np.hstack((zSS_Points1, zSS_Points2))

    xSS_Points1, Sel_xQfun1 = SelectPoints(xSS, Qfun, it-1, x0, numSS_Points/2, np, la, swifth)
    xSS_Points2, Sel_xQfun2 = SelectPoints(xSS, Qfun, it-2, x0, numSS_Points/2, np, la, swifth)

    xSS_Points = np.hstack((xSS_Points1, xSS_Points2))

    Sel_xQfun = np.hstack((Sel_xQfun1, Sel_xQfun2))


    G, E = LMPC_TermConstr(npG, npE, N, n, d, np, spmatrix, zSS_Points, xSS_Points)

    M, q = LMPC_BuildMatCost(Sel_xQfun, numSS_Points, N, np, spmatrix, Qslack, Q_LMPC, R_LMPC)

    startTimer = datetime.datetime.now()
    Sol, feasible = LMPC_FTOCP(M, q, G, L, E, E_LMPC, F, b, x0, np, qp, matrix)
    zPred, uPred, xPred, lambdPred, slack = LMPC_GetPred(Sol, n,d,N, np)

    if x0[4] > 18.5:
        np.savetxt('xSS_Points.csv', xSS_Points, delimiter=',', fmt='%f')
        np.savetxt('Sel_xQfun.csv', Sel_xQfun, delimiter=',', fmt='%f')
    # print SS_Points.shape, lambdPred.shape
    # print "Term Costr \n", np.dot(SS_Points, lambdPred),"\n", xPred[:,-1], "\n", xPred.T
    # print "Here lambda ", lambdPred, lambdPred.shape, np.dot(np.ones(lambdPred.shape[0]), lambdPred)
    # print "Check Slack", slack, np.dot(SS_Points, lambdPred)-xPred[:,-1]

    # name = raw_input("Please enter name: ")


    endTimer = datetime.datetime.now();
    deltaTimer = endTimer - startTimer

    return Sol, feasible, deltaTimer, slack

def SelectPoints(SS, Qfun, it, x0, numSS_Points, np, la, swifth):
    x = SS[:, :, it]

    oneVec = np.ones((x.shape[0], 1))
    x0Vec = (np.dot(np.array([x0]).T, oneVec.T)).T
    diff = x - x0Vec
    norm = la.norm(diff, 1, axis=1)
    MinNorm = np.argmin(norm)

    SS_Points = x[swifth + MinNorm:swifth + MinNorm + numSS_Points, :].T
    Sel_Qfun = Qfun[swifth + MinNorm:swifth + MinNorm + numSS_Points, it]
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


def LMPC_TermConstr(G, E, N ,n ,d ,np, spmatrix, zSS_Points, xSS_Points):
    # Update the matrices for the Equality constraint in the LMPC. Now we need an extra row to constraint the terminal point to be equal to a point in SS
    # The equality constraint has now the form: G_LMPC*z = E_LMPC*x0 + TermPoint.
    # Note that the vector TermPoint is updated to constraint the predicted trajectory into a point in SS. This is done in the FTOCP_LMPC function

    TermCons_z = np.zeros((n, 2 * (N + 1) * n + N * d))
    TermCons_z[:,  N * n:(N + 1) * n] = np.eye(n)  # Constrain the terminal point for z

    TermCons_x = np.zeros((n, 2*(N + 1) * n + N * d))
    TermCons_x[:, (N + 1) * n + N * d + N * n: 2 * (N + 1) * n + N * d] = np.eye(n) # Constrain the terminal point for x

    G_enlarged = np.vstack((G, TermCons_z, np.zeros((1,2*(N + 1) * n + N * d)), TermCons_x, np.zeros((1,2*(N + 1) * n + N * d)) ))

    # np.savetxt('G_enlarged.csv', G_enlarged, delimiter=',', fmt='%f')

    G_lambda_z = np.zeros(( n+1, 2*xSS_Points.shape[1] + 2*n))
    G_lambda_z[0:n, 0:zSS_Points.shape[1]] = -zSS_Points
    G_lambda_z[ -1, 0:zSS_Points.shape[1]] = np.ones((1,zSS_Points.shape[1]))
    G_lambda_z[0:n, 2*zSS_Points.shape[1]:2*zSS_Points.shape[1]+n] = np.eye(6)

    G_lambda_x = np.zeros((n + 1, 2 * xSS_Points.shape[1] + 2 * n))
    G_lambda_x[0:n, xSS_Points.shape[1]:2 * xSS_Points.shape[1]] = -xSS_Points
    G_lambda_x[-1, xSS_Points.shape[1]:2 * xSS_Points.shape[1]] = np.ones((1, xSS_Points.shape[1]))
    G_lambda_x[0:n, 2 * xSS_Points.shape[1] + n:2 * xSS_Points.shape[1] + 2 * n] = np.eye(6)

    G_lambda = np.vstack(( np.zeros((G.shape[0], G_lambda_z.shape[1])), G_lambda_z, G_lambda_x ))

    # np.savetxt('G_lambda.csv', G_lambda, delimiter=',', fmt='%f')


    G_LMPC = np.hstack((G_enlarged, G_lambda))

    # print "Before term constr: ", E.shape

    E_LMPC = E

    # np.savetxt('Gtot.csv', G_LMPC, delimiter=',', fmt='%f')
    # np.savetxt('E_LMPC.csv', E_LMPC, delimiter=',', fmt='%f')

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

    M00 = linalg.block_diag(Mx, P, Mu, Mx, P)
    M0  = linalg.block_diag(M00, np.zeros((numSS_Points, numSS_Points)), np.zeros((numSS_Points, numSS_Points)), Qslack, Qslack)
    xtrack = np.array([vt, 0, 0, 0, 0, 0])
    # print np.append(np.zeros(Q.shape[0]*(N+1) + R.shape[0]*N), np.tile(xtrack, N+1)).shape, M00.shape
    q0 = - 2 * np.dot(np.append(np.zeros(Q.shape[0]*(N+1) + R.shape[0]*N), np.tile(xtrack, N+1)), M00)
    # print q0.shape, Sel_Qfun.shape, Q.shape[0], np.zeros(Q.shape[0]).shape

    q  = np.append(np.append(np.append(q0, 0*Sel_Qfun), Sel_Qfun), np.zeros(2*Q.shape[0]) )

    M = 2 * M0  # Need to multiply by two because CVX considers 1/2 in front of quadratic cost

    # np.savetxt('q.csv', q, delimiter=',', fmt='%f')
    # np.savetxt('M.csv', M, delimiter=',', fmt='%f')

    M_sparse = spmatrix(M[np.nonzero(M)], np.nonzero(M)[0], np.nonzero(M)[1], M.shape)
    return M_sparse, q

def LMPC_FTOCP(M, q, G, L, E, E_LMPC, F, b, x0, np, qp, matrix):
    from numpy.linalg import matrix_rank
    G_cvx = F
    A_cvx = G

    # print A_cvx.size, G_cvx.size
    # print M.size,  matrix(q).size, F.size, matrix(b).size, np.dot(E_LMPC, x0).shape, G.size, L.size
#    print G.size, E.size, np.dot(E, x0).size
    res_cons = qp(M, matrix(q),         F, matrix(b + np.dot(E_LMPC, x0)),           G, L + E * matrix(x0))
    if res_cons['status'] == 'optimal':
        feasible = 1
    else:
        feasible = 0

    return np.squeeze(res_cons['x']), feasible


def LMPC_BuildMatEqConst(Ax, Bx, A, B, C, N, n, d, np, spmatrix, Ke, linalg):
    from numpy import linalg as LA
    # Buil matrices for optimization (Convention from Chapter 15.2 Borrelli, Bemporad and Morari MPC book)
    # We are going to build our optimization vector z \in \mathbb{R}^((N+1) \dot n \dot N \dot d), note that this vector
    # stucks the predicted trajectory x_{k|t} \forall k = t, \ldots, t+N+1 over the horizon and
    # the predicte input u_{k|t} \forall k = t, \ldots, t+N over the horizon
    Gx = np.eye(n * (N + 1))
    Gxu = np.zeros((n * (N + 1), d * (N)))

    Gz  = np.eye(n * (N + 1))
    Gze = np.zeros((n * (N + 1),n * (N + 1)))
    Gzu = np.zeros((n * (N + 1), d * (N)))


    Lold      = np.zeros((n * (N + 1), 1))
    ErrorProp = np.zeros((n * (N + 1), n))

    for i in range(0, N):
        ind1 = n + i * n + np.arange(n)
        ind2x = i * n + np.arange(n)
        ind2u = i * d + np.arange(d)

        Gx[np.ix_(ind1, ind2x)]  = -Ax
        Gxu[np.ix_(ind1, ind2u)] = -Bx

        Gz[np.ix_(ind1, ind2x)]             = -A[i]
        if i == 0:
            Gze[np.ix_(ind1, ind2x)]        =  np.dot(B[i], Ke) - np.dot(np.dot(B[i], Ke), LA.matrix_power(Ax, i))
        else:
            Gze[np.ix_(ind1, ind2x)]        =  np.dot(B[i], Ke)
            Gze[np.ix_(ind1, np.arange(n))] = -np.dot(np.dot(B[i], Ke), LA.matrix_power(Ax, i))
        ErrorProp[ind1, :]                  = -np.dot(np.dot(B[i], Ke), LA.matrix_power(Ax, i))
        Gzu[np.ix_(ind1, ind2u)]            = -B[i]
        Lold[ind1, :]                       =  C[i]

    L = np.vstack(( np.zeros((n*N,1)), Lold[n:n * (N + 1) + 2*(n + 1),:], np.zeros((n,1)),np.zeros((2*(n+1),1)) ))# 2*(n+1) for the terminal constraints
    sizeL = L.shape[0]
    L[sizeL -(n + 1) - 1] = 1  # Summmation of lamba for x must add up to 1
    L[sizeL - 1] = 1  # Summmation of lamba for z must add up to 1

    E = np.vstack(( np.zeros((n*N,n)), ErrorProp[n:n * (N + 1) + 2*(n + 1),:], np.zeros((n,n)),np.zeros((2*(n+1),n)) ))# 2*(n+1) for the terminal constraints

    # print "Shape L :", L.shape
    # np.savetxt('L.csv', L, delimiter=',', fmt='%f')

    Gxtot = np.hstack((Gx, Gxu))
    Gztot = np.hstack((Gzu, Gz)) # Note that for z is need to put the input first as these are the same which are applied to x

    Zeros = np.zeros((n * N, n*(N+1)))

    InitialCond = np.zeros((n, 2*n * (N + 1) + d * (N)))
    InitialCond[:, 0:n]                                           =  np.eye(n)
    InitialCond[:, n * (N + 1) + d * (N):n * (N + 1) + d * (N)+n] = -np.eye(n)

    # print InitialCond.shape
    # print np.hstack((Gxtot[n:n * (N + 1), :],                              Zeros)).shape
    # print np.hstack((                  Zeros, Gztot[n:n * (N + 1), :])).shape

    FeedbackTerm = Gze[n:n * (N + 1), :]
    # np.savetxt('Gze.csv', Gze, delimiter=',', fmt='%f')
    # np.savetxt('Gz.csv', Gz, delimiter=',', fmt='%f')
    # print np.hstack((           FeedbackTerm, Gztot[n:n * (N + 1), :])).shape
    G = np.vstack((np.hstack((Gxtot[n:n * (N + 1), :],                   Zeros)),
                   np.hstack((           FeedbackTerm, Gztot[n:n * (N + 1), :])),
                   InitialCond))

    # np.savetxt('G_LMPC.csv', G, delimiter=',', fmt='%f')
    # np.savetxt('Eprop_LMPC.csv', E, delimiter=',', fmt='%f')
    # np.savetxt('L.csv', L, delimiter=',', fmt='%f')

    G_sparse = spmatrix(G[np.nonzero(G)], np.nonzero(G)[0], np.nonzero(G)[1], G.shape)
    E_sparse = spmatrix(E[np.nonzero(E)], np.nonzero(E)[0], np.nonzero(E)[1], E.shape)
    L_sparse = spmatrix(L[np.nonzero(L)], np.nonzero(L)[0], np.nonzero(L)[1], L.shape)

    return G_sparse, E_sparse, L_sparse, G, E


def LMPC_BuildMatIneqConst(N, n, np, linalg, spmatrix, numSS_Points, Error_Bounds, Ke):
    # Buil the matrices for the state constraint in each region. In the region i we want Fx[i]x <= bx[b]
    Fx = np.array([[ 1., 0., 0., 0., 0., 0.],
                   [ 0., 0., 0., 0., 0., 1.],
                   [ 0., 0., 0., 0., 0.,-1.]])

    bx = np.array([[ 3.], # vx max
                   [ 1.8], # max ey
                   [ 1.8]])# max ey

    # Buil the matrices for the input constraint in each region. In the region i we want Fx[i]x <= bx[b]
    Fu = np.array([[ 1., 0.],
                   [-1., 0.],
                   [ 0., 1.],
                   [ 0.,-1.]])

    bu = np.array([[ 0.5],  # Max Steering
                   [ 0.5],  # Max Steering
                   [ 2.],  # Max Acceleration
                   [ 2.]]) # Max Acceleration

    # Buil the matrices for the input constraint in each region. In the region i we want Fx[i]x <= bx[b]
    Fux = np.array([-Ke[0, :],  # Need to take into account  u_tot = -Kx + u
                     Ke[0, :],  # Need to take into account -u_tot =  Kx - u
                    -Ke[1, :],  # Need to take into account  u_tot = -Kx + u
                     Ke[1, :]]) # Need to take into account -u_tot =  Kx - u

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

    rep_c = [np.squeeze(Fux)] * (N)       # This parts considers that u_tot = -Kx + u
    Mat1  = linalg.block_diag(*rep_c)
    NoTerminalConstr_x_N = np.zeros((np.shape(Mat1)[0],n)) # No need to consider x_N as there is not u_N
    Fxutot = np.hstack((Mat1, NoTerminalConstr_x_N))


    # Let's stack all together
    rFxtot, cFxtot = np.shape(Fxtot)
    rFutot, cFutot = np.shape(Futot)
    Dummy1 = np.hstack( (Fxtot  , np.zeros((rFxtot,cFutot))))
    Dummy2 = np.hstack( (Fxutot , Futot))

    FDummy = np.hstack((np.vstack( ( Dummy1, Dummy2)), np.zeros( (rFxtot + rFutot, cFxtot) ) ))
    I = -np.eye(numSS_Points)

    FDummy2 = linalg.block_diag( FDummy, I, I ) # Make sure the lambda are => 0

    Fslack = np.zeros((FDummy2.shape[0], 2*n)) # Adding terms related with the slack variable associated with the terminal constraint

    F_horizon = np.hstack((FDummy2, Fslack))
    b_horizon = np.hstack((bxtot, butot, np.zeros(2*numSS_Points)))

    # Now constraint x(t) - x_0 \in O_e, basically -x(t) + Lower_bound <= -x_0 <= -x(t) + Upper_bound
    # --> -x_0 <= -x(t) + Upper_bound, x_0 <= x(t) - Lower_bound)
    F_0 = np.zeros((2 * n, F_horizon.shape[1]))
    F_0[0:n, 0:n] = -np.eye(n)
    F_0[n:2 * n, 0:n] = np.eye(n)

    F = np.vstack((F_0, F_horizon))
    Lower_bound = Error_Bounds[0]
    Upper_bound = Error_Bounds[1]
    b = np.hstack((Upper_bound, -Lower_bound, b_horizon))

    # np.savetxt('F.csv', F, delimiter=',', fmt='%f')
    # np.savetxt('b.csv', b, delimiter=',', fmt='%f')

    E = np.zeros((F.shape[0], n))
    E[0:n,0:n]   = -np.eye(n)
    E[n:2*n,0:n] =  np.eye(n)

    F_sparse = spmatrix(F[np.nonzero(F)], np.nonzero(F)[0], np.nonzero(F)[1], F.shape)
    return F_sparse, b, E

def LMPC_GetPred(Solution,n,d,N, np):
    zPred = np.squeeze(np.transpose(np.reshape((Solution[np.arange(n*(N+1))]),(N+1,n))))
    uPred = np.squeeze(np.transpose(np.reshape((Solution[n*(N+1)+np.arange(d*N)]),(N, d))))
    xPred = np.squeeze(np.transpose(np.reshape((Solution[n*(N+1)+d*N + np.arange(n * (N + 1))]), (N + 1, n))))
    lambd = Solution[2*n*(N+1)+d*N:Solution.shape[0]-2*n]
    slack = Solution[Solution.shape[0]-2*n:]
    return zPred, uPred, xPred, lambd, slack