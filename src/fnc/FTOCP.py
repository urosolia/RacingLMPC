def FTOCP(M, q, G, L, E, F, b, x0, np, qp, matrix):
    res_cons = qp(M, matrix(q), F, matrix(b), G, E * matrix(x0) + L)
    if res_cons['status'] == 'optimal':
        feasible = 1
    else:
        feasible = 0

    return np.squeeze(res_cons['x']), feasible

def GetPred(Solution,n,d,N, np):
    xPred = np.squeeze(np.transpose(np.reshape((Solution[np.arange(n*(N+1))]),(N+1,n))))
    uPred = np.squeeze(np.transpose(np.reshape((Solution[n*(N+1)+np.arange(d*N)]),(N, d))))

    return xPred, uPred

def BuildMatEqConst(A, B, C, N, n, d, np, spmatrix, TimeVarying):
    # Buil matrices for optimization (Convention from Chapter 15.2 Borrelli, Bemporad and Morari MPC book)
    # We are going to build our optimization vector z \in \mathbb{R}^((N+1) \dot n \dot N \dot d), note that this vector
    # stucks the predicted trajectory x_{k|t} \forall k = t, \ldots, t+N+1 over the horizon and
    # the predicte input u_{k|t} \forall k = t, \ldots, t+N over the horizon
    Gx = np.eye(n * (N + 1))
    Gu = np.zeros((n * (N + 1), d * (N)))

    E = np.zeros((n * (N + 1), n))
    E[np.arange(n)] = np.eye(n)

    L = np.zeros((n * (N + 1), 1))

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

    return G_sparse, E_sparse, L_sparse

def BuildMatIneqConst(N, n, np, linalg, spmatrix):
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
    F = np.vstack( ( Dummy1, Dummy2) )
    b = np.hstack((bxtot, butot))


    F_sparse = spmatrix(F[np.nonzero(F)], np.nonzero(F)[0], np.nonzero(F)[1], F.shape)
    return F_sparse, b

def BuildMatCost(Q, R, P, N, linalg, np, spmatrix, vt):

    b = [Q] * (N)
    Mx = linalg.block_diag(*b)

    c = [R] * (N)
    Mu = linalg.block_diag(*c)

    M0 = linalg.block_diag(Mx, P, Mu)
    xtrack = np.array([vt, 0, 0, 0, 0, 0])
    q = - 2 * np.dot(np.append(np.tile(xtrack, N+1), np.zeros(R.shape[0]*N)), M0)
    M = 2 * M0  # Need to multiply by two because CVX considers 1/2 in front of quadratic cost

    M_sparse = spmatrix(M[np.nonzero(M)], np.nonzero(M)[0], np.nonzero(M)[1], M.shape)
    return M_sparse, q