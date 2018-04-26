def FTOCP(M, q, G, L, E, F, b, x0, np, qp, matrix):
    res_cons = qp(M, matrix(q), F, matrix(b+np.dot(E, x0)), G,  matrix(L))
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
    Gx = np.eye(n * (N+1))
    Gu = np.zeros((n * (N+1), d * (N)))

    Ltot = np.zeros((n * (N+1), 1))

    Etot = np.zeros((n * (N + 1), n))
    Etot[np.arange(n)] = np.eye(n)

    for i in range(0, N):
        ind1 =  n + i * n + np.arange(n)
        ind2x = i * n + np.arange(n)
        ind2u = i * d + np.arange(d)

        if TimeVarying == 0:
            Gx[np.ix_(ind1, ind2x)] = -A
            Gu[np.ix_(ind1, ind2u)] = -B
            Ltot[ind1, :]           =  C
        else:
            Gx[np.ix_(ind1, ind2x)] = -A[i]
            Gu[np.ix_(ind1, ind2u)] = -B[i]
            Ltot[ind1, :]           =  C[i]

    Gtot = np.hstack((Gx, Gu))

    # Now select the rows related with model dynamics
    G = Gtot[n:Gtot.shape[0], :]
    L = Ltot[n:Ltot.shape[0]]

    # np.savetxt('G_MPC.csv', G, delimiter=',', fmt='%f')
    # np.savetxt('E_MPC.csv', E, delimiter=',', fmt='%f')

    G_sparse = spmatrix(G[np.nonzero(G)], np.nonzero(G)[0], np.nonzero(G)[1], G.shape)
    L_sparse = spmatrix(L[np.nonzero(L)], np.nonzero(L)[0], np.nonzero(L)[1], L.shape)

    return G_sparse, L_sparse

def BuildMatIneqConst(N, n, np, linalg, spmatrix, Ke, Error_Bounds):
    # Buil the matrices for the state constraint in each region. In the region i we want Fx[i]x <= bx[b]
    Fx = np.array([[ 1., 0., 0., 0., 0., 0.],
                   [ 0., 0., 0., 0., 0., 1.],
                   [ 0., 0., 0., 0., 0.,-1.]])

    bx = np.array([[ 10.], # vx max
                   [ 2.], # max ey
                   [ 2.]])# max ey

    # Buil the matrices for the input constraint in each region. In the region i we want Fx[i]x <= bx[b]
    Fux = np.array([-Ke[0, :],  # Need to take into account  u_tot = -Kx + u
                     Ke[0, :],  # Need to take into account -u_tot =  Kx - u
                    -Ke[1, :],  # Need to take into account  u_tot = -Kx + u
                     Ke[1, :]]) # Need to take into account -u_tot =  Kx - u

    print "Fux ", np.squeeze(Fux)
    print "Fx ", Fx

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

    rep_c = [np.squeeze(Fux)] * (N)       # This parts considers that u_tot = -Kx + u
    Mat1  = linalg.block_diag(*rep_c)
    NoTerminalConstr_x_N = np.zeros((np.shape(Mat1)[0],n)) # No need to consider x_N as there is not u_N
    Fxutot = np.hstack((Mat1, NoTerminalConstr_x_N))


    # Let's stack all together
    rFxtot, cFxtot = np.shape(Fxtot)
    rFutot, cFutot = np.shape(Futot)
    Dummy1 = np.hstack( (Fxtot                    , np.zeros((rFxtot,cFutot))))
    Dummy2 = np.hstack( (Fxutot                   , Futot))
    F_horizon = np.vstack( ( Dummy1, Dummy2) )
    b_horizon = np.hstack((bxtot, butot))


    # Now constraint x(t) - x_0 \in O_e, basically -x(t) + Lower_bound <= -x_0 <= -x(t) + Upper_bound
    # --> -x_0 <= -x(t) + Upper_bound, x_0 <= x(t) - Lower_bound)
    F_0 = np.zeros((2*n, F_horizon.shape[1]))
    F_0[0:n,0:n]    = -np.eye(n)
    F_0[n:2*n, 0:n] =  np.eye(n)

    F = np.vstack((F_0, F_horizon))
    Lower_bound = Error_Bounds[0]
    Upper_bound = Error_Bounds[1]
    b = np.hstack((Upper_bound, -Lower_bound, b_horizon))

    E = np.zeros((F.shape[0], n))
    E[0:n,0:n]   = -np.eye(n)
    E[n:2*n,0:n] =  np.eye(n)

    # np.savetxt('F_MPC.csv', F, delimiter=',', fmt='%f')
    # np.savetxt('b_MPC.csv', b, delimiter=',', fmt='%f')
    # np.savetxt('E_MPC.csv', E, delimiter=',', fmt='%f')



    F_sparse = spmatrix(F[np.nonzero(F)], np.nonzero(F)[0], np.nonzero(F)[1], F.shape)
    return F_sparse, b, E

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