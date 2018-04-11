def Regression(x, u, lamb):
    import numpy as np
    # Want to solve W^* = argmin sum_i ||W^T z_i - y_i ||_2^2 + lamb ||W||_F,
    # with z_i = [x_i u_i] and W \in R^{n + d} x n
    Y = x[2:x.shape[0], :]
    X = np.hstack( (x[1:(x.shape[0]-1), :], u[1:(x.shape[0]-1), :]))

    Q = np.linalg.inv( np.dot( X.T, X) + lamb * np.eye( X.shape[1] ) )
    b = np.dot(X.T, Y)
    W = np.dot( Q , b)

    A = W.T[:, 0:6]
    B = W.T[:, 6:8]

    return A, B

def LocLinReg(h, x, u, x0, i, stateFeatures, inputFeatures, qp, matrix, lamb):
    import numpy as np
    # What to learn a model such that: x_{k+1} = A x_k  + B u_k + C
    # print x.shape
    oneVec = np.ones( (x.shape[0], 1) )
    # print "Shape: ",np.array([x0]).shape,  oneVec.shape, oneVec.T.shape
    # print x0
    x0Vec = (np.dot( np.array([x0]).T, oneVec.T )).T
    # print np.dot( np.array([x0]).T, oneVec.T )
    print x[:, i].shape, x0Vec.shape
    diff  = ( x[:, i] - x0Vec )
    norm0 = np.abs(diff)
    index0 =  norm0 < h
    index = index0[0,:]
    K  = ( 1 - ( norm0[index] / h )**2 ) * 3/4

    SelectedStates = x[index,:]
    SelectedInputs = u[index,:]
    X0 = np.hstack( (SelectedStates[:, stateFeatures], SelectedInputs[:, inputFeatures]) )

    M = np.hstack( ( X0, np.ones((X0.shape[0],1)) ) )

    y = x[:,i]
    swift = np.squeeze(np.where(index)) + 1
    b  = matrix( -np.dot( np.dot(M.T, np.diag(K)), y[swift]) )
    Q0 = np.dot( np.dot(M.T, np.diag(K)), M )
    Q  = matrix( Q0 + lamb * np.eye(Q0.shape[0]) )

    res_cons = qp(Q, b) # This is ordered as [A B C]
    print res_cons['x']

    return index