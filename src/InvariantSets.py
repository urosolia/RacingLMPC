import numpy as np
import scipy
from polyhedron import Vrep, Hrep

def GenerateW(Error):
    Vertices = []

    for j in range(0, Error.shape[0]):
        for i in range(0, Error.shape[1]):
            point = np.zeros(Error.shape[1])
            point[i] = Error[j, i]
            Vertices.append(point)

    W = Vrep(Vertices)

    return W

def PropagatePoly(A, W):
    Vetices = W.generators

    NewVetices = []
    for j in range(0, Vetices.shape[0]):
        NewVetices.append( np.dot( A, Vetices[j,:]) )

    return Vrep(NewVetices)

def Invariance(A, W, rho, max_r):
    InvariantSet = []
    rhoW = Vrep(W.generators*rho)
    ListVertex = []
    r_selected = 0
    for i in range(0, 20):#max_r):
        AiW = PropagatePoly(np.power(A, i), W)
        print SetContained(AiW, rhoW)
        if SetContained(AiW, rhoW):
            r_selected = i
            break
        else:
            ListVertex.append(AiW.generators[:,:])

    if r_selected == 0:
        print "============ Error A^i * W not \in rho * W ---> The invariance is not valid ============"
        InvariantSet = 0
    else:
        Vertex = np.vstack(ListVertex)
        InvariantSet = Vrep(1.0/(1.0-rho) * Vertex)

    return InvariantSet

def SetContained(A, B):
    # Checks if A \in B
    Vetices = A.generators
    Contained = True
    for j in range(0, Vetices.shape[0]):
        if not (Belongs(Vetices[j,:], B)):
            Contained = False

    return Contained

def Belongs(x, R):
    return (np.dot(R.A, x) <= R.b).all()


def dlqr(A, B, Q, R):
    """Solve the discrete time lqr controller. From MWM website


    x[k+1] = A x[k] + B u[k]

    cost = sum x[k].T*Q*x[k] + u[k].T*R*u[k]
    """
    # ref Bertsekas, p.151

    # first, try to solve the ricatti equation
    X = np.matrix(scipy.linalg.solve_discrete_are(A, B, Q, R))

    # compute the LQR gain
    K = np.matrix(scipy.linalg.inv(B.T * X * B + R) * (B.T * X * A))

    eigVals, eigVecs = scipy.linalg.eig(A - B * K)

    return K, X, eigVals