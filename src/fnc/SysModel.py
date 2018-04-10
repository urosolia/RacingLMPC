def Curvature(s, PointAndTangent):
    import numpy as np
    TrackLength = PointAndTangent[-1,3]+PointAndTangent[-1,4]

    # In case on a lap after the first one
    while (s > TrackLength):
        s = s - TrackLength

    # Given s \in [0, TrackLength] compute the curvature
    # Compute the segment in which system is evolving
    index = np.all([[s >= PointAndTangent[:, 3]], [s < PointAndTangent[:, 3] + PointAndTangent[:, 4]]], axis=0)
    i = int(np.where(np.squeeze(index))[0])
    curvature = PointAndTangent[i, 5]

    return curvature

def DynModel(x, u, np, dt, PointAndTangent, CurAbsOn):
    # This function computes the system evolution. Note that the discretization is deltaT and therefore is needed that
    # dt <= deltaT and ( dt / deltaT) = integer value

    # Vehicle Parameters
    m  = 1.98
    lf = 0.125
    lr = 0.125
    Iz = 0.024
    Df = 0.8 * m * 9.81 / 2.0
    Cf = 1.25
    Bf = 1.0
    Dr = 0.8 * m * 9.81 / 2.0
    Cr = 1.25
    Br = 1.0

    # Discretization Parameters
    deltaT = 0.01
    x_next = np.zeros(x.shape[0])

    # Extract the value of the states
    delta = u[0]
    a     = u[1]
    vx    = x[0]
    vy    = x[1]
    wz    = x[2]
    epsi  = x[3]
    s     = x[4]
    ey    = x[5]

    # Initialize counter
    i = 0
    while( (i+1) * deltaT <= dt):
        # Compute tire split angle
        alpha_f = delta - np.arctan2( vy + lf * wz, vx )
        alpha_r = - np.arctan2( vy - lf * wz , vx)

        # Compute lateral force at front and rear tire
        Fyf = 2 * Df * np.sin( Cf * np.arctan(Bf * alpha_f ) )
        Fyr = 2 * Dr * np.sin( Cr * np.arctan(Br * alpha_r ) )

        # Propagate the dynamics of deltaT

        x_next[0] = vx   + deltaT * (a - 1 / m * Fyf * np.sin(delta) + wz*vy)
        x_next[1] = vy   + deltaT * (1 / m * (Fyf * np.cos(delta) + Fyr) - wz * vx)
        x_next[2] = wz   + deltaT * (1 / Iz *(lf * Fyf * np.cos(delta) - lr * Fyr) )
        if CurAbsOn == 1:
            cur = Curvature(s, PointAndTangent)
            x_next[3] = epsi + deltaT * ( wz - (vx * np.cos(epsi) - vy * np.sin(epsi)) / (1 - cur * ey) * cur )
            x_next[4] = s    + deltaT * ( (vx * np.cos(epsi) - vy * np.sin(epsi)) / (1 - cur * ey) )
        else:
            cur = 0
            x_next[3] = epsi + deltaT * (wz)
            x_next[4] = s + deltaT * ((vx * np.cos(epsi) - vy * np.sin(epsi)))

        x_next[5] = ey + deltaT * (vx * np.sin(epsi) + vy * np.cos(epsi))

        # Update the value of the states
        vx   = x_next[0]
        vy   = x_next[1]
        wz   = x_next[2]
        epsi = x_next[3]
        s    = x_next[4]
        ey   = x_next[5]

        # Increment counter
        i = i+1


    return x_next