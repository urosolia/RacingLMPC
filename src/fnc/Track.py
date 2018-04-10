def Evaluate_e_ey(s, ey, PointAndTangent):
    import numpy as np
    x = 1000000
    y = 1000000
    for i in range(1, PointAndTangent.shape[0]):
        if (s >= PointAndTangent[i, 3]) and (s < PointAndTangent[i, 3] + PointAndTangent[i, 4]):
            if PointAndTangent[i, 5] == 0.0:
                xf  = PointAndTangent[i, 0]
                yf  = PointAndTangent[i, 1]
                xs  = PointAndTangent[i-1, 0]
                ys  = PointAndTangent[i-1, 1]
                psi = PointAndTangent[i, 2]

                deltaL = PointAndTangent[i, 4]
                reltaL = s - PointAndTangent[i, 3]

                x = (1-reltaL/deltaL) * xs + reltaL/deltaL * xf + ey * np.cos(psi + np.pi/2)
                y = (1-reltaL/deltaL) * ys + reltaL/deltaL * yf + ey * np.sin(psi + np.pi/2)
            else:
                r = 1 / PointAndTangent[i, 5]
                ang = PointAndTangent[i-1, 2]
                CenterX = PointAndTangent[i-1, 0] + r * np.cos(ang + np.pi / 2)
                CenterY = PointAndTangent[i-1, 1] + r * np.sin(ang + np.pi / 2)
                spanAng = (s - PointAndTangent[i, 3] ) / (np.pi * np.abs(r)) * np.pi

                x = CenterX + (np.abs(r) + ey) * np.cos(np.pi / 2 - spanAng + ang)
                y = CenterY + (np.abs(r) + ey) * np.sin(np.pi / 2 - spanAng + ang)


    if x == 1000000:
        print "Error point not on the track, s: ", s, " Track length:", PointAndTangent[-1,3]+PointAndTangent[-1,4]

    return x, y

def CreateTrack():
    import numpy as np
    # Each line of the vector spec specifies a segment of length s of radius of curvature r. Note that the radius of
    # curvature is negative for counter clock-wise curve.
    spec = np.array([[5            ,  0],
                     [10 * np.pi / 2, -10], # Note s = 1 * np.pi / 2 and r = -1 ---> Angle span an angle = np.pi / 2
                     [10           ,   0],
                     [10 * np.pi / 2, -10],
                     [20           ,  0],
                     [10 * np.pi / 2, -10],
                     [10           ,   0],
                     [10 * np.pi / 2, -10]])

    # PointAndTangent = [x, y, psi, cumulative s, segment length, signed curvature]
    PointAndTangent = np.zeros((spec.shape[0] + 2, 6))

    for i in range(0, spec.shape[0]):
        if spec[i, 1] == 0.0:
            l   = spec[i, 0]
            ang = PointAndTangent[i, 2]
            x   = PointAndTangent[i, 0] + l*np.cos(ang)
            y   = PointAndTangent[i, 1] + l*np.sin(ang)
            psi = ang

            if i == 0:
                NewLine = np.array([ x, y, psi, PointAndTangent[i, 3], l, 0])
            else:
                NewLine = np.array([x, y, psi, PointAndTangent[i, 3] + PointAndTangent[i, 4], l, 0])

            PointAndTangent[i+1, :] = NewLine
        else:
            l   = spec[i, 0]
            r   = spec[i, 1]
            ang = PointAndTangent[i, 2]
            CenterX = PointAndTangent[i, 0] + r * np.cos(ang + np.pi / 2)
            CenterY = PointAndTangent[i, 1] + r * np.sin(ang + np.pi / 2)
            spanAng = l / np.abs(r)
            psi     = ang + spanAng * np.sign(r)

            x   = CenterX + np.abs(r)*np.cos(np.pi/2 - spanAng + ang)
            y   = CenterY + np.abs(r)*np.sin(np.pi/2 - spanAng + ang)

            if i == 0:
                NewLine = np.array([ x, y, psi, PointAndTangent[i, 3], l, 1/r])
            else:
                NewLine = np.array([x, y, psi, PointAndTangent[i, 3] + PointAndTangent[i, 4], l, 1/r])

            PointAndTangent[i+1, :] = NewLine

    xs   = PointAndTangent[PointAndTangent.shape[0] - 2, 0]
    ys   = PointAndTangent[PointAndTangent.shape[0] - 2, 1]
    xf   = PointAndTangent[0, 0]
    yf   = PointAndTangent[0, 1]
    psif = PointAndTangent[PointAndTangent.shape[0] - 2, 2]

    l  = np.sqrt( (xf - xs)**2 + (yf - ys)**2)

    NewLine = np.array([xf, yf, psif, PointAndTangent[PointAndTangent.shape[0] - 2, 3] + PointAndTangent[PointAndTangent.shape[0] - 2, 4],  l, 0])
    PointAndTangent[-1, :] = NewLine

    np.set_printoptions(precision=2)
    print "Track length:", PointAndTangent[-1,3]+PointAndTangent[-1,4]
    return PointAndTangent