def Evaluate_e_ey(s, ey, PointAndTangent):
    import numpy as np

    # Compute the segment in which system is evolving
    index = np.all([[s >= PointAndTangent[:, 3]], [s < PointAndTangent[:, 3] + PointAndTangent[:, 4]]], axis=0)
    i = int(np.where(np.squeeze(index))[0])

    if PointAndTangent[i, 5] == 0.0:            # If segment is a straight line
        # Extract the first final and initial point of the segment
        xf  = PointAndTangent[i, 0]
        yf  = PointAndTangent[i, 1]
        xs  = PointAndTangent[i-1, 0]
        ys  = PointAndTangent[i-1, 1]
        psi = PointAndTangent[i, 2]

        # Compute the segment length
        deltaL = PointAndTangent[i, 4]
        reltaL = s - PointAndTangent[i, 3]

        # Do the linear combination
        x = (1-reltaL/deltaL) * xs + reltaL/deltaL * xf + ey * np.cos(psi + np.pi/2)
        y = (1-reltaL/deltaL) * ys + reltaL/deltaL * yf + ey * np.sin(psi + np.pi/2)
    else:
        r = 1 / PointAndTangent[i, 5]           # Extract curvature
        ang = PointAndTangent[i-1, 2]           # Extract angle of the tangent at the initial point (i-1)
        # Compute the center of the arc
        CenterX = PointAndTangent[i-1, 0] + r * np.cos(ang + np.pi / 2)
        CenterY = PointAndTangent[i-1, 1] + r * np.sin(ang + np.pi / 2)
        # Compute the spanned angle
        spanAng = (s - PointAndTangent[i, 3] ) / (np.pi * np.abs(r)) * np.pi

        # Finally compute the point
        x = CenterX + (np.abs(r) + ey) * np.cos(np.pi / 2 - spanAng + ang)
        y = CenterY + (np.abs(r) + ey) * np.sin(np.pi / 2 - spanAng + ang)

    return x, y

def CreateTrack():
    import numpy as np
    # The track is created stucking lines and arc of circles. In particular, each line of spec specifies a segment of
    # length s of radius of curvature r. Note that the radius of curvature is negative for counter clock-wise curve.
    spec = np.array([[5            ,  0],
                     [2 * np.pi / 2, -2], # Note s = 1 * np.pi / 2 and r = -1 ---> Angle spanned = np.pi / 2
                     [5           ,   0],
                     [2 * np.pi / 2, -2],
                     [10           ,  0],
                     [2 * np.pi / 2, -2],
                     [5           ,   0],
                     [2 * np.pi / 2, -2]])

    # Now given the above segments we compute the (x, y) points of the track and the angle of the tangent vector (psi) at
    # these points. For each segment we compute the (x, y, psi) coordinate at the last point of the segment. Furthermore,
    # we compute also the cumulative s at the starting point of the segment at signed curvature
    # PointAndTangent = [x, y, psi, cumulative s, segment length, signed curvature]
    PointAndTangent = np.zeros((spec.shape[0] + 2, 6))

    for i in range(0, spec.shape[0]):
        if spec[i, 1] == 0.0:                              # If the current segment is a straight line
            l   = spec[i, 0]                               # Length of the segments
            ang = PointAndTangent[i, 2]                    # Angle of the tangent vector at the starting point of the segment
            x   = PointAndTangent[i, 0] + l*np.cos(ang)    # x coordinate of the last point of the segment
            y   = PointAndTangent[i, 1] + l*np.sin(ang)    # y coordinate of the last point of the segment
            psi = ang                                      # Angle of the tangent vector at the last point of the segment

            # With the above information create the new line
            if i == 0:
                NewLine = np.array([ x, y, psi, PointAndTangent[i, 3], l, 0])
            else:
                NewLine = np.array([x, y, psi, PointAndTangent[i, 3] + PointAndTangent[i, 4], l, 0])

            PointAndTangent[i+1, :] = NewLine # Write the new info
        else:
            l   = spec[i, 0]                              # Length of the segment
            r   = spec[i, 1]                              # Radius of curvature
            ang = PointAndTangent[i, 2]                   # Angle of the tangent vector at the starting point of the segment
            CenterX = PointAndTangent[i, 0] \
                      + r * np.cos(ang + np.pi / 2)       # x coordinate center of circle
            CenterY = PointAndTangent[i, 1] \
                      + r * np.sin(ang + np.pi / 2)       # y coordinate center of circle
            spanAng = l / np.abs(r)                       # Angle spanned by the circle
            psi     = ang + spanAng * np.sign(r)          # Angle of the tangent vector at the last point of the segment

            x   = CenterX + np.abs(r)*np.cos(np.pi/2 - spanAng + ang) # x coordinate of the last point of the segment
            y   = CenterY + np.abs(r)*np.sin(np.pi/2 - spanAng + ang) # y coordinate of the last point of the segment

            # With the above information create the new line
            if i == 0:
                NewLine = np.array([ x, y, psi, PointAndTangent[i, 3], l, 1/r])
            else:
                NewLine = np.array([x, y, psi, PointAndTangent[i, 3] + PointAndTangent[i, 4], l, 1/r])

            PointAndTangent[i+1, :] = NewLine # Write the new info

    # Now update info on last point
    xs   = PointAndTangent[PointAndTangent.shape[0] - 2, 0]
    ys   = PointAndTangent[PointAndTangent.shape[0] - 2, 1]
    xf   = PointAndTangent[0, 0]
    yf   = PointAndTangent[0, 1]
    psif = PointAndTangent[PointAndTangent.shape[0] - 2, 2]

    l  = np.sqrt( (xf - xs)**2 + (yf - ys)**2)

    NewLine = np.array([xf, yf, psif, PointAndTangent[PointAndTangent.shape[0] - 2, 3] + PointAndTangent[PointAndTangent.shape[0] - 2, 4],  l, 0])
    PointAndTangent[-1, :] = NewLine

    # Plot info
    np.set_printoptions(precision=2)
    print "Track length:", PointAndTangent[-1,3]+PointAndTangent[-1,4]
    return PointAndTangent