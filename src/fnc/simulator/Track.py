import numpy as np
import numpy.linalg as la
import pdb

class Map():
    """map object
    Attributes:
        getGlobalPosition: convert position from (s, ey) to (X,Y)
    """
    def __init__(self, halfWidth):
        """Initialization
        halfWidth: track halfWidth
        Modify the vector spec to change the geometry of the track
        """
        # Goggle-shaped track
        # self.slack = 0.15
        # self.halfWidth = halfWidth
        # spec = np.array([[60 * 0.03, 0],
        #                  [80 * 0.03, -80 * 0.03 * 2 / np.pi],
        #                  # Note s = 1 * np.pi / 2 and r = -1 ---> Angle spanned = np.pi / 2
        #                  [20 * 0.03, 0],
        #                  [80 * 0.03, -80 * 0.03 * 2 / np.pi],
        #                  [40 * 0.03, +40 * 0.03 * 10 / np.pi],
        #                  [60 * 0.03, -60 * 0.03 * 5 / np.pi],
        #                  [40 * 0.03, +40 * 0.03 * 10 / np.pi],
        #                  [80 * 0.03, -80 * 0.03 * 2 / np.pi],
        #                  [20 * 0.03, 0],
        #                  [80 * 0.03, -80 * 0.03 * 2 / np.pi]])

        # L-shaped track
        self.halfWidth = 0.4
        self.slack = 0.45
        lengthCurve = 4.5
        spec = np.array([[1.0, 0],
                         [lengthCurve, lengthCurve / np.pi],
                         # Note s = 1 * np.pi / 2 and r = -1 ---> Angle spanned = np.pi / 2
                         [lengthCurve / 2, -lengthCurve / np.pi],
                         [lengthCurve, lengthCurve / np.pi],
                         [lengthCurve / np.pi * 2, 0],
                         [lengthCurve / 2, lengthCurve / np.pi]])


        # spec = np.array([[1.0, 0],
        #                  [4.5, -4.5 / np.pi],
        #                  # Note s = 1 * np.pi / 2 and r = -1 ---> Angle spanned = np.pi / 2
        #                  [2.0, 0],
        #                  [4.5, -4.5 / np.pi],
        #                  [1.0, 0]])

        # Now given the above segments we compute the (x, y) points of the track and the angle of the tangent vector (psi) at
        # these points. For each segment we compute the (x, y, psi) coordinate at the last point of the segment. Furthermore,
        # we compute also the cumulative s at the starting point of the segment at signed curvature
        # PointAndTangent = [x, y, psi, cumulative s, segment length, signed curvature]
        PointAndTangent = np.zeros((spec.shape[0] + 1, 6))
        for i in range(0, spec.shape[0]):
            if spec[i, 1] == 0.0:              # If the current segment is a straight line
                l = spec[i, 0]                 # Length of the segments
                if i == 0:
                    ang = 0                          # Angle of the tangent vector at the starting point of the segment
                    x = 0 + l * np.cos(ang)          # x coordinate of the last point of the segment
                    y = 0 + l * np.sin(ang)          # y coordinate of the last point of the segment
                else:
                    ang = PointAndTangent[i - 1, 2]                 # Angle of the tangent vector at the starting point of the segment
                    x = PointAndTangent[i-1, 0] + l * np.cos(ang)  # x coordinate of the last point of the segment
                    y = PointAndTangent[i-1, 1] + l * np.sin(ang)  # y coordinate of the last point of the segment
                psi = ang  # Angle of the tangent vector at the last point of the segment


                if i == 0:
                    NewLine = np.array([x, y, psi, PointAndTangent[i, 3], l, 0])
                else:
                    NewLine = np.array([x, y, psi, PointAndTangent[i-1, 3] + PointAndTangent[i-1, 4], l, 0])

                PointAndTangent[i, :] = NewLine  # Write the new info
            else:
                l = spec[i, 0]                 # Length of the segment
                r = spec[i, 1]                 # Radius of curvature


                if r >= 0:
                    direction = 1
                else:
                    direction = -1

                if i == 0:
                    ang = 0                                                      # Angle of the tangent vector at the
                                                                                 # starting point of the segment
                    CenterX = 0 \
                              + np.abs(r) * np.cos(ang + direction * np.pi / 2)  # x coordinate center of circle
                    CenterY = 0 \
                              + np.abs(r) * np.sin(ang + direction * np.pi / 2)  # y coordinate center of circle
                else:
                    ang = PointAndTangent[i - 1, 2]                              # Angle of the tangent vector at the
                                                                                 # starting point of the segment
                    CenterX = PointAndTangent[i-1, 0] \
                              + np.abs(r) * np.cos(ang + direction * np.pi / 2)  # x coordinate center of circle
                    CenterY = PointAndTangent[i-1, 1] \
                              + np.abs(r) * np.sin(ang + direction * np.pi / 2)  # y coordinate center of circle

                spanAng = l / np.abs(r)  # Angle spanned by the circle
                psi = wrap(ang + spanAng * np.sign(r))  # Angle of the tangent vector at the last point of the segment

                angleNormal = wrap((direction * np.pi / 2 + ang))
                angle = -(np.pi - np.abs(angleNormal)) * (sign(angleNormal))
                x = CenterX + np.abs(r) * np.cos(
                    angle + direction * spanAng)  # x coordinate of the last point of the segment
                y = CenterY + np.abs(r) * np.sin(
                    angle + direction * spanAng)  # y coordinate of the last point of the segment

                if i == 0:
                    NewLine = np.array([x, y, psi, PointAndTangent[i, 3], l, 1 / r])
                else:
                    NewLine = np.array([x, y, psi, PointAndTangent[i-1, 3] + PointAndTangent[i-1, 4], l, 1 / r])

                PointAndTangent[i, :] = NewLine  # Write the new info
            # plt.plot(x, y, 'or')


        xs = PointAndTangent[-2, 0]
        ys = PointAndTangent[-2, 1]
        xf = 0
        yf = 0
        psif = 0

        # plt.plot(xf, yf, 'or')
        # plt.show()
        l = np.sqrt((xf - xs) ** 2 + (yf - ys) ** 2)

        NewLine = np.array([xf, yf, psif, PointAndTangent[-2, 3] + PointAndTangent[-2, 4], l, 0])
        PointAndTangent[-1, :] = NewLine

        self.PointAndTangent = PointAndTangent
        self.TrackLength = PointAndTangent[-1, 3] + PointAndTangent[-1, 4]

    def getGlobalPosition(self, s, ey):
        """coordinate transformation from curvilinear reference frame (e, ey) to inertial reference frame (X, Y)
        (s, ey): position in the curvilinear reference frame
        """

        # wrap s along the track
        while (s > self.TrackLength):
            s = s - self.TrackLength

        # Compute the segment in which system is evolving
        PointAndTangent = self.PointAndTangent

        index = np.all([[s >= PointAndTangent[:, 3]], [s < PointAndTangent[:, 3] + PointAndTangent[:, 4]]], axis=0)
        i = int(np.where(np.squeeze(index))[0])

        if PointAndTangent[i, 5] == 0.0:  # If segment is a straight line
            # Extract the first final and initial point of the segment
            xf = PointAndTangent[i, 0]
            yf = PointAndTangent[i, 1]
            xs = PointAndTangent[i - 1, 0]
            ys = PointAndTangent[i - 1, 1]
            psi = PointAndTangent[i, 2]

            # Compute the segment length
            deltaL = PointAndTangent[i, 4]
            reltaL = s - PointAndTangent[i, 3]

            # Do the linear combination
            x = (1 - reltaL / deltaL) * xs + reltaL / deltaL * xf + ey * np.cos(psi + np.pi / 2)
            y = (1 - reltaL / deltaL) * ys + reltaL / deltaL * yf + ey * np.sin(psi + np.pi / 2)
        else:
            r = 1 / PointAndTangent[i, 5]  # Extract curvature
            ang = PointAndTangent[i - 1, 2]  # Extract angle of the tangent at the initial point (i-1)
            # Compute the center of the arc
            if r >= 0:
                direction = 1
            else:
                direction = -1

            CenterX = PointAndTangent[i - 1, 0] \
                      + np.abs(r) * np.cos(ang + direction * np.pi / 2)  # x coordinate center of circle
            CenterY = PointAndTangent[i - 1, 1] \
                      + np.abs(r) * np.sin(ang + direction * np.pi / 2)  # y coordinate center of circle

            spanAng = (s - PointAndTangent[i, 3]) / (np.pi * np.abs(r)) * np.pi

            angleNormal = wrap((direction * np.pi / 2 + ang))
            angle = -(np.pi - np.abs(angleNormal)) * (sign(angleNormal))

            x = CenterX + (np.abs(r) - direction * ey) * np.cos(
                angle + direction * spanAng)  # x coordinate of the last point of the segment
            y = CenterY + (np.abs(r) - direction * ey) * np.sin(
                angle + direction * spanAng)  # y coordinate of the last point of the segment

        return x, y

    def getLocalPosition(self, x, y, psi):
        """coordinate transformation from inertial reference frame (X, Y) to curvilinear reference frame (s, ey)
        (X, Y): position in the inertial reference frame
        """
        PointAndTangent = self.PointAndTangent
        CompletedFlag = 0



        for i in range(0, PointAndTangent.shape[0]):
            if CompletedFlag == 1:
                break

            if PointAndTangent[i, 5] == 0.0:  # If segment is a straight line
                # Extract the first final and initial point of the segment
                xf = PointAndTangent[i, 0]
                yf = PointAndTangent[i, 1]
                xs = PointAndTangent[i - 1, 0]
                ys = PointAndTangent[i - 1, 1]

                psi_unwrap = np.unwrap([PointAndTangent[i - 1, 2], psi])[1]
                epsi = psi_unwrap - PointAndTangent[i - 1, 2]
                # Check if on the segment using angles
                if (la.norm(np.array([xs, ys]) - np.array([x, y]))) == 0:
                    s  = PointAndTangent[i, 3]
                    ey = 0
                    CompletedFlag = 1

                elif (la.norm(np.array([xf, yf]) - np.array([x, y]))) == 0:
                    s = PointAndTangent[i, 3] + PointAndTangent[i, 4]
                    ey = 0
                    CompletedFlag = 1
                else:
                    if np.abs(computeAngle( [x,y] , [xs, ys], [xf, yf])) <= np.pi/2 and np.abs(computeAngle( [x,y] , [xf, yf], [xs, ys])) <= np.pi/2:
                        v1 = np.array([x,y]) - np.array([xs, ys])
                        angle = computeAngle( [xf,yf] , [xs, ys], [x, y])
                        s_local = la.norm(v1) * np.cos(angle)
                        s       = s_local + PointAndTangent[i, 3]
                        ey      = la.norm(v1) * np.sin(angle)

                        if np.abs(ey)<= self.halfWidth + self.slack:
                            CompletedFlag = 1

            else:
                xf = PointAndTangent[i, 0]
                yf = PointAndTangent[i, 1]
                xs = PointAndTangent[i - 1, 0]
                ys = PointAndTangent[i - 1, 1]

                r = 1 / PointAndTangent[i, 5]  # Extract curvature
                if r >= 0:
                    direction = 1
                else:
                    direction = -1

                ang = PointAndTangent[i - 1, 2]  # Extract angle of the tangent at the initial point (i-1)

                # Compute the center of the arc
                CenterX = xs + np.abs(r) * np.cos(ang + direction * np.pi / 2)  # x coordinate center of circle
                CenterY = ys + np.abs(r) * np.sin(ang + direction * np.pi / 2)  # y coordinate center of circle

                # Check if on the segment using angles
                if (la.norm(np.array([xs, ys]) - np.array([x, y]))) == 0:
                    ey = 0
                    psi_unwrap = np.unwrap([ang, psi])[1]
                    epsi = psi_unwrap - ang
                    s = PointAndTangent[i, 3]
                    CompletedFlag = 1
                elif (la.norm(np.array([xf, yf]) - np.array([x, y]))) == 0:
                    s = PointAndTangent[i, 3] + PointAndTangent[i, 4]
                    ey = 0
                    psi_unwrap = np.unwrap([PointAndTangent[i, 2], psi])[1]
                    epsi = psi_unwrap - PointAndTangent[i, 2]
                    CompletedFlag = 1
                else:
                    arc1 = PointAndTangent[i, 4] * PointAndTangent[i, 5]
                    arc2 = computeAngle([xs, ys], [CenterX, CenterY], [x, y])
                    if np.sign(arc1) == np.sign(arc2) and np.abs(arc1) >= np.abs(arc2):
                        v = np.array([x, y]) - np.array([CenterX, CenterY])
                        s_local = np.abs(arc2)*np.abs(r)
                        s    = s_local + PointAndTangent[i, 3]
                        ey   = -np.sign(direction) * (la.norm(v) - np.abs(r))
                        psi_unwrap = np.unwrap([ang + arc2, psi])[1]
                        epsi = psi_unwrap - (ang + arc2)

                        if np.abs(ey) <= self.halfWidth + self.slack:
                            CompletedFlag = 1

        if epsi>1.0:
            pdb.set_trace()

        if CompletedFlag == 0:
            s    = 10000
            ey   = 10000
            epsi = 10000

            print("Error!! POINT OUT OF THE TRACK!!!! <==================")
            pdb.set_trace()

        return s, ey, epsi, CompletedFlag

    def curvature(self, s):
        """curvature computation
        s: curvilinear abscissa at which the curvature has to be evaluated
        PointAndTangent: points and tangent vectors defining the map (these quantities are initialized in the map object)
        """
        TrackLength = self.PointAndTangent[-1,3]+self.PointAndTangent[-1,4]

        # In case on a lap after the first one
        while (s > TrackLength):
            s = s - TrackLength

        # Given s \in [0, TrackLength] compute the curvature
        # Compute the segment in which system is evolving
        index = np.all([[s >= self.PointAndTangent[:, 3]], [s < self.PointAndTangent[:, 3] + self.PointAndTangent[:, 4]]], axis=0)

        i = int(np.where(np.squeeze(index))[0])
        curvature = self.PointAndTangent[i, 5]

        return curvature

    def getAngle(self, s, epsi):
        """TO DO
        """
        TrackLength = self.PointAndTangent[-1,3]+self.PointAndTangent[-1,4]

        # In case on a lap after the first one
        while (s > TrackLength):
            s = s - TrackLength

        # Given s \in [0, TrackLength] compute the curvature
        # Compute the segment in which system is evolving
        index = np.all([[s >= self.PointAndTangent[:, 3]], [s < self.PointAndTangent[:, 3] + self.PointAndTangent[:, 4]]], axis=0)

        i = int(np.where(np.squeeze(index))[0])

        if i > 0:
            ang = self.PointAndTangent[i - 1, 2]
        else:
            ang = 0

        if self.PointAndTangent[i, 5] == 0:
            r= 0
        else:
            r = 1 / self.PointAndTangent[i, 5]  # Radius of curvature

        if r == 0:
            # On a straight part of the circuit
            angle_at_s = ang + epsi
        else:
            # On a curve
            cumulative_s = self.PointAndTangent[i, 3]
            relative_s = s - cumulative_s
            spanAng = relative_s / np.abs(r)  # Angle spanned by the circle
            psi = wrap(ang + spanAng * np.sign(r))  # Angle of the tangent vector at the last point of the segment
            # pdb.set_trace()
            angle_at_s = psi + epsi

        return angle_at_s

# ======================================================================================================================
# ======================================================================================================================
# ====================================== Internal utilities functions ==================================================
# ======================================================================================================================
# ======================================================================================================================
def computeAngle(point1, origin, point2):
    # The orientation of this angle matches that of the coordinate system. Tha is why a minus sign is needed
    v1 = np.array(point1) - np.array(origin)
    v2 = np.array(point2) - np.array(origin)

    dot = v1[0] * v2[0] + v1[1] * v2[1]  # dot product between [x1, y1] and [x2, y2]
    det = v1[0] * v2[1] - v1[1] * v2[0]  # determinant
    angle = np.arctan2(det, dot)  # atan2(y, x) or atan2(sin, cos)

    return angle # np.arctan2(sinang, cosang)

def wrap(angle):
    if angle < -np.pi:
        w_angle = 2 * np.pi + angle
    elif angle > np.pi:
        w_angle = angle - 2 * np.pi
    else:
        w_angle = angle

    return w_angle

def sign(a):
    if a >= 0:
        res = 1
    else:
        res = -1

    return res