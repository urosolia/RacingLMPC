import sys
sys.path.append('fnc')
from SysModel import DynModel
from Track import CreateTrack, Evaluate_e_ey
import numpy as np
import matplotlib.pyplot as plt

dt = 1.0/50.0
Time = 2*140
Points = int(Time/dt)
u = np.zeros((Points, 2))
x      = np.zeros((Points+1, 6))
x_glob = np.zeros((Points+1, 6))

PointAndTangent = CreateTrack()

deltaMax = 3

for i in range(0, Points):
    u[i, 0] = - 0.3 * x[i, 5] - 0.2 * x[i, 3]
    if u[i, 0] < -deltaMax:
        u[i, 0] = -deltaMax
    if u[i, 0] > deltaMax:
        u[i, 0] = deltaMax

    u[i, 1] = 1 - x[i, 0]
    x[i+1,]      = DynModel(x[i, :], u[i, :], np, dt, PointAndTangent, 1)

    # print "x = ", x[i, :], " u = ", u[i, :]
    x_glob[i+1,] = DynModel(x_glob[i, :], u[i, :], np, dt, PointAndTangent, 0)

print "Number of laps completed: ", int(np.floor(x[-1, 4] / (PointAndTangent[-1, 3] + PointAndTangent[-1, 4])))
# ========================================= PLOT TRACK =================================================================
Points = np.floor(10*(PointAndTangent[-1, 3] + PointAndTangent[-1, 4]))
Points1 = np.zeros((Points,2))
Points2 = np.zeros((Points,2))
Points0 = np.zeros((Points,2))
for i in range(0,int(Points)):
    Points1[i, :] = Evaluate_e_ey(i*0.1, 1.5, PointAndTangent)
    Points2[i, :] = Evaluate_e_ey(i*0.1,-1.5, PointAndTangent)
    Points0[i, :] = Evaluate_e_ey(i*0.1, 0, PointAndTangent)


plt.plot(PointAndTangent[:,0], PointAndTangent[:,1], 'o')
plt.plot(Points0[:,0], Points0[:,1], '--')
plt.plot(Points1[:,0], Points1[:,1], '-b')
plt.plot(Points2[:,0], Points2[:,1], '-b')
plt.plot(x_glob[:,4], x_glob[:,5], '-r')

plt.show()

