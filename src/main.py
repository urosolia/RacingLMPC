import sys
sys.path.append('fnc')
from SysModel import DynModel
from Track import CreateTrack, Evaluate_e_ey
import numpy as np
import matplotlib.pyplot as plt

# ======================================================================================================================
# ====================================== INITIALIZE PARAMETERS =========================================================
# ======================================================================================================================
dt = 1.0/50.0                      # Controller discretization time
Time = 70                       # Simulation time
Points = int(Time/dt)              # Number of points in the simulation
u = np.zeros((Points, 2))          # Initialize the input vector
x      = np.zeros((Points+1, 6))   # Initialize state vector (In curvilinear abscissas)
x_glob = np.zeros((Points+1, 6))   # Initialize the state vector in absolute reference frame

# Create the track. The vector PointAndTangent = [xf yf]
PointAndTangent = CreateTrack()

# ======================================================================================================================
# ======================================= PID path following ===========================================================
# ======================================================================================================================
for i in range(0, Points):
    u[i, 0] = - 0.5 * x[i, 5] - 0.2 * x[i, 3]
    u[i, 1] = 2 - x[i, 0]
    x[i+1,]      = DynModel(x[i, :], u[i, :], np, dt, PointAndTangent, 1)
    x_glob[i+1,] = DynModel(x_glob[i, :], u[i, :], np, dt, PointAndTangent, 0)

print "Number of laps completed: ", int(np.floor(x[-1, 4] / (PointAndTangent[-1, 3] + PointAndTangent[-1, 4])))

# ======================================================================================================================
# ========================================= PLOT TRACK =================================================================
# ======================================================================================================================
Points = np.floor(10*(PointAndTangent[-1, 3] + PointAndTangent[-1, 4]))
Points1 = np.zeros((Points,2))
Points2 = np.zeros((Points,2))
Points0 = np.zeros((Points,2))
for i in range(0,int(Points)):
    Points1[i, :] = Evaluate_e_ey(i*0.1, 1.5, PointAndTangent)
    Points2[i, :] = Evaluate_e_ey(i*0.1,-1.5, PointAndTangent)
    Points0[i, :] = Evaluate_e_ey(i*0.1, 0, PointAndTangent)

plt.figure(1)
plt.plot(PointAndTangent[:,0], PointAndTangent[:,1], 'o')
plt.plot(Points0[:,0], Points0[:,1], '--')
plt.plot(Points1[:,0], Points1[:,1], '-b')
plt.plot(Points2[:,0], Points2[:,1], '-b')
plt.plot(x_glob[:,4], x_glob[:,5], '-r')
plt.show(block=False)

plt.figure(2)
plt.subplot(511)
plt.plot(x[:, 4], x[:, 0])
plt.ylabel('vx')
plt.subplot(512)
plt.plot(x[:, 4], x[:, 1])
plt.ylabel('vy')
plt.subplot(513)
plt.plot(x[:, 4], x[:, 2])
plt.ylabel('wz')
plt.subplot(514)
plt.plot(x[:, 4], x[:, 3])
plt.ylabel('epsi')
plt.subplot(515)
plt.plot(x[:, 4], x[:, 5])
plt.ylabel('ey')
plt.show()
