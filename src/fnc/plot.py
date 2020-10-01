import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches

import pdb

def plotTrajectory(map, x, x_glob, u, stringTitle):
    Points = int(np.floor(10 * (map.PointAndTangent[-1, 3] + map.PointAndTangent[-1, 4])))
    Points1 = np.zeros((Points, 2))
    Points2 = np.zeros((Points, 2))
    Points0 = np.zeros((Points, 2))
    for i in range(0, int(Points)):
        Points1[i, :] = map.getGlobalPosition(i * 0.1, map.halfWidth)
        Points2[i, :] = map.getGlobalPosition(i * 0.1, -map.halfWidth)
        Points0[i, :] = map.getGlobalPosition(i * 0.1, 0)

    plt.figure()
    plt.plot(map.PointAndTangent[:, 0], map.PointAndTangent[:, 1], 'o')
    plt.plot(Points0[:, 0], Points0[:, 1], '--')
    plt.plot(Points1[:, 0], Points1[:, 1], '-b')
    plt.plot(Points2[:, 0], Points2[:, 1], '-b')
    plt.plot(x_glob[:, 4], x_glob[:, 5], '-r')
    plt.title(stringTitle)

    plt.figure()
    plt.subplot(711)
    plt.plot(x[:, 4], x[:, 0], '-o')
    plt.ylabel('vx')
    plt.subplot(712)
    plt.plot(x[:, 4], x[:, 1], '-o')
    plt.ylabel('vy')
    plt.subplot(713)
    plt.plot(x[:, 4], x[:, 2], '-o')
    plt.ylabel('wz')
    plt.subplot(714)
    plt.plot(x[:, 4], x[:, 3], '-o')
    plt.ylabel('epsi')
    plt.subplot(715)
    plt.plot(x[:, 4], x[:, 5], '-o')
    plt.ylabel('ey')
    plt.subplot(716)
    plt.plot(x[:, 4], u[:, 0], '-o')
    plt.ylabel('steering')
    plt.subplot(717)
    plt.plot(x[:, 4], u[:, 1], '-o')
    plt.ylabel('acc')
    plt.title(stringTitle)

def plotClosedLoopLMPC(lmpc, map):
    SS_glob = lmpc.SS_glob
    LapTime  = lmpc.LapTime
    SS      = lmpc.SS
    uSS     = lmpc.uSS

    TotNumberIt = lmpc.it
    Points = int(np.floor(10 * (map.PointAndTangent[-1, 3] + map.PointAndTangent[-1, 4])))
    Points1 = np.zeros((Points, 2))
    Points2 = np.zeros((Points, 2))
    Points0 = np.zeros((Points, 2))
    for i in range(0, int(Points)):
        Points1[i, :] = map.getGlobalPosition(i * 0.1, map.halfWidth)
        Points2[i, :] = map.getGlobalPosition(i * 0.1, -map.halfWidth)
        Points0[i, :] = map.getGlobalPosition(i * 0.1, 0)

    plt.figure()
    plt.plot(map.PointAndTangent[:, 0], map.PointAndTangent[:, 1], 'o')
    plt.plot(Points0[:, 0], Points0[:, 1], '--')
    plt.plot(Points1[:, 0], Points1[:, 1], '-b')
    plt.plot(Points2[:, 0], Points2[:, 1], '-b')

    for i in range(TotNumberIt-5, TotNumberIt):
        plt.plot(SS_glob[i][0:LapTime[i], 4], SS_glob[i][0:LapTime[i], 5], '-r')

    plt.figure()
    plt.subplot(711)
    for i in range(2, TotNumberIt):
        plt.plot(SS[i][0:LapTime[i], 4], SS[i][0:LapTime[i], 0], '-o')
    plt.ylabel('vx')
    plt.subplot(712)
    for i in range(2, TotNumberIt):
        plt.plot(SS[i][0:LapTime[i], 4], SS[i][0:LapTime[i], 1], '-o')
    plt.ylabel('vy')
    plt.subplot(713)
    for i in range(2, TotNumberIt):
        plt.plot(SS[i][0:LapTime[i], 4], SS[i][0:LapTime[i], 2], '-o')
    plt.ylabel('wz')
    plt.subplot(714)
    for i in range(2, TotNumberIt):
        plt.plot(SS[i][0:LapTime[i], 4], SS[i][0:LapTime[i], 3], '-o')
    plt.ylabel('epsi')
    plt.subplot(715)
    for i in range(2, TotNumberIt):
        plt.plot(SS[i][0:LapTime[i], 4], SS[i][0:LapTime[i], 5], '-o')
    plt.ylabel('ey')
    plt.subplot(716)
    for i in range(2, TotNumberIt):
        plt.plot(uSS[i][0:LapTime[i] - 1, 0], '-o')
    plt.ylabel('Steering')
    plt.subplot(717)
    for i in range(2, TotNumberIt):
        plt.plot(uSS[i][0:LapTime[i] - 1, 1], '-o')
    plt.ylabel('Acc')


def animation_xy(map, lmpc, it):
    SS_glob = lmpc.SS_glob
    LapTime = lmpc.LapTime
    SS = lmpc.SS
    uSS = lmpc.uSS

    Points = int(np.floor(10 * (map.PointAndTangent[-1, 3] + map.PointAndTangent[-1, 4])))
    Points1 = np.zeros((Points, 2))
    Points2 = np.zeros((Points, 2))
    Points0 = np.zeros((Points, 2))

    for i in range(0, int(Points)):
        Points1[i, :] = map.getGlobalPosition(i * 0.1, map.halfWidth)
        Points2[i, :] = map.getGlobalPosition(i * 0.1, -map.halfWidth)
        Points0[i, :] = map.getGlobalPosition(i * 0.1, 0)

    plt.figure(200)
    plt.plot(map.PointAndTangent[:, 0], map.PointAndTangent[:, 1], 'o')
    plt.plot(Points0[:, 0], Points0[:, 1], '--')
    plt.plot(Points1[:, 0], Points1[:, 1], '-b')
    plt.plot(Points2[:, 0], Points2[:, 1], '-b')
    plt.plot(SS_glob[it][0:LapTime[it], 4], SS_glob[it][0:LapTime[it], 5], '-ok', label="Closed-loop trajectory",zorder=-1)

    ax = plt.axes()
    SSpoints_x = []; SSpoints_y = []
    xPred = []; yPred = []
    SSpoints, = ax.plot(SSpoints_x, SSpoints_y, 'sb', label="SS",zorder=0)
    line, = ax.plot(xPred, yPred, '-or', label="Predicted Trajectory",zorder=1)

    v = np.array([[ 1.,  1.],
                  [ 1., -1.],
                  [-1., -1.],
                  [-1.,  1.]])
    rec = patches.Polygon(v, alpha=0.7,closed=True, fc='r', ec='k',zorder=10)
    ax.add_patch(rec)

    plt.legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",
                mode="expand", borderaxespad=0, ncol=3)


    N = lmpc.N
    numSS_Points = lmpc.numSS_Points
    for i in range(0, int(lmpc.LapTime[it])):

        xPred = np.zeros((N+1, 1)); yPred = np.zeros((N+1, 1))
        SSpoints_x = np.zeros((numSS_Points, 1)); SSpoints_y = np.zeros((numSS_Points, 1))
        
        for j in range(0, N+1):
            xPred[j,0], yPred[j,0]  = map.getGlobalPosition( lmpc.xStoredPredTraj[it][i][j, 4],
                                                             lmpc.xStoredPredTraj[it][i][j, 5] )

            if j == 0:
                x = SS_glob[it][i, 4]
                y = SS_glob[it][i, 5]
                psi = SS_glob[it][i, 3]
                l = 0.4; w = 0.2
                car_x = [ x + l * np.cos(psi) - w * np.sin(psi), x + l*np.cos(psi) + w * np.sin(psi),
                          x - l * np.cos(psi) + w * np.sin(psi), x - l * np.cos(psi) - w * np.sin(psi)]
                car_y = [ y + l * np.sin(psi) + w * np.cos(psi), y + l * np.sin(psi) - w * np.cos(psi),
                          y - l * np.sin(psi) - w * np.cos(psi), y - l * np.sin(psi) + w * np.cos(psi)]

        for j in range(0, numSS_Points):
            SSpoints_x[j,0], SSpoints_y[j,0] = map.getGlobalPosition(lmpc.SSStoredPredTraj[it][i][j, 4],
                                                                     lmpc.SSStoredPredTraj[it][i][j, 5])
        SSpoints.set_data(SSpoints_x, SSpoints_y)

        line.set_data(xPred, yPred)
        rec.set_xy(np.array([car_x, car_y]).T)
        plt.draw()
        plt.pause(1e-17)

def animation_states(map, LMPCOpenLoopData, lmpc, it):
    SS_glob = lmpc.SS_glob
    LapTime = lmpc.LapTime
    SS = lmpc.SS
    uSS = lmpc.uSS

    xdata = []; ydata = []
    fig = plt.figure(100)

    axvx = fig.add_subplot(3, 2, 1)
    plt.plot(SS[0:LapTime[it], 4, it], SS[0:LapTime[it], 0, it], '-ok', label="Closed-loop trajectory")
    lineSSvx, = axvx.plot(xdata, ydata, 'sb-', label="SS")
    linevx, = axvx.plot(xdata, ydata, 'or-', label="Predicted Trajectory")
    plt.ylabel("vx")
    plt.xlabel("s")

    plt.legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",
                mode="expand", borderaxespad=0, ncol=3)

    axvy = fig.add_subplot(3, 2, 2)
    axvy.plot(SS[0:LapTime[it], 4, it], SS[0:LapTime[it], 1, it], '-ok')
    lineSSvy, = axvy.plot(xdata, ydata, 'sb-')
    linevy, = axvy.plot(xdata, ydata, 'or-')
    plt.ylabel("vy")
    plt.xlabel("s")

    axwz = fig.add_subplot(3, 2, 3)
    axwz.plot(SS[0:LapTime[it], 4, it], SS[0:LapTime[it], 2, it], '-ok')
    lineSSwz, = axwz.plot(xdata, ydata, 'sb-')
    linewz, = axwz.plot(xdata, ydata, 'or-')
    plt.ylabel("wz")
    plt.xlabel("s")

    axepsi = fig.add_subplot(3, 2, 4)
    axepsi.plot(SS[0:LapTime[it], 4, it], SS[0:LapTime[it], 3, it], '-ok')
    lineSSepsi, = axepsi.plot(xdata, ydata, 'sb-')
    lineepsi, = axepsi.plot(xdata, ydata, 'or-')
    plt.ylabel("epsi")
    plt.xlabel("s")

    axey = fig.add_subplot(3, 2, 5)
    axey.plot(SS[0:LapTime[it], 4, it], SS[0:LapTime[it], 5, it], '-ok')
    lineSSey, = axey.plot(xdata, ydata, 'sb-')
    lineey, = axey.plot(xdata, ydata, 'or-')
    plt.ylabel("ey")
    plt.xlabel("s")

    Points = np.floor(10 * (map.PointAndTangent[-1, 3] + map.PointAndTangent[-1, 4]))
    Points1 = np.zeros((Points, 2))
    Points2 = np.zeros((Points, 2))
    Points0 = np.zeros((Points, 2))
    for i in range(0, int(Points)):
        Points1[i, :] = map.getGlobalPosition(i * 0.1, map.halfWidth)
        Points2[i, :] = map.getGlobalPosition(i * 0.1, -map.halfWidth)
        Points0[i, :] = map.getGlobalPosition(i * 0.1, 0)

    axtr = fig.add_subplot(3, 2, 6)
    plt.plot(map.PointAndTangent[:, 0], map.PointAndTangent[:, 1], 'o')
    plt.plot(Points0[:, 0], Points0[:, 1], '--')
    plt.plot(Points1[:, 0], Points1[:, 1], '-b')
    plt.plot(Points2[:, 0], Points2[:, 1], '-b')
    plt.plot(SS_glob[it][0:LapTime[it], 4], SS_glob[it][0:LapTime[it], 5], '-ok')

    SSpoints_x = []; SSpoints_y = []
    xPred = []; yPred = []
    SSpoints_tr, = axtr.plot(SSpoints_x, SSpoints_y, 'sb')
    line_tr, = axtr.plot(xPred, yPred, '-or')

    N = lmpc.N
    numSS_Points = lmpc.numSS_Points
    for i in range(0, int(lmpc.LapTime[it])):

        xPred    = lmpc.xStoredPredTraj[it][i]
        SSpoints = lmpc.SSStoredPredTraj[it][i]

        linevx.set_data(xPred[:, 4], xPred[:, 0]);   axvx.set_title(str(xPred[0, 0]))
        linevy.set_data(xPred[:, 4], xPred[:, 1]);   axvy.set_title(str(xPred[0, 1]))
        linewz.set_data(xPred[:, 4], xPred[:, 2]);   axwz.set_title(str(xPred[0, 2]))
        lineepsi.set_data(xPred[:, 4], xPred[:, 3]); axepsi.set_title(str(xPred[0, 3]))
        lineey.set_data(xPred[:, 4], xPred[:, 5]);   axey.set_title(str(xPred[0, 5]))

        epsiReal = xPred[0, 3]

        lineSSvx.set_data(SSpoints[4,:], SSpoints[0,:])
        lineSSvy.set_data(SSpoints[4,:], SSpoints[1,:])
        lineSSwz.set_data(SSpoints[4,:], SSpoints[2,:])
        lineSSepsi.set_data(SSpoints[4,:], SSpoints[3,:])
        lineSSey.set_data(SSpoints[4,:], SSpoints[5,:])

        xPred = np.zeros((N + 1, 1));yPred = np.zeros((N + 1, 1))
        SSpoints_x = np.zeros((numSS_Points, 1));SSpoints_y = np.zeros((numSS_Points, 1))

        for j in range(0, N + 1):
            xPred[j, 0], yPred[j, 0] = map.getGlobalPosition(lmpc.xStoredPredTraj[it][i][j, 4],
                                                             lmpc.xStoredPredTraj[it][i][j, 5])

        for j in range(0, numSS_Points):
            SSpoints_x[j, 0], SSpoints_y[j, 0] = map.getGlobalPosition(LMPCOpenLoopData.SSused[4, j, i, it],
                                                                       LMPCOpenLoopData.SSused[5, j, i, it])

        line_tr.set_data(xPred, yPred)


        vec = np.array([xPred[0, 0], yPred[0, 0]]) - np.array([SS_glob[i, 4, it], SS_glob[i, 5, it]])

        s, ey, epsi, _ = map.getLocalPosition( SS_glob[i, 4, it], SS_glob[i, 5, it], SS_glob[i, 3, it])
        axtr.set_title(str(s)+" "+str(ey)+" "+str(epsi))

        # axepsi.set_title(str(epsiReal)+" "+str(epsi))
        SSpoints_tr.set_data(SSpoints_x, SSpoints_y)

        plt.draw()
        plt.pause(1e-17)

def saveGif_xyResults(map, LMPCOpenLoopData, lmpc, it):
    SS_glob = lmpc.SS_glob
    LapTime = lmpc.LapTime
    SS = lmpc.SS
    uSS = lmpc.uSS

    Points = int(np.floor(10 * (map.PointAndTangent[-1, 3] + map.PointAndTangent[-1, 4])))
    Points1 = np.zeros((Points, 2))
    Points2 = np.zeros((Points, 2))
    Points0 = np.zeros((Points, 2))
    for i in range(0, int(Points)):
        Points1[i, :] = map.getGlobalPosition(i * 0.1, map.halfWidth)
        Points2[i, :] = map.getGlobalPosition(i * 0.1, -map.halfWidth)
        Points0[i, :] = map.getGlobalPosition(i * 0.1, 0)

    fig = plt.figure(101)
    # plt.ylim((-5, 1.5))
    fig.set_tight_layout(True)
    plt.plot(map.PointAndTangent[:, 0], map.PointAndTangent[:, 1], 'o')
    plt.plot(Points0[:, 0], Points0[:, 1], '--')
    plt.plot(Points1[:, 0], Points1[:, 1], '-b')
    plt.plot(Points2[:, 0], Points2[:, 1], '-b')
    plt.plot(SS_glob[it][0:LapTime[it], 4], SS_glob[it][0:LapTime[it], 5], '-ok', label="Closed-loop trajectory", markersize=1,zorder=-1)

    ax = plt.axes()
    SSpoints_x = []; SSpoints_y = []
    xPred = []; yPred = []
    SSpoints, = ax.plot(SSpoints_x, SSpoints_y, 'og', label="SS",zorder=0)
    line, = ax.plot(xPred, yPred, '-or', label="Predicted Trajectory",zorder=1)

    v = np.array([[ 1.,  1.],
                  [ 1., -1.],
                  [-1., -1.],
                  [-1.,  1.]])
    rec = patches.Polygon(v, alpha=0.7,closed=True, fc='g', ec='k',zorder=10)
    ax.add_patch(rec)

    plt.legend(mode="expand", ncol=3)
    plt.title('Lap: '+str(it))
    # plt.legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",
    #             mode="expand", borderaxespad=0, ncol=3)

    N = lmpc.N
    numSS_Points = lmpc.numSS_Points

    def update(i):
        xPred = np.zeros((N + 1, 1)); yPred = np.zeros((N + 1, 1))
        SSpoints_x = np.zeros((numSS_Points, 1)); SSpoints_y = np.zeros((numSS_Points, 1))

        for j in range(0, N + 1):
            xPred[j, 0], yPred[j, 0] = map.getGlobalPosition(lmpc.xStoredPredTraj[it][i][j, 4],
                                                             lmpc.xStoredPredTraj[it][i][j, 5])

            if j == 0:
                x = SS_glob[i, 4, it]
                y = SS_glob[i, 5, it]
                psi = SS_glob[i, 3, it]
                l = 0.4;w = 0.2
                car_x = [x + l * np.cos(psi) - w * np.sin(psi), x + l * np.cos(psi) + w * np.sin(psi),
                         x - l * np.cos(psi) + w * np.sin(psi), x - l * np.cos(psi) - w * np.sin(psi)]
                car_y = [y + l * np.sin(psi) + w * np.cos(psi), y + l * np.sin(psi) - w * np.cos(psi),
                         y - l * np.sin(psi) - w * np.cos(psi), y - l * np.sin(psi) + w * np.cos(psi)]

        for j in range(0, numSS_Points):
            SSpoints_x[j, 0], SSpoints_y[j, 0] = map.getGlobalPosition(LMPCOpenLoopData.SSused[4, j, i, it],
                                                                       LMPCOpenLoopData.SSused[5, j, i, it])
        SSpoints.set_data(SSpoints_x, SSpoints_y)

        line.set_data(xPred, yPred)

        rec.set_xy(np.array([car_x, car_y]).T)

    anim = FuncAnimation(fig, update, frames=np.arange(0, int(lmpc.LapTime[it])), interval=100)

    anim.save('ClosedLoop.gif', dpi=80, writer='imagemagick')
