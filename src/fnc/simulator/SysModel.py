import numpy as np
import pdb
import datetime
from Utilities import wrap

class Simulator():
    """Vehicle simulator
    Attributes:
        Sim: given a Controller object run closed-loop simulation and write the data in the ClosedLoopData object
    """
    def __init__(self, map, dt = 0.1, multiLap = True, flagLMPC = False):
        """Initialization
        map: map
        lap: number of laps to run. If set to 0 then the simulation is completed when ClosedLoopData is full
        flagLMPC: set to 0 for standart controller. Set to 1 for LMPC --> at iteration j add data to SS^{j-1} (look line 9999)
        """
        self.map = map
        self.multiLap = multiLap
        self.flagLMPC = flagLMPC
        self.dt = dt

    def sim(self, x0,  Controller, maxSimTime = 100):
        """Simulate closed-loop system
        """
        x_cl      = [x0[0]]
        x_cl_glob = [x0[1]]
        u_cl   = []
        sim_t  = 0
        SimulationTime = 0
        
        i=0
        flagExt = False
        while (i<int(maxSimTime/self.dt)) and (flagExt==False):
            Controller.solve(x_cl[-1])
            u_cl.append(Controller.uPred[0,:].copy())

            if self.flagLMPC == True:
                Controller.addPoint(x_cl[-1], u_cl[-1])

            xt, xt_glob = self.dynModel(x_cl[-1], x_cl_glob[-1], u_cl[-1])
            
            x_cl.append(xt)
            x_cl_glob.append(xt_glob)

            if (self.multiLap == False) and ( x_cl[-1][4] > (self.map.TrackLength)):
                print("Lap completed")
                flagExt = True
            i += 1

        xF = [np.array(x_cl[-1]) - np.array([0, 0, 0, 0, self.map.TrackLength, 0]), np.array(x_cl_glob[-1])]
        x_cl.pop()
        x_cl_glob.pop()

        return np.array(x_cl), np.array(u_cl), np.array(x_cl_glob), xF

    def dynModel(self, x, x_glob, u):
        # This method computes the system evolution. Note that the discretization is deltaT and therefore is needed that
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
        deltaT = 0.001
        x_next     = np.zeros(x.shape[0])
        cur_x_next = np.zeros(x.shape[0])

        # Extract the value of the states
        delta = u[0]
        a     = u[1]

        psi = x_glob[3]
        X = x_glob[4]
        Y = x_glob[5]

        vx    = x[0]
        vy    = x[1]
        wz    = x[2]
        epsi  = x[3]
        s     = x[4]
        ey    = x[5]

        # Initialize counter
        i = 0
        while( (i+1) * deltaT <= self.dt):
            # Compute tire split angle
            alpha_f = delta - np.arctan2( vy + lf * wz, vx )
            alpha_r = - np.arctan2( vy - lf * wz , vx)

            # Compute lateral force at front and rear tire
            Fyf = Df * np.sin( Cf * np.arctan(Bf * alpha_f ) )
            Fyr = Dr * np.sin( Cr * np.arctan(Br * alpha_r ) )

            # Propagate the dynamics of deltaT
            x_next[0] = vx  + deltaT * (a - 1 / m * Fyf * np.sin(delta) + wz*vy)
            x_next[1] = vy  + deltaT * (1 / m * (Fyf * np.cos(delta) + Fyr) - wz * vx)
            x_next[2] = wz  + deltaT * (1 / Iz *(lf * Fyf * np.cos(delta) - lr * Fyr) )
            x_next[3] = psi + deltaT * (wz)
            x_next[4] =   X + deltaT * ((vx * np.cos(psi) - vy * np.sin(psi)))
            x_next[5] =   Y + deltaT * (vx * np.sin(psi)  + vy * np.cos(psi))

            cur = self.map.curvature(s)
            cur_x_next[0] = vx   + deltaT * (a - 1 / m * Fyf * np.sin(delta) + wz*vy)
            cur_x_next[1] = vy   + deltaT * (1 / m * (Fyf * np.cos(delta) + Fyr) - wz * vx)
            cur_x_next[2] = wz   + deltaT * (1 / Iz *(lf * Fyf * np.cos(delta) - lr * Fyr) )
            cur_x_next[3] = epsi + deltaT * ( wz - (vx * np.cos(epsi) - vy * np.sin(epsi)) / (1 - cur * ey) * cur )
            cur_x_next[4] = s    + deltaT * ( (vx * np.cos(epsi) - vy * np.sin(epsi)) / (1 - cur * ey) )
            cur_x_next[5] = ey   + deltaT * (vx * np.sin(epsi) + vy * np.cos(epsi))

            # Update the value of the states
            psi  = x_next[3]
            X    = x_next[4]
            Y    = x_next[5]

            vx   = cur_x_next[0]
            vy   = cur_x_next[1]
            wz   = cur_x_next[2]
            epsi = cur_x_next[3]
            s    = cur_x_next[4]
            ey   = cur_x_next[5]

            if (s < 0):
                print("Start Point: ", x, " Input: ", u)
                print("x_next: ", x_next)

            # Increment counter
            i = i+1

        # Noises
        noise_vx = np.max([-0.05, np.min([np.random.randn() * 0.01, 0.05])])
        noise_vy = np.max([-0.05, np.min([np.random.randn() * 0.01, 0.05])])
        noise_wz = np.max([-0.05, np.min([np.random.randn() * 0.005, 0.05])])

        cur_x_next[0] = cur_x_next[0] + 0.01*noise_vx
        cur_x_next[1] = cur_x_next[1] + 0.01*noise_vy
        cur_x_next[2] = cur_x_next[2] + 0.01*noise_wz

        return cur_x_next, x_next



