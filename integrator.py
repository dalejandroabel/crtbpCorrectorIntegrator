from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt


class crtbpCorrectorPropagator():

    def __init__(self, initial_conditions, massratio, period=None, reference="barycenter"):
        """class of a crtbp orbit

        Args:
            initial_conditions (array): initial state vector [x,y,z,vx,vy,vz]
            massratio (float): mass ratio between bodies: m2/(m1+m2)
            period (float, optional): if known, period of orbit. Defaults to None.
            reference (str, optional): reference system to the given initial conditions. Defaults to "barycenter". Possible ["barycenter","secondary"]

        Raises:
            ValueError: incorrect reference value given
        """

        VALID = ["barycenter", "secondary"]
        if reference not in VALID:
            raise ValueError(
                "reference point not valid, select a valid value ['barycenter','secondary']")

        self.ic = initial_conditions
        self.x, self.y, self.z, self.vx, self.vy, self.vz = initial_conditions
        self.mu = massratio
        self.period = period
        self.centered = reference == "barycenter"

    def EoM(self, t, Y):
        """Equations of Movement 

        Args:
            t (float): Time of evaluation (Not used)
            Y (array): State Vector [x,y,z,vx,vy,vz]

        Returns:
            array: temporal derivative of the state vector[vx,vy,vx,ax,ay,az]
        """
        x, y, z, vx, vy, vz = Y

        if self.centered:
            r1 = np.sqrt((x+self.mu)**2+y**2+z**2)
            mmur1 = (1-self.mu)/(r1**3)
            r2 = np.sqrt(x**2+y**2+z**2-2 * x*(1-self.mu)+(1-self.mu)**2)
            mur2 = self.mu/r2**3

            ax = 2*vy+x-mmur1*(x+self.mu)-mur2*(x-(1-self.mu))
            ay = -2*vx+y-mmur1*y-mur2*y
            az = -mmur1*z-mur2*z

        else:
            r1 = np.sqrt((x+1)**2+y**2+z**2)
            mmur1 = (1-self.mu)/(r1**3)

            r2 = np.sqrt(x**2+y**2+z**2)
            mmur2 = self.mu/r2**3

            ax = 2*vy+x+1-self.mu-mmur1*(x+1)-mmur2*x
            ay = -2*vx+y-mmur1*y-mmur2*y
            az = -mmur1*z-mmur2*z

        return np.array([vx, vy, vz, ax, ay, az])

    def propagate(self, time=None, plot=False):
        """Propagation of the orbit in a given time. 

        Args:
            time (float, optional): Final time of propagation, if None the period is taken. Defaults to None.
            plot (bool, optional): If plot = True, the orbits is shown. Defaults to False.

        Returns:
            array[6,1e5]: array containing the state for each time from 0 to 'time'
        """

        if not time:
            time = self.period
        t_eval = np.linspace(0, time, 10000)

        solution = solve_ivp(
            self.EoM, [0, time], self.ic, t_eval=t_eval,
            method="DOP853", atol=1e-12, rtol=1e-12)

        propagation = solution.y

        if plot:
            plt.plot(propagation[0, :], propagation[1, :])
            if self.centered:
                plt.plot(1-mu, 0, "ro")
                if min(propagation[0, :]) < 0:
                    plt.plot(-mu, 0, "bo")
            else:
                plt.plot(0, 0, "ro")
                if min(propagation[0, :]) < -1:
                    plt.plot(-1, 0, "bo")
            plt.axis("equal")
            plt.show()

        return propagation

    def FBarycenter(self, x, y, z):

        r1 = np.sqrt(x**2+2*x*self.mu+self.mu**2+y**2+z**2)
        mmur1 = (1-self.mu)/(r1**3)

        r2 = np.sqrt(x**2+y**2+z**2-2*x*(1-self.mu)+(1-self.mu)**2)
        mmur2 = self.mu/r2**3
        # Derivative of state vector
        # Diagonals

        dg1dx = 1 - mmur1*(1 - 3*(x+self.mu)**2/(r1**2)) - \
            mmur2*(1 - 3*(x+self.mu-1)**2/(r2**2))
        dg2dy = 1 - mmur1*(1 - 3*(y**2)/(r1**2)) - mmur2*(1 - 3*(y**2)/(r2**2))
        dg3dz = -mmur1*(1 - 3*(z**2)/(r1**2)) - mmur2*(1 - 3*(z**2)/(r2**2))

        # G12
        dg1dy = 3*((1-self.mu)*y*(x+self.mu)/r1 **
                   5 + self.mu*y*(x+self.mu-1)/r2**5)
        # G13
        dg1dz = 3*((1-self.mu)*z*(x+self.mu)/r1 **
                   5 + self.mu*z*(x+self.mu-1)/r2**5)
        # G23
        dg2dz = 3*(1-self.mu)*z*y/r1**5 + 3*self.mu*z*y/r2**5

        I = np.identity(3)

        G = np.array([[dg1dx, dg1dy, dg1dz],
                      [dg1dy, dg2dy, dg2dz],
                      [dg1dz, dg2dz, dg3dz]])  # Symmetric matrix

        H = np.array([[0, 2, 0],
                      [-2, 0, 0],
                      [0, 0, 0]])

        F = np.zeros((6, 6), dtype=np.double)
        F[0:3, 3:6] = I
        F[3:6, 0:3] = G
        F[3:6, 3:6] = H

        return F

    def FSecondary(self, x, y, z):

        r1 = np.sqrt((x+1)**2+y**2+z**2)
        mmur1 = (1-self.mu)/(r1**3)

        r2 = np.sqrt(x**2+y**2+z**2)
        mmur2 = self.mu/r2**3
        # Derivative of state vector
        # Diagonals

        dg1dx = 1 - mmur1*(1 - 3*(x+1)**2/(r1**2)) - \
            mmur2*(1 - 3*(x)**2/(r2**2))
        dg2dy = 1 - mmur1*(1 - 3*(y**2)/(r1**2)) - mmur2*(1 - 3*(y**2)/(r2**2))
        dg3dz = -mmur1*(1 - 3*(z**2)/(r1**2)) - mmur2*(1 - 3*(z**2)/(r2**2))

        # G12
        dg1dy = 3*((1-self.mu)*y*(x+1)/r1**5 + self.mu*y*x/r2**5)
        # G13
        dg1dz = 3*((1-self.mu)*z*(x+1)/r1**5 + self.mu*z*x/r2**5)
        # G23
        dg2dz = 3*(1-self.mu)*z*y/r1**5 + 3*self.mu*z*y/r2**5

        I = np.identity(3)

        G = np.array([[dg1dx, dg1dy, dg1dz],
                      [dg1dy, dg2dy, dg2dz],
                      [dg1dz, dg2dz, dg3dz]])  # Symmetric matrix

        H = np.array([[0, 2, 0],
                      [-2, 0, 0],
                      [0, 0, 0]])

        F = np.zeros((6, 6), dtype=np.double)
        F[0:3, 3:6] = I
        F[3:6, 0:3] = G
        F[3:6, 3:6] = H

        return F

    def STMPropagator(self, t, Y):

        # Last 6 components
        x, y, z, vx, vy, vz = Y[36:]
        # Give matrix shape to the first 36 components
        STM = Y[:36].reshape((6, 6))
        # Propagate initial position

        dydt = self.EoM(t, [x, y, z, vx, vy, vz])
        if self.centered:
            F = self.FBarycenter(x, y, z)
        else:
            F = self.FSecondary(x, y, z)

        # Matrix Product Between F and M
        dSTMdt = np.array(np.matmul(F, STM)).reshape((36))

        return np.concatenate([dSTMdt, dydt])

    def STM(self, statevector=None, t=None, flag=None):
        initialstate = np.zeros(42)
        initialstate[:36] = np.identity(6).flatten()

        if None in statevector:
            statevector = self.ic
        initialstate[36:42] = statevector

        if not t:
            t = self.period

        solution = solve_ivp(self.STMPropagator, y0=initialstate,
                             t_span=[0, t], method="DOP853", atol=1e-11, rtol=1e-11, events=(flag))
        y = solution.y[36:, :]
        STM = solution.y[:36, :].reshape((6, 6, solution.t.shape[0]))

        if flag:
            event = np.argmin(np.abs(solution.t_events[0]-t/2))
            HalfState = solution.y_events[0][event][36:]
            HalfSTM = solution.y_events[0][event][:36].reshape((6, 6))
            HalfTime = solution.t_events[0][event]

            return HalfState, HalfSTM, HalfTime

        finalState = y[:, -1]
        finalSTM = STM[:, :, -1]

        return finalState, finalSTM

    def crossingFlag(self, t, Y):
        return Y[36:][1]

    def corrector(self):
        X = self.ic
        corrected = False
        counter = 0
        Xmid, STM, Halftime = self.STM(statevector=X, flag=self.crossingFlag)

        while (abs(Xmid[1]) > 1e-10 or abs(Xmid[3]) > 1e-10) and counter < 10:
            Xmid, STM, Halftime = self.STM(
                statevector=X, flag=self.crossingFlag)
            ax = self.EoM(Halftime, Xmid)[3]
            deltavy = -Xmid[3]/(STM[3, 4]-ax*STM[1, 4]/Xmid[4])
            X[4] += deltavy
            counter += 1
            corrected = (abs(Xmid[1]) > 1e-10 or abs(Xmid[3]) > 1e-10)
        if corrected:
            self.ic = X
            return None

        X = self.ic
        counter = 0

        Xmid, STM, Halftime = self.STM(statevector=X, flag=self.crossingFlag)
        while abs(Xmid[1]) > 1e-10 or abs(Xmid[3]) > 1e-10 and counter < 10:
            Xmid, STM, Halftime = self.STM(
                statevector=X, flag=self.crossingFlag)
            ax = self.EoM(Halftime, Xmid)
            deltavy = -Xmid[3]/(STM[3, 4]-ax*STM[1, 4]/Xmid[4])
            X[4] += deltavy
            corrected = (abs(Xmid[1]) > 1e-10 or abs(Xmid[3]) > 1e-10)
        if corrected:
            self.ic = X
            return None
        raise Warning('Orbit not corrected')
        return None


if __name__ == "__main__":
    initial_conditions = np.array([-0.013,
                                   0, 0, 0, 0.02493428539689399, 0])
    mu = 0.00000304043
    period = 8.405150131082836
    initial_conditions[0] += (1-mu)
    system = crtbpCorrectorPropagator(
        initial_conditions, mu, period,
        reference="barycenter")
    # print(system.ic)
    system.corrector()
    system.propagate(plot=True)
    # print(system.ic)
