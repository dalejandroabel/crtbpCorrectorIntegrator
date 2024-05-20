from scipy.integrate import solve_ivp
from pycrtbp.particles import _Particle
import numpy as np

SolarSystem = {'Mercury': 1.66012e-07,
      'Venus': 2.44784e-06,
      'Moon': 0.0121437,
      'Earth': 3.04043e-06,
      'Mars': 3.22717e-07,
      'Io': 4.70422e-05,
      'Europa': 2.52801e-05,
      'Ganymede': 7.80432e-05,
      'Callisto': 5.66663e-05,
      'Jupiter': 0.000953886,
      'Enceladus': 1.90115e-07,
      'Tethys': 1.08647e-06,
      'Dione': 1.9276e-06,
      'Rhea': 4.05844e-06,
      'Titan': 0.000236639,
      'Iapetus': 3.17714e-06,
      'Saturn': 0.000285805,
      'Ariel': 1.44051e-05,
      'Umbriel': 1.46864e-05,
      'Titania': 3.91676e-05,
      'Oberon': 3.54362e-05,
      'Uranus': 4.36607e-05,
      'Triton': 0.000208819,
      'Neptune': 5.15114e-05}

class System():

    def __init__(self, mu):
        """_summary_

        Args:
            mu (float): Mass ratio  m1/(m1+m2)
        """
        self.mu = mu
        self.refsystem = 'barycenter'
        self.particles = {}

    @property
    def refsystem(self):
        return self._refsystem

    @refsystem.setter
    def refsystem(self, refsystem):
        if refsystem not in ["barycenter", "secondary"]:
            raise ValueError(
                "reference point not valid, select a valid value: [barycenter,secondary]")
        self._refsystem = refsystem

    @property
    def mu(self):
        return self._mu

    @mu.setter
    def mu(self, mu):
        if type(mu) not in (float, int, np.number):
            if type(mu) == str:
                if mu in SolarSystem.keys():
                    self._mu = SolarSystem[mu]
                    return
            raise Exception("parameter mu must be a number")
        self._mu = mu

    def add(self, r, v, name=None):
        if not (r is None) and not (v is None):
            self._HandleVectors(r, v)
        if name in self.particles.keys():
            raise Exception(
                f"Particle with name (id) = {name} already in particles list")
        if name:
            self.particles[name] = _Particle(r, v, name)
        else:
            self.particles[len(self.particles)] = _Particle(
                r, v, str(len(self.particles)))

    def delete(self, p=None, name=None):
        if (p == None) and (name == None):
            self.particles = {}
        if p != None:
            if type(p) in (list, tuple, np.array):
                for particle in p:
                    self.particles.pop(particle)
            else:
                self.particles.pop(p)
        if name:
            if type(name) in (list, tuple):
                for particle in name:
                    self.particles.pop(particle)
            else:
                self.particles.pop(name)

    def getLagrangePoints(self):
        mu1 = 1 - self.mu
        mu2 = self.mu
        alpha = (mu2/(3*mu1))**(1/3)

        L1 = [mu1 - (alpha - (alpha**2)/3 - (alpha**3)/9-(alpha**4)*(23/81)),
              0]

        L2 = [mu1 + (alpha + (alpha**2)/3 - (alpha**3)/9-(alpha**4)*(31/81)),
              0]

        mu2mu1 = mu2/mu1

        L3 = [-mu2 - 1 - (-(7/12)*(mu2mu1)+(7/12)*((mu2mu1)**2) - (13223/20736)*((mu2mu1)**3)),
              0]
        L4 = [1/2 - mu2, np.sqrt(3)/2]
        L5 = [1/2 - mu2, -np.sqrt(3)/2]
        return {"L1": L1, "L2": L2, "L3": L3, "L4": L4, "L5": L5}

    def getJacobiConstant(self, r=None, v=None, p=None):
        if p != None:
            x, y, z = self.particles[p].r
            vx, vy, vz = self.particles[p].v

        else:
            if (r is None) or (v is None):
                raise ValueError("Missing vector")
            self._HandleVectors(r, v)
            x, y, z = r
            vx, vy, vz = v
        r1 = np.sqrt((x+self.mu)**2 + y**2 + z**2)
        r2 = np.sqrt((x-(1-self.mu))**2 + y**2 + z**2)
        Cj = x**2 + y**2 + 2 * \
            ((1-self.mu)/r1 + (self.mu)/r2) - vx**2 - vy**2 - vz**2
        return Cj

    def _HandleVectors(self, r, v):
        if np.array(r).reshape(-1, 3).shape[1] != 3 or np.array(v).reshape(-1, 3).shape[1] != 3:
            raise ValueError(
                "Input does not have correct shape [nParticles,3]")
        if np.array(r).shape[0] != np.array(v).shape[0]:
            raise ValueError(
                "Positional vector and velocity vector are not compatible (different particles number)")

    def _EoM(self, t, Y):
        """Equations of Movement 

        Args:
            t (float): Time of evaluation (Not used)
            Y (array): State Vector [x,y,z,vx,vy,vz]

        Returns:
            array: temporal derivative of the state vector[vx,vy,vx,ax,ay,az]
        """
        x, y, z, vx, vy, vz = Y

        if self._refsystem == 'barycenter':
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

    def propagate(self, time=None, r=None, v=None, p=None, N=10000, from_current=False, **kwargs):
        if not (p is None):
            if not time:
                if self.particles[p].period == None:
                    raise ValueError(
                        'Particle selected but not time given or the particle does not have period estimated')
                time = self.particles[p].period

            if from_current:
                x, y, z = self.particles[p].r
                vx, vy, vz = self.particles[p].v
            else:
                if not time:
                    raise ValueError("propagation time not given")
                x, y, z = self.particles[p].r0
                vx, vy, vz = self.particles[p].v0

        else:
            if (r is None) or (v is None):
                raise ValueError("Missing vector")
            self._HandleVectors(r, v)
            x, y, z = r
            vx, vy, vz = v
        default = dict(method='DOP853', atol=1e-11, rtol=1e-11)
        default.update(kwargs)
        t_eval = np.linspace(0, time, int(N))

        solution = solve_ivp(self._EoM, y0=np.array([x, y, z, vx, vy, vz]), t_span=[
                             0, time], t_eval=t_eval, **default)

        if not (p is None):
            self.particles[p].r = solution.y[:3, -1]
            self.particles[p].v = solution.y[3:, -1]
            self.particles[p].time += time
        return solution.y.T, solution.t

    def _FBarycenter(self, x, y, z):
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

    def _FSecondary(self, x, y, z):

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

    def _STMPropagator(self, t, Y):

        # Last 6 components
        x, y, z, vx, vy, vz = Y[36:]
        # Give matrix shape to the first 36 components
        STM = Y[:36].reshape((6, 6))
        # Propagate initial position

        dydt = self._EoM(t, [x, y, z, vx, vy, vz])
        if self._refsystem == 'barycenter':
            F = self._FBarycenter(x, y, z)
        else:
            F = self._FSecondary(x, y, z)

        # Matrix Product Between F and M
        dSTMdt = np.array(np.matmul(F, STM)).reshape((36))

        return np.concatenate([dSTMdt, dydt])

    def getSTM(self, time, r=None, v=None, p=None, from_current=False, flag=None, **kwargs):

        if p:
            if from_current:
                x, y, z = self.particles[p].r
                vx, vy, vz = self.particles[p].v
            else:
                x, y, z = self.particles[p].r0
                vx, vy, vz = self.particles[p].v0

        else:
            if (not r) or (not v):
                raise ValueError("Missing vector")
            self._HandleVectors(r, v)
            x, y, z = r
            vx, vy, vz = v

        initialstate = np.zeros(42)
        initialstate[:36] = np.identity(6).flatten()
        initialstate[36:42] = x, y, z, vx, vy, vz

        default = dict(n=10000, method='DOP853', atol=1e-11, rtol=1e-11)
        default.update(kwargs)
        t_eval = np.linspace(0, time, default['n'])

        solution = solve_ivp(self.STMPropagator, y0=initialstate,
                             t_span=[0, time], t_eval=t_eval, events=(flag), **default)
        Y = solution.y[36:, :]
        STM = solution.y[:36, :].reshape((6, 6, solution.t.shape[0]))

        if flag:
            event = np.argmin(np.abs(solution.t_events[0]-time/2))
            HalfState = solution.y_events[0][event][36:]
            HalfSTM = solution.y_events[0][event][:36].reshape((6, 6))
            HalfTime = solution.t_events[0][event]

            return HalfState, HalfSTM, HalfTime

        finalSTM = STM[:, :, -1]

        return finalSTM, solution.t

    def _crossingFlag(self, t, Y):
        return Y[36:][1]

    def diffcorrector(self, time=None, r=None, v=None, p=None, from_current=False):

        if p:
            if not time:
                if self.particles[p].period == None:
                    raise ValueError(
                        'Particle selected but not time given or the particle does not have period estimated')
                time = self.particles[p].period
            if from_current:
                x, y, z = self.particles[p].r
                vx, vy, vz = self.particles[p].v
            else:
                x, y, z = self.particles[p].r0
                vx, vy, vz = self.particles[p].v0

        else:
            if not time:
                raise ValueError(
                    "If a particle is not selected, a time (estimated period) should be pass")
            if (not r) or (not v):
                raise ValueError("Missing vector")
            self._HandleVectors(r, v)
            x, y, z = r
            vx, vy, vz = v
        corrected = False
        counter = 0
        r = np.array([x, y, z])
        v = np.array([vx, vy, vz])
        Xmid, STM, Halftime = self.STM(time=time,
                                       r=r, v=v, flag=self.crossingFlag)
        while (abs(Xmid[1]) > 1e-10 or abs(Xmid[3]) > 1e-10) and counter < 10:
            Xmid, STM, Halftime = self.STM(
                r=r, v=v, flag=self.crossingFlag)
            ax = self._EoM(Halftime, Xmid)[3]
            deltavy = -Xmid[3]/(STM[3, 4]-ax*STM[1, 4]/Xmid[4])
            v[1] += deltavy
            counter += 1
            corrected = (abs(Xmid[1]) > 1e-10 or abs(Xmid[3]) > 1e-10)
        if corrected:
            if p:
                self.particles[p].r = r
                self.particles[p].v = v
            return np.array([*r, *v])
        if not corrected:
            raise Warning("Orbit not corrected")

        Xmid, STM, Halftime = self.STM(time=time,
                                       r=r, v=v, flag=self.crossingFlag)
        while (abs(Xmid[1]) > 1e-10 or abs(Xmid[3]) > 1e-10) and counter < 10:
            Xmid, STM, Halftime = self.STM(
                r=r, v=v, flag=self.crossingFlag)
            ax = self._EoM(Halftime, Xmid)[3]
            deltavy = -Xmid[3]/(STM[3, 4]-ax*STM[1, 4]/Xmid[4])
            v[1] += deltavy
            counter += 1
            corrected = (abs(Xmid[1]) > 1e-10 or abs(Xmid[3]) > 1e-10)
        if corrected:
            if p:
                self.particles[p].r = r
                self.particles[p].v = v
            return np.array([*r, *v])
        if not corrected:
            raise Warning("Orbit not corrected")

    def toInertialFrame(self, time=0, r=None, v=None, p=None, from_current=False):
        if not (p is None):
            if from_current:
                x, y, z = self.particles[p].r
                vx, vy, vz = self.particles[p].v
            else:
                x, y, z = self.particles[p].r0
                vx, vy, vz = self.particles[p].v0

        else:
            if (r is None) or (v is None):
                raise ValueError("Missing vector")
            self._HandleVectors(r, v)
            x, y, z = r
            vx, vy, vz = v
        X = np.array([x, y, z, vx, vy, vz])
        XInertial = np.zeros(X.shape)

        r1 = np.zeros_like((X[:3]))
        r2 = np.zeros_like((X[:3]))
        R = self._rotationMatrix(-time)
        r_i = np.array(X[:3])
        v_i = np.array(X[3:])
        XInertial[:3] = np.matmul(R, r_i).T
        XInertial[3:] = np.matmul(R, (v_i + np.cross([0, 0, 1], r_i))).T
        r1 = -self.mu * \
            np.array([np.cos(time), np.sin(time), 0])
        r2 = (1-self.mu) * \
            np.array([np.cos(time), np.sin(time), 0])
        return r1, r2, XInertial

    def toRotatingFrame(self, r, v, time=0):
        if (r is None) or (v is None):
            raise ValueError("Missing vector")
        self._HandleVectors(r, v)
        x, y, z = r
        vx, vy, vz = v
        X = np.array([x, y, z, vx, vy, vz])
        XRotational = np.zeros(X.shape)

        R = self._rotationMatrix(time)
        r_i = np.array(X[:3])
        v_i = np.array(X[3:])
        XRotational[:3] = np.matmul(R, r_i).T
        XRotational[3:] = np.matmul(R, (v_i - np.cross([0, 0, 1], r_i))).T

        return XRotational

    def _rotationMatrix(self, angle):
        return np.array(
            [[np.cos(angle), -np.sin(angle), 0],
             [np.sin(angle),  np.cos(angle), 0],
             [0,              0, 1]])

    def getUnits(self, M=None, m=None, L=None, period=None):
        G = 6.67430e-11
        UL = L
        if M:
            m = self.mu*M/(1-self.mu)
        if m:
            M = (1-self.mu)*m/self.mu
        UM = M + m
        UT = np.sqrt(UL**3/((UM)*(G)))
        UV = UL/UT
        return {"UL": UL, "UM": UM, "UT": UT, "UV": UV}


if __name__ == "__main__":

    sys = System(mu=0.02)
    r = np.array([1.1, 0, 0])
    v = np.array([-0.1, 1.3, 0])
    r_rotational = sys.toRotatingFrame(r=r, v=v)
    print(r_rotational)
