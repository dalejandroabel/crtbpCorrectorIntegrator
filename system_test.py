from system import System
import pytest
import numpy as np


def test_validateMuRaisesException():
    with pytest.raises(Exception):
        System("a")


def test_ExamplesSystems():
    sys = System(mu="Earth")
    assert sys.mu == pytest.approx(3.04043e-06)


def test_validateAddParticles():
    sys = System(mu=0.5)
    r = np.array([1, 1, 1])
    v = [1, 1, 1]
    name = "first"
    sys.add(r=r, v=v, name=name)
    sys.add(r=[2, 3, 2], v=[4, 3, 2], name='second')
    assert len(sys.particles) == 2


def test_ExceptionRepeatedParticles():
    with pytest.raises(Exception):
        sys = System(mu=0.5)
        r = [1, 1, 1]
        v = [1, 1, 1]
        name = "first"
        sys.add(r=r, v=v, name=name)
        sys.add(r=[2, 3, 2], v=[4, 3, 2], name='first')


def test_DeleteParticle():
    sys = System(mu=0.5)
    r = [1, 1, 1]
    v = [1, 1, 1]
    sys.add(r=r, v=v)
    sys.add(r=[2, 3, 2], v=[4, 3, 2], name='second')
    sys.delete(p=0)
    sys.delete(name="second")
    assert len(sys.particles) == 0


def test_DeleteAllParticles():
    sys = System(mu=0.5)
    for i in range(5):
        sys.add(r=[i, i, i], v=[i, i, i])

    sys.delete()
    assert len(sys.particles) == 0


def test_ModifyRefsystem():
    sys = System(mu=0.5)
    sys.refsystem = 'secondary'
    assert sys.refsystem == 'secondary'


def test_ExceptionModifySystem():
    sys = System(mu=0.5)
    with pytest.raises(Exception):
        sys.refsystem = 'Secondary'


def test_LagrangePoints():
    sys = System(mu=0.02)
    LP = sys.getLagrangePoints()
    AreEqual = True
    if not (0.8035 <= LP['L1'][0] <= 0.8037):
        AreEqual = False
    if not (1.18018 <= LP['L2'][0] <= 1.1802):
        AreEqual = False
    if not (-1.009 <= LP['L3'][0] <= -1.007):
        AreEqual = False
    if not (0.47 <= LP['L4'][0] <= 0.49) or not (0.865 <= LP['L4'][1] <= 0.867):
        AreEqual = False
    if not (0.47 <= LP['L5'][0] <= 0.49) or not (-0.867 <= LP['L5'][1] <= -0.865):
        AreEqual = False

    assert AreEqual


def test_GetJacobiConstantfromP():
    sys = System(mu=0.02)
    r = np.array([1.1, 0, 0])
    v = np.array([-0.1, 0.2, 0])
    sys = System(mu=0.02)
    sys.add(r=r, v=v)
    JC = sys.getJacobiConstant(p=0)
    assert (3.2432 < JC < 3.244)


def test_GetJacobiConstantfromVector():
    sys = System(mu=0.02)
    r = np.array([1.1, 0, 0])
    v = np.array([-0.1, 0.2, 0])
    sys = System(mu=0.02)
    JC = sys.getJacobiConstant(r=r, v=v)
    assert (3.2432 < JC < 3.244)


def test_PropagateSingleTimefromP():

    sys = System(mu=0.02)
    r = np.array([1.1, 0, 0])
    v = np.array([-0.1, 0.2, 0])
    N = 2
    sys = System(mu=0.02)
    sys.add(r=r, v=v)
    y, t = sys.propagate(time=0.5, p=0, N=N)

    assert (t[1] == pytest.approx(0.5)) and (y.shape == (N, 6))


def test_PropagateSingleTimefromVectors():
    sys = System(mu=0.02)
    r = np.array([1.1, 0, 0])
    v = np.array([-0.1, 0.2, 0])
    N = 2
    sys = System(mu=0.02)
    y, t = sys.propagate(time=0.5, r=r, v=v, N=N)
    assert (t[1] == pytest.approx(0.5)) and (y.shape == (N, 6))


def test_PropagateTimesFromCurrent():
    sys = System(mu=0.02)
    r = np.array([1.1, 0, 0])
    v = np.array([-0.1, 0.2, 0])
    N = 2
    sys.add(r=r, v=v)
    sys.propagate(time=0.5, p=0, N=N, from_current=True)
    sys.propagate(time=0.5, p=0, N=N, from_current=True)
    assert (sys.particles[0].time == 1)


def test_GetUnits():
    mu_Moon = 0.0121437
    M_Earth = 5.9722e24
    M_moon = 0.07346e24
    distance_EarthMoon = 384400000
    sys = System(mu_Moon)
    Units = sys.getUnits(L=distance_EarthMoon, M=M_Earth)

    assert Units['UM']-M_Earth == pytest.approx(M_moon, rel=1e18)


def test_toInertialFrameFromP():
    sys = System(mu=0.02)
    r = np.array([1.1, 0, 0])
    v = np.array([-0.1, 0.2, 0])
    sys.add(r=r, v=v)
    r1, r2, inertial_r = sys.toInertialFrame(p=0)
    equal = True
    if r1[0] != pytest.approx(-0.02) or r2[0] != pytest.approx(0.98):
        equal = False
    if inertial_r[0] != pytest.approx(1.1) or inertial_r[3] != pytest.approx(-0.1) or inertial_r[4] != pytest.approx(1.3):
        equal = False
    assert equal


def test_toInertialFrameFromVector():
    sys = System(mu=0.02)
    r = np.array([1.1, 0, 0])
    v = np.array([-0.1, 0.2, 0])
    r1, r2, inertial_r = sys.toInertialFrame(r=r, v=v)
    equal = True
    if r1[0] != pytest.approx(-0.02) or r2[0] != pytest.approx(0.98):
        equal = False
    if inertial_r[0] != pytest.approx(1.1) or inertial_r[3] != pytest.approx(-0.1) or inertial_r[4] != pytest.approx(1.3):
        equal = False
    assert equal


def test_toRotatingFrameFromVector():
    sys = System(mu=0.02)
    r = np.array([1.1, 0, 0])
    v = np.array([-0.1, 1.3, 0])
    r_rotational = sys.toRotatingFrame(r=r, v=v)
    print(r_rotational)
    equal = True
    if (r_rotational[0] != pytest.approx(1.1)) or (r_rotational[3] != pytest.approx(-0.1)) or (r_rotational[4] != pytest.approx(0.2)):
        equal = False
    assert equal
