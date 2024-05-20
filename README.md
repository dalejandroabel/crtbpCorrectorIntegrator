# PyCRTBP

## CRTBP Orbits and Utilities

[![version](https://img.shields.io/pypi/v/pycrtbp?color=blue)](https://pypi.org/project/pycrtbp/)
[![downloads](https://img.shields.io/pypi/dw/pycrtbp)](https://pypi.org/project/pycrtbp/)
[![license](https://img.shields.io/pypi/l/pycrtbp)](https://pypi.org/project/pycrtbp/)
[![implementation](https://img.shields.io/pypi/implementation/pycrtbp)](https://pypi.org/project/pycrtbp/)
[![pythonver](https://img.shields.io/pypi/pyversions/pycrtbp)](https://pypi.org/project/pycrtbp/)

The package `pycrtbp` allows you to integrate orbits in the CRTBP problem.

For the science behind the model please refer to the following papers:

> Acosta, D.A., Zuluaga, J.I. & Restrepo, R. (2024), **A web interface
  for a collection of CRTBP periodic orbits**, in preparation.

<!-- This is the format for including a reference:
[Astronomy and Computing 40 (2022)
  100623](https://www.sciencedirect.com/science/article/pii/S2213133722000476),
  [arXiv:2207.08636](https://arxiv.org/abs/2207.08636).
-->

## Download and install

`pycrtbp` is available at `PyPI`, https://pypi.org/project/pycrtbp/.
To install use:

```
   pip install -U pycrtbp
```

If you prefer you may download the package directly from the  
[sources](https://pypi.org/project/pycrtbp/#files) cloning the repo:

```
   git clone https://github.com/seap-udea/crtbpCorrectorIntegrator
```

## Quick start

Import package:

```python
import pycrtbp as py3
```

Create a system:

```python
sys = p3.System(mu=0.3)
```

Add a particle:

```python
sys.add(r=[1,0,0],v=[0,0.5,0])
```

Propagate solution:

```python
solution,ts = sys.propagate(p=0,time=10,N=1000)
```

You may plot the trajectory:

```python
plt.plot(solucion[:,0],solucion[:,1])
plt.axis('equal')
```

Or calculate quantities of interest:

- Jacobi constant:
  ```python
  sys.getJacobiConstant(p=0)
  ```

- Position of the Lagrange equilibrium points:
  ```python
  sys.getLagrangePoints()
  ```

- Value of the state transition matrix:
  ```python
  STM, ts = sys.getSTM(p=0,time=0.5)
  ```

The following code plot a Lyapunov periodic orbit around the L1 point in the Earth-Moon system:

```python
import plotly.graph_objects as go

# Data taken from https://ssd.jpl.nasa.gov/tools/periodic_orbits.html
sys = p3.System(mu=1.215058560962404E-2)
sys.add(r=[4.3840151982551506E-1,8.1854815386736461E-23,-2.4989800229106000E-25],
        v=[2.1780773271699781E-13,1.3613843962742438E+0,-8.2105956297402337E-25])
sys.refsystem = 'barycenter'
solucion,ts = sys.propagate(time=10,p=0,N=1000)

fig = go.Figure(data=go.Scatter3d(
    x=solucion[:, 0],
    y=solucion[:, 1],
    z=solucion[:, 2],
    mode='lines',
    line=dict(color='blue', width=2),
))

fig.add_trace(go.Scatter3d(
    x=[-sys.mu, 1-sys.mu],
    y=[0, 0],
    z=[0, 0],
    mode='markers',
    marker=dict(color='red', size=[5, 1]),
))

rang = 1.5
fig.update_layout(scene=dict(
    xaxis=dict(title='X',range=[-rang,rang]),
    yaxis=dict(title='Y',range=[-rang,rang]),
    zaxis=dict(title='Z',range=[-rang,rang]),
))

fig.show()
```

## What's new

For a detailed list of the newest features introduced in the latest
releases pleas check [What's
new](https://github.com/seap-udea/crtbpCorrectorIntegrator/raw/main/WHATSNEW.md).

------------

This package has been designed and written originally by Diego A.
Acosta & Jorge I. Zuluaga with the scientific and techincal advise of
Ricardo Restrepo (C) 2024
