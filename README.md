# DiscreteExteriorCalculusForPDEs
Python implementation for a PDE solver based on discrete exterior calculus.

## 📖 Overview
This repository provides a Python-based implementation for solving Partial Differential Equations (PDEs) using the principles of Discrete Exterior Calculus (DEC). DEC is a numerical framework particularly suited for problems involving differential forms, allowing for consistent and accurate simulations on discrete meshes.

## ⚙️ Examples

### Gray-Scott Equations
The Gray-Scott model describes a reaction-diffusion system. The equations are:

$$\frac{du}{dt} = D_u\Delta u - uv^2 + F(1-u)$$  
$$\frac{dv}{dt} = D_v\Delta v + uv^2 + (F+k)v$$  

Simulation results:  
<img src="https://github.com/pcestola/DiscreteExteriorCalculusForPDEs/blob/main/GIFs/gs.gif" width="200" height="200" />
<img src="https://github.com/pcestola/DiscreteExteriorCalculusForPDEs/blob/main/GIFs/gs3.gif" width="200" height="200" />

### Wave Equation
The wave equation models the propagation of waves and is expressed as:

$$\frac{d^2u}{dt^2} = \nu^2\Delta u$$  

Simulation results:  
<img src="https://github.com/pcestola/DiscreteExteriorCalculusForPDEs/blob/main/GIFs/wave.gif" width="200" height="200" />

### Heat Equation
The heat equation models the variation in temperature in a given region over time:

$$\frac{du}{dt} = \alpha\Delta u$$  

Simulation results:  
<img src="https://github.com/pcestola/DiscreteExteriorCalculusForPDEs/blob/main/GIFs/Heat.gif" width="200" height="200" />
