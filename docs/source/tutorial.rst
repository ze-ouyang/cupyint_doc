Tutorial
=====

Overview
--------
**cupyint** is a Python package tailored to perform numerical integration based on `CuPy <https://cupy.dev/>`_, with highlighs in

* high dimensional integration  
* fast & parallel computation on GPU  
* vectorized integration incorporating with broadcasting mechanism  
* complicated integration boundaries  
* tunable sampling points per dimension  
* multiple integration methods, including deterministic & stochastic ones  
* user-friendly interface  

In the following sections, theories and examples are provided with each integration method.

1. Trapezoidal integration  
2. Simpson's integration  
3. Boole's integration  
4. Gaussian quadrature  
5. Monte Carlo integration  
6. Importance-sampling Monte Carlo integration  

Trapezoidal integration
--------

* Theory  
Trapezoidal integration is based on linear interpolation, which divides the integration interval into small trapezoids and approximates the definite integral by summing their areas. It is suitable for continuous functions and simple to implement, but provides low accuracy. For example, for 1D integration of :math:`f(x)`, we have the trapezoidal integration :math:`I` given by

.. math::

   E = mc^2

