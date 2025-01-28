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

In the following sections, examples are provided with each integration method:

1. Trapezoidal integration  
2. Simpson's integration  
3. Boole's integration  
4. Gaussian quadrature  
5. Monte Carlo integration  
6. Importance-sampling Monte Carlo integration  

Trapezoidal integration
--------
 
Trapezoidal integration is based on linear interpolation, which divides the integration interval into small trapezoids and approximates the definite integral by summing their areas. It is suitable for continuous functions and simple to implement, but provides low accuracy. For example, for 1D integration of :math:`f(x)`, we have the trapezoidal integration :math:`I` given by

.. math::

   I = \frac{\Delta x}{2} \left( f(x_0) + 2\sum_{i=1}^{n-1} f(x_i) + f(x_n) \right)

where :math:`x_0, x_1,...,x_n` are equally spaced point. This method is determininstic, suitable for linear functions, and accepts arbitrary points (odd or even) per dimension. The truncation eeror for the method is :math:`\mathcal{O}(\Delta x^2)`.


