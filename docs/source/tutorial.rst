Tutorial
=====

Overview
--------
**cupyint** is a Python package tailored to perform numerical integration based on `CuPy <https://cupy.dev/>`_, with highlights in

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

Generally, all methods share same interface, baring minor difference.

.. _trapez integration:
Trapezoidal integration
--------
 
Trapezoidal integration is based on linear interpolation, which divides the integration interval into small trapezoids and approximates the definite integral by summing their areas. It is suitable for continuous functions and simple to implement, but provides low accuracy. For example, for 1D integration of :math:`f(x)`, we have the trapezoidal integration :math:`I` given by

.. math::

   I = \frac{\Delta x}{2} \left( f(x_0) + 2\sum_{i=1}^{n-1} f(x_i) + f(x_n) \right)

where :math:`x_0, x_1,...,x_n` are equally spaced points. This method is determininstic, suitable for linear functions, and accepts arbitrary number of points (odd or even) per dimension. The truncation eeror for this method is :math:`\mathcal{O}(\Delta x^2)`.

In this section, we provide 2 examples on utilizing trapezoidal integration method with **cupyint**. The trapezoidal integration function is ``cupyint.trapz_integrate(function, params, bounds, num_points, boundaries)``.

Our first example is to integrate :math:`f(x)=\mathrm{sin}(x)` over :math:`(0,1)`. We need to define 6 quantities as indicated by , listed as: data precision format, integrand, parameters, integral bounds, number of sampling points, and boundary function before calculating the integral value.  

* Data precision format: this sets whether data is in float32 or float64. The former uses less memory but provides less accuracy. Here we will use float32.  
* Integrand: function to be integrated. Here it is :math:`f(x)=\mathrm{sin}(x)`.  
* Parameters: Parameters are with the integrand but we don't have parameters in this case. However, this is a general interface in **cupyint**, we can set this as :math:`1` in the integrand.  
* Integral bounds: This is the integral limitation. In this example we set it as :math:`(0,1)`.  
* Number of sampling points: This defines number of spaced points we have in each dimension. In this example we set it as :math:`20`.  
* Boundary function: Other than the usual hyper-cubic integral limitations, we might meet cases in which integral limitations are functions of variables, i.e. :math:`x_1^2+x_2^2+x_3^2<1`. In the case here, we don't have a special boundary and we can input "None" instead.  

.. code-block:: python

  import cupy as cp #required package for cupyint
  import cupyint

  data_type = cp.float32
  cupyint.set_backend(data_type) #this sets single precision data type in the backend

  def function(x, params):
    # "params" is necessary in defining the integrand.
    # However, this depends on how many parameters we have for the integrand. For the case here, we will set "params" to 1 later.
      return cp.sin(x) * params[0]

  params = cp.asarray([[1]], dtype=data_type) # We only accept 2D parameters in cp.array form. Please pay special attention.
  bound = [[0, 1]] # This sets integral limitation as (0, 1).
  num_point = [20] # This sets number of sampling points per dimension. 
  integral_value = cupyint.trapz_integrate(function, params, bound, num_point, boundaries=None) #We use trapz_integrate function
  print(f"integral_value: {integral_value.item():.10f}") # Convert to Python float 

The output of the program is:

.. code-block:: python  

  integral_value: 0.4595915675




