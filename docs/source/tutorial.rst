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

In the following sections, examples are provided with each integration method, while each example is well developed for separate running usage.

1. Trapezoidal integration  
2. Simpson's integration  
3. Boole's integration  
4. Gaussian quadrature  
5. Monte Carlo integration  
6. Importance-sampling Monte Carlo integration  

Generally, all methods share almost same interface, with minor difference in some cases.

Trapezoidal integration
--------
 
Trapezoidal integration is based on linear interpolation, which divides the integration interval into small trapezoids and approximates the definite integral by summing their areas. It is suitable for continuous functions and simple to implement, but provides low accuracy. For example, for 1D integration of :math:`f(x)`, we have the trapezoidal integration :math:`I` given by

.. math::

   I = \frac{\Delta x}{2} \left( f(x_0) + 2\sum_{i=1}^{n-1} f(x_i) + f(x_n) \right)

where :math:`x_0, x_1,...,x_n` are equally spaced points. This method is determininstic, suitable for linear functions, and accepts arbitrary number of points (odd or even) per dimension. The truncation eeror for this method is :math:`\mathcal{O}(\Delta x^2)`.

In this section, we provide 2 examples on utilizing trapezoidal integration method with **cupyint**. The trapezoidal integration function is ``cupyint.trapz_integrate(function, params, bounds, num_points, boundaries)``.

Our first example is to integrate :math:`f(x)=\mathrm{sin}(x)` over :math:`(0,1)`. We need to define 6 quantities as indicated by ``cupyint.trapz_integrate(function, params, bounds, num_points, boundaries)``, listed as: data precision format, integrand, parameters, integral bounds, number of sampling points, and boundary function before calculating the integral value.  

* Data precision format: this sets whether data is float32 or float64. The former uses less memory but provides less accuracy. Here we will use float32.  
* Integrand: function to be integrated. Here it is :math:`f(x)=\mathrm{sin}(x)`.  
* Parameters: Parameters go with the integrand but we don't have parameters in this case, so we input ``None`` in the ``params`` position.   
* Integral bounds: This is the integral limitation. In this example we set it as :math:`(0,1)`.  
* Number of sampling points: This defines number of spaced points we have in each dimension. In this example we set it as :math:`20`.  
* Boundary function: Other than the usual hyper-cubic integral limitations, we might meet cases in which integral limitations are functions of variables, i.e. :math:`x_1^2+x_2^2+x_3^2<1`. In the case here, we don't have a special boundary and we can input ``None`` in its position instead.  

.. code-block:: python

  import cupy as cp #required package for cupyint
  import cupyint

  data_type=cp.float32
  cupyint.set_backend(data_type) #this sets single precision data type in the backend

  def function (x):
      return cp.sin(x)

  bound = [[0, 1]] # This sets integral limitation as (0,1).
  num_point = [20] # This sets number of sampling points per dimension.
  integral_value = cupyint.trapz_integrate(function, None, bound, num_point, None) #We use trapz_integrate function

  analytical_value = cp.cos(0)-cp.cos(1) # absolute value of this integral
  relative_error = cp.abs(integral_value-analytical_value)/analytical_value # relative error

  print(f"integral value: {integral_value.item():.10f}") # Convert to Python float
  print(f"analytical value: {analytical_value.item():.10f}") 
  print(f"relative error: {relative_error.item():.10%}")

The output of the program is:

.. code-block:: python  

  integral value: 0.4595915675
  analytical value: 0.4596976941
  relative error: 0.0230861753%

To estimate the error in this case, we compare the integral value with the analytical one, obataining a relative error of ~0.02% with 20 segments in the integral domain. In general case, to estimate the error, we encourage users to refine the grids and analyze the convergence.


Our second example is a more complicated one, as we will try to integrate :math:`f(x_1,x_2,x_3)=a_1\cdot e^{-a_2(x_1^2+x_2^2+x_3^2)}+a_3\cdot\mathrm{sin}(x1)\cdot\mathrm{sin}(x2)\cdot\mathrm{sin}(x3)`, over the domain :math:`x_1\in (0,1)`, :math:`x_2\in (0,1)`, :math:`x_3\in (0,1)`, :math:`x_1^2+x_2^2+x_3^2>0.2`, and :math:`x_1^2+x_2^2+x_3^2<0.8`. For the parameters, we will have multiple sets of :math:`a_1`, :math:`a_2`, and :math:`a_3`. Details can be found in the code below.

.. code-block:: python  

  import cupy as cp #required package for cupyint
  import cupyint
  
  data_type=cp.float32
  cupyint.set_backend(data_type) #this sets single precision data type in the backend
  
  def function(x1, x2, x3, params): # this is the standard way to define an integrand with parameters
      a1 = params[0]
      a2 = params[1]
      a3 = params[2]
      return a1 * cp.exp(-a2 * (x1**2 + x2**2 + x3**2)) + a3 * cp.sin(x1) * cp.cos(x2) * cp.exp(x3)

  # This sets the parameter set, which is a 2d array in all cases. In this case, we have 1e4 parameter sets
  a1_values = cp.linspace(1.0, 10.0, 10000, dtype=data_type)
  a2_values = cp.linspace(2.0, 20.0, 10000, dtype=data_type)
  a3_values = cp.linspace(0.5, 5, 10000, dtype=data_type)
  param_values = cp.stack((a1_values, a2_values, a3_values), axis=1) 

  bound = [[0, 1], [0, 1], [0, 1]] # This sets integral limitation as (0,1),(0,1), and (0,1) for x1, x2, and x3, respectively.
  num_point = [20, 20, 20] # This sets number of sampling points per dimension.
  
  def boundary(x1, x2, x3):
      condition1 = x1**2 + x2**2 + x3**2 > 0.2
      condition2 = x1**2 + x2**2 + x3**2 < 0.8
      return condition1 & condition2
  
  integral_value = cupyint.trapz_integrate(function, param_values, bound, num_point, boundary) #We use trapz_integrate function
  
  print(f"integral value: {integral_value.get()}") # Output integral value
  print(f"length of integral value: {integral_value.size}") # Output length of the integral value

The output of the program is 

.. code-block:: python  

  integral value: [0.19233355 0.19240522 0.1924768  ... 0.73139507 0.7314593  0.7315235 ]
  length of integral value: 10000

Actually, **cupyint** is capable of handling multiple paramaters, and can automatically vectorize the integrand to perform faster calculation. The output ``integral_value`` should have the same length of the input ``param`` length, corresponding to various parameter sets.

