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
* user-friendly interface  (So for users **not familiar** with CuPy, this package is still simple to use)


In the following sections, examples are provided with each integration method, while each example is well developed for separate running usage.

1. Trapezoidal integration  
2. Simpson's integration  
3. Boole's integration  
4. Gaussian quadrature  
5. Monte Carlo integration    

Generally, all methods share almost same interface, with minor difference in Method 5.

Trapezoidal integration
--------
 
Trapezoidal integration is based on linear interpolation, which divides the integration interval into small trapezoids and approximates the definite integral by summing their areas. It is suitable for continuous functions and simple to implement, but provides low accuracy. For example, for 1D integration of :math:`f(x)`, we have the trapezoidal integration :math:`I` given by

.. math::

   I = \frac{\Delta x}{2} \left( f(x_0) + 2\sum_{i=1}^{n-1} f(x_i) + f(x_n) \right)

where :math:`x_0, x_1,...,x_n` are equally spaced points, and :math:`\Delta x=x_i-x_{i-1}`. This method is determininstic, suitable for linear functions, and accepts arbitrary number of points (odd or even) per dimension. The truncation error for this method is :math:`\mathcal{O}(\Delta x^2)`.

In this section, we provide 2 examples on utilizing trapezoidal integration method with **cupyint**. The trapezoidal integration function is ``cupyint.trapz_integrate(function, params, bounds, num_points, boundaries)``.

Our first example is to integrate :math:`f(x)=\mathrm{sin}(x)` over :math:`(0,1)`. We need to define 6 quantities as indicated by ``cupyint.trapz_integrate(function, params, bounds, num_points, boundaries)``, listed as: data precision format, integrand, parameters, integral bounds, number of sampling points, and boundary function before calculating the integral value.  

* Data precision format: this sets whether data is float32 or float64. The former uses less memory but provides less accuracy. Here we will use float32.  
* Integrand: function to be integrated. Here it is :math:`f(x)=\mathrm{sin}(x)`.  
* Parameters: Parameters go with the integrand but we don't have parameters in this case, so we input ``None`` in the ``params`` position.   
* Integral bounds: This is the integral limitation. In this example we set it as :math:`(0,1)`.  
* Number of sampling points: This defines number of spaced points we have in each dimension. In this example we set it as :math:`20`.  
* Boundary function: Other than the usual hyper-cubic integral limitations, we might meet cases in which integral limitations are functions of variables, i.e. :math:`x_1^2+x_2^2+x_3^2<1`. In the case here, we don't have a special boundary and we can input ``None`` in its position instead. 

.. note::

    For users not familiar with CuPy, the only thing to bear in mind is to set all the variables in your code in the date type of ``cp.array``, then everying should work fine.

.. code-block:: python

  import cupy as cp # Required package for cupyint
  import cupyint

  data_type = cp.float32
  cupyint.set_backend(data_type) # This sets single precision data type in the backend

  def function (x):
      return cp.sin(x)

  bound = [[0, 1]] # This sets integral limitation as (0,1).
  num_point = [20] # This sets number of sampling points per dimension.
  integral_value = cupyint.trapz_integrate(function, None, bound, num_point, None) #We use trapz_integrate function

  analytical_value = cp.cos(0) - cp.cos(1) # absolute value of this integral
  relative_error = cp.abs(integral_value - analytical_value) / analytical_value # relative error

  print(f"integral value: {integral_value.item():.10f}") # Convert to Python float
  print(f"analytical value: {analytical_value.item():.10f}") 
  print(f"relative error: {relative_error.item():.10%}")

The output of the program is:

.. code-block:: none 

  integral value: 0.4595915675
  analytical value: 0.4596976941
  relative error: 0.0230861753%

To estimate the error in this case, we compare the integral value with the analytical one, obataining a relative error of ~0.02% with 20 segments in the integral domain. In general case, to estimate the error, we encourage users to refine the grids and analyze the convergence.


Our second example is a more complicated one, as we will try to integrate :math:`f(x_1,x_2,x_3)=a_1\cdot e^{-a_2(x_1^2+x_2^2+x_3^2)}+a_3\cdot\mathrm{sin}(x1)\cdot\mathrm{sin}(x2)\cdot\mathrm{sin}(x3)`, over the domain :math:`x_1\in (0,1)`, :math:`x_2\in (0,1)`, :math:`x_3\in (0,1)`, :math:`x_1^2+x_2^2+x_3^2>0.2`, and :math:`x_1^2+x_2^2+x_3^2<0.8`. For the parameters, we will have multiple sets of :math:`a_1`, :math:`a_2`, and :math:`a_3`. Details can be found in the code below.

.. code-block:: python  

  import cupy as cp #required package for cupyint
  import cupyint
  
  data_type = cp.float32
  cupyint.set_backend(data_type) #this sets single precision data type in the backend
  
  def function(x1, x2, x3, params): # this is the standard way to define an integrand with parameters
      a1 = params[0]
      a2 = params[1]
      a3 = params[2]
      return a1 * cp.exp(-a2 * (x1**2 + x2**2 + x3**2)) + a3 * cp.sin(x1) * cp.cos(x2) * cp.exp(x3)

  # This sets the parameter set, which is a 2d array in all cases. In this case, we have 1e4 parameter sets
  a1_values = cp.linspace(1.0, 10.0, 10000, dtype = data_type)
  a2_values = cp.linspace(2.0, 20.0, 10000, dtype = data_type)
  a3_values = cp.linspace(0.5, 5, 10000, dtype = data_type)
  param_values = cp.stack((a1_values, a2_values, a3_values), axis=1) 

  bound = [[0, 1], [0, 1], [0, 1]] # This sets integral limitation as (0,1),(0,1), and (0,1) for x1, x2, and x3, respectively.
  num_point = [20, 20, 20] # This sets number of sampling points per dimension.
  
  def boundary(x1, x2, x3):
      condition1 = x1**2 + x2**2 + x3**2 > 0.2
      condition2 = x1**2 + x2**2 + x3**2 < 0.8
      return condition1 & condition2
  
  integral_value = cupyint.trapz_integrate(function, param_values, bound, num_point, boundary) # We use trapz_integrate function
  
  print(f"integral value: {integral_value.get()}") # Output integral value
  print(f"length of integral value: {integral_value.size}") # Output length of the integral value

  # To estimate error, we double the grids in all three dimension, and output the relative error.
  num_point = [40, 40, 40] # This sets number of sampling points per dimension, which are doubled
  integral_value2 = cupyint.trapz_integrate(function, param_values, bound, num_point, boundary) #We use trapz_integrate function
  relative_error = cp.abs(integral_value - integral_value2) / integral_value # relative error

  print(f"integral value with denser grids: {integral_value2.get()}") 
  print(f"relative error: {relative_error.get()}")

.. note::

  There are other ways to define the ``params_values`` in the above code, depending on user's habit. The core rule is that the ``params_values`` should be a 2D ``cp.array``, like [ [1, 2, 0.5], [1.00090009, 2.00180018, 0.50045005], ..., [10, 20, 5] ] in our case here.

Actually, **cupyint** is capable of handling multiple paramaters, and can automatically vectorize the integrand to avoid explicit for-loop, thus to facilitate faster calculation. The output ``integral_value`` should have the same length of the input ``param`` length, corresponding to various parameter sets. To analyze the error, we doubled the grids on all three dimensions, and obtained a relative error of ~0.6%. The output of the program is 

.. code-block:: none  

  integral value: [0.19233355 0.19240522 0.1924768  ... 0.73139507 0.7314593  0.7315235 ]
  length of integral value: 10000
  integral value with denser grids: [0.19352302 0.193595   0.1936669  ... 0.7385989  0.7386638  0.7387286 ]
  relative error: [0.00618441 0.00618374 0.00618314 ... 0.00984942 0.00984945 0.0098494 ]


Simpson's integration
--------

Simpson's integration is based on quadratic interpolation. It divides the integration interval into an even number of subintervals, fits parabolas to the function, and approximates the definite integral by summing the areas under the parabolas. It offers higher accuracy than the trapezoidal integration at the cost of slightly higher computation complexity. For example, for 1D integration of :math:`f(x)`, we have the Simpson's integration :math:`I` given by 

.. math::

   I = \frac{\Delta x}{3} \left( f(x_0) + 4\sum_{i=1,3,5,...}^{n-1} f(x_i) + 2\sum_{i=2,4,6,...}^{n-2} f(x_i) + f(x_n) \right)

where :math:`x_0, x_1,...,x_n` are equally spaced points, and :math:`\Delta x=x_i-x_{i-1}`. This method is determininstic, suitable for smooth functions, and accepts odd number of points per dimension. The truncation error for this method is :math:`\mathcal{O}(\Delta x^4)`, about 2 orders of magnitude higher than that of trapezoidal integration. 

In this section, we still provide 2 examples, which calculate the same integral as we did in the Trapzoidal integration section, but codes are different (obviously).

The code for the first example is given below

.. code-block:: python  

  import cupy as cp # Required package for cupyint
  import cupyint
  
  data_type = cp.float32
  cupyint.set_backend(data_type) # This sets single precision data type in the backend
  
  def function (x):
      return cp.sin(x)
  
  bound = [[0, 1]] # This sets integral limitation as (0,1).
  num_point = [21] # This sets number of sampling points per dimension.
  integral_value = cupyint.simpson_integrate(function, None, bound, num_point, None) #We use simpson_integrate function
  
  analytical_value = cp.cos(0) - cp.cos(1) # absolute value of this integral
  relative_error = cp.abs(integral_value - analytical_value) / analytical_value # relative error
  
  print(f"integral value: {integral_value.item():.10f}") # Convert to Python float
  print(f"analytical value: {analytical_value.item():.10f}")
  print(f"relative error: {relative_error.item():.10%}")

The output of the program is 

.. code-block:: none 

  integral value: 0.4596977234
  analytical value: 0.4596976941
  relative error: 0.0000063644%

In the output, we see a relative error of ~0.000006% with 21 segments in the integral domain. This manifests the aforementioned higher accuracy of this method.

The code for the second example is given below

.. code-block:: python  

  import cupy as cp #required package for cupyint
  import cupyint
  
  data_type = cp.float32
  cupyint.set_backend(data_type) #this sets single precision data type in the backend
  
  def function(x1, x2, x3, params): # this is the standard way to define an integrand with parameters
      a1 = params[0]
      a2 = params[1]
      a3 = params[2]
      return a1 * cp.exp(-a2 * (x1**2 + x2**2 + x3**2)) + a3 * cp.sin(x1) * cp.cos(x2) * cp.exp(x3)
  
  # This sets the parameter set, which is a 2d array in all cases. In this case, we have 1e4 parameter sets
  a1_values = cp.linspace(1.0, 10.0, 10000, dtype = data_type)
  a2_values = cp.linspace(2.0, 20.0, 10000, dtype = data_type)
  a3_values = cp.linspace(0.5, 5, 10000, dtype = data_type)
  param_values = cp.stack((a1_values, a2_values, a3_values), axis=1)
  
  bound = [[0, 1], [0, 1], [0, 1]] # This sets integral limitation as (0,1),(0,1), and (0,1) for x1, x2, and x3, respectively.
  num_point = [21, 21, 21] # This sets number of sampling points per dimension.
  
  def boundary(x1, x2, x3):
      condition1 = x1**2 + x2**2 + x3**2 > 0.2
      condition2 = x1**2 + x2**2 + x3**2 < 0.8
      return condition1 & condition2
  
  integral_value = cupyint.simpson_integrate(function, param_values, bound, num_point, boundary) # We use simpson_integrate function
  
  print(f"integral value: {integral_value.get()}") # Output integral value
  print(f"length of integral value: {integral_value.size}") # Output length of the integral value
  
  # To estimate error, we double the grids in all three dimension, and output the relative error.
  num_point = [41, 41, 41] # This sets number of sampling points per dimension, which are doubled
  integral_value2 = cupyint.simpson_integrate(function, param_values, bound, num_point, boundary) #We use simpson_integrate function
  relative_error = cp.abs(integral_value - integral_value2) / integral_value # relative error
  
  print(f"integral value with denser grids: {integral_value2.get()}")
  print(f"relative error: {relative_error.get()}")

The output of this program is

.. code-block:: none 

  integral value: [0.19431727 0.1943896  0.19446182 ... 0.7404201  0.74048513 0.74055004]
  length of integral value: 10000
  integral value with denser grids: [0.19361119 0.19368313 0.19375499 ... 0.7396032  0.73966813 0.73973316]
  relative error: [0.00363363 0.00363427 0.00363483 ... 0.00110327 0.00110333 0.00110307]

Again, we see an improvement on the accuracy when doubling the grids.


Boole's integration
--------
Boole's integration is derived from Newton-Cotes formulas with fourth-order polynomial interpolation. It divides the integration interval into subintervals and computes the integral using a five-point interpolation formula. This method is suitable for smooth functions requiring higher precision. For example, for 1D integration of :math:`f(x)`, we have the Simpson's integration :math:`I` given by 

.. math::

  I = \frac{2\Delta x}{45} \left( 7f(x_i) + 32\sum_{i=1,5,9,...}^{4N-3} f(x_i) + 12\sum_{i=2,6,10,...}^{4N-2} f(x_i) + 32\sum_{i=3,7,11,...}^{4N-1} f(x_i) + 14\sum_{i=4,8,12,...}^{4N-4} f(x_i) + 7f(x_{n}) \right)


where :math:`x_0, x_1,...,x_n` are equally spaced points, :math:`\Delta x=x_i-x_{i-1}`, and :math:`N` is an integer. This method is determininstic, suitable for smooth functions, and accepts :math:`4N+1` number of points per dimension, where :math:`N` is an integer. The truncation error for this method is :math:`\mathcal{O}(\Delta x^6)`.

In this section, we still provide 2 examples, which calculate the same integral as we did in the Trapzoidal integration section, but codes are different (obviously).

The code for the first example is given below

.. code-block:: python  

  import cupy as cp # Required package for cupyint
  import cupyint
  
  data_type = cp.float32
  cupyint.set_backend(data_type) # This sets single precision data type in the backend
  
  def function (x):
      return cp.sin(x)
  
  bound = [[0, 1]] # This sets integral limitation as (0,1).
  num_point = [21] # This sets number of sampling points per dimension.
  integral_value = cupyint.booles_integrate(function, None, bound, num_point, None) #We use booles_integrate function
  
  analytical_value = cp.cos(0) - cp.cos(1) # absolute value of this integral
  relative_error = cp.abs(integral_value - analytical_value) / analytical_value # relative error
  
  print(f"integral value: {integral_value.item():.10f}") # Convert to Python float
  print(f"analytical value: {analytical_value.item():.10f}")
  print(f"relative error: {relative_error.item():.10%}")

The output of the program is 

.. code-block:: none 

  integral value: 0.4596976936
  analytical value: 0.4596976941
  relative error: 0.0000001187%

In the output, we see a relative error of ~0.00000011% with 21 segments in the integral domain. This manifests the aforementioned even higher accuracy of this method.

The code for the second example is given below

.. code-block:: python  

  import cupy as cp #required package for cupyint
  import cupyint
  
  data_type = cp.float32
  cupyint.set_backend(data_type) #this sets single precision data type in the backend
  
  def function(x1, x2, x3, params): # this is the standard way to define an integrand with parameters
      a1 = params[0]
      a2 = params[1]
      a3 = params[2]
      return a1 * cp.exp(-a2 * (x1**2 + x2**2 + x3**2)) + a3 * cp.sin(x1) * cp.cos(x2) * cp.exp(x3)
  
  # This sets the parameter set, which is a 2d array in all cases. In this case, we have 1e4 parameter sets
    a1_values = cp.linspace(1.0, 10.0, 10000, dtype = data_type)
    a2_values = cp.linspace(2.0, 20.0, 10000, dtype = data_type)
    a3_values = cp.linspace(0.5, 5, 10000, dtype = data_type)
    param_values = cp.stack((a1_values, a2_values, a3_values), axis=1)
    
    bound = [[0, 1], [0, 1], [0, 1]] # This sets integral limitation as (0,1),(0,1), and (0,1) for x1, x2, and x3, respectively.
    num_point = [21, 21, 21] # This sets number of sampling points per dimension.
    
    def boundary(x1, x2, x3):
        condition1 = x1**2 + x2**2 + x3**2 > 0.2
        condition2 = x1**2 + x2**2 + x3**2 < 0.8
        return condition1 & condition2
    
    integral_value = cupyint.booles_integrate(function, param_values, bound, num_point, boundary) # We use booles_integrate function
    
    print(f"integral value: {integral_value.get()}") # Output integral value
    print(f"length of integral value: {integral_value.size}") # Output length of the integral value
    
    # To estimate error, we double the grids in all three dimension, and output the relative error.
    num_point = [41, 41, 41] # This sets number of sampling points per dimension, which are doubled
    integral_value2 = cupyint.booles_integrate(function, param_values, bound, num_point, boundary) #We use booles_integrate function
    relative_error = cp.abs(integral_value - integral_value2) / integral_value # relative error
    
    print(f"integral value with denser grids: {integral_value2.get()}")
    print(f"relative error: {relative_error.get()}")

The output of this program is

.. code-block:: none 

  integral value: [0.19473471 0.19480716 0.19487953 ... 0.7423441  0.74240917 0.74247426]
  length of integral value: 10000
  integral value with denser grids: [0.19354594 0.19361784 0.19368966 ... 0.7395477  0.73961276 0.73967767]
  relative error: [0.00610456 0.00610512 0.00610568 ... 0.00376692 0.00376667 0.00376658]

Again, we see an improvement on the accuracy when doubling the grids.

Gaussian quadrature
--------

Gaussian Quadrature is an efficient numerical integration method that uses the roots of orthogonal polynomials (such as Legendre polynomials) as integration points. It approximates the integral by a weighted sum of function values at these points, achieving high accuracy with relatively few points, especially for smooth functions. In this method, we choose the Legendre polynomials as the orthogonal polynomials. The theory of this method is a little more complicated as of now. We start with the standard interval :math:`(-1,1)`, and obtained 

.. math::

  \int_{-1}^1 f(x) \mathrm d x \approx \sum_{i=1}^n w_i f(x_i)

where :math:`x_i` are the nodes, or roots of the :math:`n` th degree Legendre polynomial :math:`P_n(x)`, :math:`w_i` are the weights associated with each node, :math:`n` is the number of nodes, also known as the order of quadrature. Both :math:`x_i` and :math:`w_i` are precomputed values, we suggest that users ask `DeepSeek <https://www.deepseek.com/>`_ for details. As for the general integral :math:`(a,b)`, we have the transformation that 

.. math::

  I = \int_{a}^b f(x) \mathrm d x \approx  \frac{b-a}{2} \sum_{i=1}^n w_i f\left(\frac{b-a}{2}x_i+\frac{a+b}{2}\right)

Above is the case for 1D integral of function :math:`f(x)` over domain :math:`(a,b)`.

In this section, we still provide 2 examples, which calculate the same integral as we did in the Trapzoidal integration section, but codes are different (obviously).

The code for the first example is given below

.. code-block:: python 

  import cupy as cp # Required package for cupyint
  import cupyint
  
  data_type = cp.float32
  cupyint.set_backend(data_type) # This sets single precision data type in the backend
  
  def function (x):
      return cp.sin(x)
  
  bound = [[0, 1]] # This sets integral limitation as (0,1).
  num_point = [20] # This sets number of sampling points per dimension.
  integral_value = cupyint.gauss_integrate(function, None, bound, num_point, None) #We use gauss_integrate function
  
  analytical_value = cp.cos(0) - cp.cos(1) # absolute value of this integral
  relative_error = cp.abs(integral_value - analytical_value) / analytical_value # relative error
  
  print(f"integral value: {integral_value.item():.10f}") # Convert to Python float
  print(f"analytical value: {analytical_value.item():.10f}")
  print(f"relative error: {relative_error.item():.10%}")

The output of the program is 

.. code-block:: none 

  integral value: 0.4596976936
  analytical value: 0.4596976941
  relative error: 0.0000001187%

In the output, we see a relative error of ~0.00000011% with 20 nodes in the integral domain.  

The code for the second example is given below

.. code-block:: python  

  import cupy as cp #required package for cupyint
  import cupyint
  
  data_type = cp.float32
  cupyint.set_backend(data_type) #this sets single precision data type in the backend
  
  def function(x1, x2, x3, params): # this is the standard way to define an integrand with parameters
      a1 = params[0]
      a2 = params[1]
      a3 = params[2]
      return a1 * cp.exp(-a2 * (x1**2 + x2**2 + x3**2)) + a3 * cp.sin(x1) * cp.cos(x2) * cp.exp(x3)
  
  # This sets the parameter set, which is a 2d array in all cases. In this case, we have 1e4 parameter sets
  a1_values = cp.linspace(1.0, 10.0, 10000, dtype = data_type)
  a2_values = cp.linspace(2.0, 20.0, 10000, dtype = data_type)
  a3_values = cp.linspace(0.5, 5, 10000, dtype = data_type)
  param_values = cp.stack((a1_values, a2_values, a3_values), axis=1)
  
  bound = [[0, 1], [0, 1], [0, 1]] # This sets integral limitation as (0,1),(0,1), and (0,1) for x1, x2, and x3, respectively.
  num_point = [20, 20, 20] # This sets number of sampling points per dimension.
  
  def boundary(x1, x2, x3):
      condition1 = x1**2 + x2**2 + x3**2 > 0.2
      condition2 = x1**2 + x2**2 + x3**2 < 0.8
      return condition1 & condition2
  
  integral_value = cupyint.gauss_integrate(function, param_values, bound, num_point, boundary) # We use gauss_integrate function
  
  print(f"integral value: {integral_value.get()}") # Output integral value
  print(f"length of integral value: {integral_value.size}") # Output length of the integral value
  
  # To estimate error, we double the grids in all three dimension, and output the relative error.
  num_point = [40, 40, 40] # This sets number of sampling points per dimension, which are doubled
  integral_value2 = cupyint.gauss_integrate(function, param_values, bound, num_point, boundary) #We use gauss_integrate function
  relative_error = cp.abs(integral_value - integral_value2) / integral_value # relative error
  
  print(f"integral value with denser grids: {integral_value2.get()}")
  print(f"relative error: {relative_error.get()}")

The output of this program is

.. code-block:: none 

  integral value: [0.19423467 0.19430667 0.19437855 ... 0.7441452  0.7442106  0.74427605]
  length of integral value: 10000
  integral value with denser grids: [0.19395865 0.19403082 0.19410291 ... 0.7405079  0.7405729  0.7406379 ]
  relative error: [0.00142103 0.00141966 0.00141807 ... 0.00488791 0.00488796 0.00488817]

Again, we see an improvement on the accuracy when doubling the grids.



Monte Carlo integration
--------

Monte Carlo integration is based on random sampling. It estimates the integral by generating random samples within the integration domain and averaging the function values. This method is stochastic, and is particularly effective for high-dimensional integrals, with statistical errors decreasing as the sample size increases. For example, for 1D integration :math:`f(x)` over domain :math:`(a,b)`, we have the integration :math:`I` given by

.. math::

  I \approx \frac{b-a}{n}\sum_{i=1}^n f(x_i)

where :math:`x_i` are randomly generated points in the integration domain, :math:`n` is the number of random points generated. The error of this method scales as :math:`\displaystyle{\mathcal{O}\left(\frac{1}{\sqrt n} \right)}`. 

In this section, we still provide 2 examples, which calculate the same integral as we did in the Trapzoidal integration section, but codes are different (obviously).

.. note::

    As mentioned before, the interfaces of Monte Carlo integration and Importance-sampling Monte Carlo integration are slightly different from that of previous four methods. For Monte Carlo integration case here, the only difference is that ``num_points`` parameter should be an integer rather than a list as before. And with this manner, we sample the same number of points (which is indicated by ``num_points``) in each dimension.

The code for the first example is given below

.. code-block:: python 

  import cupy as cp # Required package for cupyint
  import cupyint
  
  data_type = cp.float32
  cupyint.set_backend(data_type) # This sets single precision data type in the backend
  
  def function (x):
      return cp.sin(x)
  
  bound = [[0, 1]] # This sets integral limitation as (0,1).
  num_point = 1000 # This sets number of sampling points per dimension.
  integral_value = cupyint.mc_integrate(function, None, bound, num_point, None) #We use mc_integrate function
  
  analytical_value = cp.cos(0) - cp.cos(1) # absolute value of this integral
  relative_error = cp.abs(integral_value - analytical_value) / analytical_value # relative error
  
  print(f"integral value: {integral_value.item():.10f}") # Convert to Python float
  print(f"analytical value: {analytical_value.item():.10f}")
  print(f"relative error: {relative_error.item():.10%}")

The output of the program is 

.. code-block:: none 

  integral value: 0.4639013112
  analytical value: 0.4596976941
  relative error: 0.9144307402%

For 1000 sampling points here, the relative error ranges from about 0.5% to 5%. More sampling points are expected to lead to a more stable relative error.

The code for the second example is given below

.. code-block:: python  

  import cupy as cp #required package for cupyint
  import cupyint
  
  data_type = cp.float32
  cupyint.set_backend(data_type) #this sets single precision data type in the backend
  
  def function(x1, x2, x3, params): # this is the standard way to define an integrand with parameters
      a1 = params[0]
      a2 = params[1]
      a3 = params[2]
      return a1 * cp.exp(-a2 * (x1**2 + x2**2 + x3**2)) + a3 * cp.sin(x1) * cp.cos(x2) * cp.exp(x3)
  
  # This sets the parameter set, which is a 2d array in all cases. In this case, we have 1e4 parameter sets
  a1_values = cp.linspace(1.0, 10.0, 10000, dtype = data_type)
  a2_values = cp.linspace(2.0, 20.0, 10000, dtype = data_type)
  a3_values = cp.linspace(0.5, 5, 10000, dtype = data_type)
  param_values = cp.stack((a1_values, a2_values, a3_values), axis=1)
  
  bound = [[0, 1], [0, 1], [0, 1]] # This sets integral limitation as (0,1),(0,1), and (0,1) for x1, x2, and x3, respectively.
  num_point = 1000 # This sets number of sampling points per dimension.
  
  def boundary(x1, x2, x3):
      condition1 = x1**2 + x2**2 + x3**2 > 0.2
      condition2 = x1**2 + x2**2 + x3**2 < 0.8
      return condition1 & condition2
  
  integral_value = cupyint.mc_integrate(function, param_values, bound, num_point, boundary) # We use mc_integrate function
  
  print(f"integral value: {integral_value.get()}") # Output integral value
  print(f"length of integral value: {integral_value.size}") # Output length of the integral value
  
  # To estimate error, we double the grids in all three dimension, and output the relative error.
  num_point = 10000 # This sets number of sampling points per dimension, which are doubled
  integral_value2 = cupyint.mc_integrate(function, param_values, bound, num_point, boundary) #We use mc_integrate function
  relative_error = cp.abs(integral_value - integral_value2) / integral_value # relative error
  
  print(f"integral value with denser grids: {integral_value2.get()}")
  print(f"relative error: {relative_error.get()}")
  
The output of this program is

.. code-block:: none 

  integral value: [0.19863702 0.19797117 0.19770709 ... 0.76723963 0.73395705 0.80251575]
  length of integral value: 10000
  integral value with denser grids: [0.19366343 0.19144712 0.19564854 ... 0.729802   0.722396   0.73479635]
  relative error: [0.02503859 0.0329545  0.01041212 ... 0.04879521 0.01575165 0.08438389]

Again, we see an improvement on the accuracy when tenfolding the grids.



.. Importance-sampling Monte Carlo integration  
.. --------

.. Importance Sampling Monte Carlo Integration is an improved Monte Carlo method that reduces variance and increases efficiency by sampling from a probability distribution related to the target function. It performs well when the probability density function closely matches the target function. For example, we try to integrate :math:`f(x)` over domain :math:`(a,b)`, given by

.. .. math::

..  I = \int_a^b f(x)\mathrm d x

.. We first introduce a normalized importance distribution :math:`p(x)` such that :math:`p(x)>0`. The integral can be rewritten as 

.. .. math::

..  I = \int_a^b \frac{f(x)}{p(x)}p(x)  \mathrm d x \approx \frac{1}{n}\sum_{i=1}^n \frac{f(x_i)}{p(x_i)}

..A proper chosen :math:`p(x)` can drastically reduce the variance of the estimate. In our code, we introduce discrete :math:`p(x)` to perform quasi-multi-importance-sampling, and this leads to another difference of the interface, seen below.

.... note::

..    The interface of Importance-sampling Monte Carlo integration is given by ``adpmc_integrate(func, params, bounds, num_points, boundaries, num_iterations)``, in which you shall see another parameter named ``num_iterations``. The higher this number, the more close the sampling function :math:`p(x)` is to the integrand.






