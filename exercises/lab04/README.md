## Lab 4
### Global and local error computation and estimation

author: Luca Heltai (luca.heltai@sissa.it)

Some useful resources
https://www.dealii.org/current/doxygen/deal.II/step_6.html https://www.dealii.org/current/doxygen/deal.II/step_7.html https://www.dealii.org/current/doxygen/deal.II/group__numerics.html

### Using step-5 as a base (or your previously modified step-3):
1. Solve the problem $-\Delta u(x) = f(x) \text{ in } [0, 1]^2$ , with $u(x) = g(x)$ on $\partial \Omega$, and
	- Set the boundary conditions $g(x)$ and the forcing function $f(x)$ to get the manufactured solution $u(x) = \sin(\pi x_1) \cos(\pi x_2)$.
	- Make sure the L2 errors are converging.
	Tip: Look at the `VectorTools::integrate_difference` function.
	- Use the `KellyErrorEstimator` to predict the regions of the geometry where the solution approximation is inaccurate. Visualise this error using Paraview. Do you observe any correlation between the gradient of the solution and the estimated local solution error?
	Tip: Use a different quadrature rule to prevent super-convergent effects when using the `KellyErrorEstimator`.
	- Perform local cell marking and refinement using the cell-based estimated error. For this, the logic of the refine_mesh function must be modified.
2. Additional tasks
	- Compare the convergence rates (number of DoFs versus the solution error, best viewed in a log-log plot) when using global refinement and when using local refinement with the Kelly error estimator.
	- Investigate the influence of the coarsening and refinement parameters on the solution accuracy.
	- Investigate the effect of changing the polynomial order for the solution ansatz on the solution accuracy.
	- Integrate a non-constant coefficient into the governing equation, i.e. solve the heterogeneous Poisson equation
	$-\nabla\cdot(k(x)\nabla u(x)) = f(x)$ in $\Omega$
	where $k(x)$ represents some material parameter. Repeat the calculation of the error using the `KellyErrorEstimator`, while taking this spatially dependent coefficient into consideration. 
    Tip: Look at the documentation for the `KellyErrorEstimator` before deciding on how to implement $k(x)$
    - Choose $k(x)$ to be spatially discontinuous. Do you observe any correlation between the location of the material discontinuity and the estimated local solution error? What influence does this have on the location of the refined cells?
    Tip: Verify your conclusions by looking to the results of `step-6`.
    Further information can be found in the discussion “Playing with the regularity of the solution” in the “Possibilities for extensions” section of step-6.