## Lab 2
### Creating a DoFHandler and visualising sparsity patterns

author: Luca Heltai (luca.heltai@sissa.it)

Some useful resources
https://www.dealii.org/current/doxygen/deal.II/step_3.html 

### Using step-3 as a base:
1. Compile and run this tutorial, and inspect at the output.
    - Modify the code so that the problem is dimension-independent.
    - Switch to vtk output and visualize in Paraview. Figure out how to warp the solution by the solution variable.
    - Add a zero Neumann boundary condition to one edge of the domain. Assign this Neumann boundary the indicator 1.
    Tip: Look at the instructions in “Modify the type of boundary condition” in the “Possibilities for extensions” section of the tutorial.
    - Add a non-zero Dirichlet boundary condition to one edge of the domain 
    - Set the value to 0.5 for the boundary with indicator 1.
    Tip: Look at the instructions in “A slight variation of the last point” in the “Possibilities 
    for extensions” section of the tutorial.
    - Change the setup to have f = 0. Compare this result to that where f is non-zero.
2. Additional tasks
    - Do “Convergence of the mean”. Can you see the order $h^2$?
    - Increase the polynomial order (you need to increase all orders of the quadratures in the program!) and check the convergence of the mean now.
    - Switch to an L-shaped domain and experiment with a combination of Dirichlet and Neumann boundary conditions. By experimentation, identify the faces adjacent to the re-entrant corner and apply Dirichlet conditions only there.
    Tip: There is more than one way to generate such a grid using the built-in functions.
