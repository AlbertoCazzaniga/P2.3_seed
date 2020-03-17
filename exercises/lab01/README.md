## Lab 1
### Creating and manipulating Triangulations

author: Luca Heltai (luca.heltai@sissa.it)

Some useful resources

https://www.dealii.org/current/doxygen/deal.II/step_1.html https://www.dealii.org/current/doxygen/deal.II/step_49.html

### Using step-1 as a base:
1. Compile and run this tutorial on the command line or inside a suitable IDE, and inspect the output.
2. Create a helper function that takes a reference to a Triangulation and prints the following information:
	- number of levels
	- number of cells
	- number of active cells
3. Test this with all of your meshes.
4. Modifying an existing meshing function
	- Immediately after creating a mesh, call its method `reset_all_manifolds()`. What happens now?
	- Output mesh two as an `svg` file instead of `eps`. Open it in a browser to display it.
5. Creating a mesh from scratch
	- Generate a circle using `GridGenerator::hyper_ball()` in 2d (add a function `third_grid()` to `step-1`)
	- Use a `SphericalManifold` everywhere, only on the boundary, or on all cells except the center cell and refine the mesh globally twice. Can you understand what happens in the center cell?
	- Set the output format of the previous example to vtk and inspect the mesh in Paraview.
	- Create an image of an L-shape domain with one global refinement.
	- Inspect the mesh in Paraview.
	- Refine the L-shaped mesh adaptively:
		- Refine all cells with the distance between the center of the cell and re-entrant corner is smaller than 0.3.
		- Refine exactly at the re-entrant corner (i.e. those with the corner as a vertex) several times.
