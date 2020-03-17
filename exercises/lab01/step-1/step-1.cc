/* ---------------------------------------------------------------------
 *
 * Copyright (C) 1999 - 2019 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * The deal.II library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE.md at
 * the top level directory of deal.II.
 *
 * ---------------------------------------------------------------------

 */


#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>

#include <cmath>
#include <fstream>
#include <iostream>

using namespace dealii;


void
first_grid()
{
  Triangulation<2> triangulation;

  const Point<2> center;
  const double   inner_radius = 0.5, outer_radius = 1.0;
  GridGenerator::hyper_shell(
    triangulation, center, inner_radius, outer_radius, 0, true);

  triangulation.reset_all_manifolds();

  SphericalManifold<2> manifold(center);

  for (auto cell : triangulation.active_cell_iterators())
    {
      if (cell->center()[1] > 0)
        cell->set_all_manifold_ids(50);
    }

  triangulation.set_manifold(50, manifold);

  const unsigned int ref_level = 7;

  for (unsigned int i = 0; i < ref_level; ++i)
    {
      std::ofstream out("grid_" + std::to_string(i) + ".vtu");
      GridOut       grid_out;
      grid_out.write_vtu(triangulation, out);
      triangulation.refine_global(1);
    }
}



template <int dim = 2>
void
second_grid()
{
  Triangulation<dim> triangulation;

  const Point<dim> center;
  const double     inner_radius = 0.5, outer_radius = 1.0;
  GridGenerator::hyper_shell(triangulation, center, inner_radius, outer_radius);
  for (unsigned int step = 0; step < 5; ++step)
    {
      for (auto &cell : triangulation.active_cell_iterators())
        //        for (const auto &v : cell->vertex_iterators())
        {
          for (unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_cell;
               ++v)
            {
              const double distance_from_center =
                center.distance(cell->vertex(v));

              if (std::fabs(distance_from_center - inner_radius) < 1e-10)
                {
                  cell->set_refine_flag();
                  break;
                }
            }
        }

      triangulation.execute_coarsening_and_refinement();
    }


  for (auto cell : triangulation.active_cell_iterators())
    {
      std::cout << "Cell " << cell << ", " << cell->center() << std::endl;
    }

  //  AssertDimension(triangulation.n_active_quads(),
  //                  triangulation.n_active_cells());

  //  Assert(triangulation.n_active_quads() == triangulation.n_active_cells(),
  //         ExcMessage("Dimensions do not match."));


  std::ofstream out("grid-2.vtk");
  GridOut       grid_out;
  grid_out.write_vtk(triangulation, out);

  std::cout << "Grid written to grid-2.vtk" << std::endl;
}



int
main()
{
  first_grid();
  // second_grid<2>();
}
