/* ---------------------------------------------------------------------
 *
 * Copyright (C) 1999 - 2015 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * The deal.II library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE at
 * the top level of the deal.II distribution.
 *
 * ---------------------------------------------------------------------

 */

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_q.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>

#include <deal.II/lac/vector.h>

#include <deal.II/numerics/data_out.h>

#include <cmath>
#include <fstream>
#include <iostream>

using namespace dealii;


int main()
{
  const int dim    = 2;
  const int degree = 2;

  Triangulation<dim> triangulation;

  FE_Q<dim>            fe(degree);
  DoFHandler<dim>      dh(triangulation);
  MappingQGeneric<dim> mapping(degree);

  GridGenerator::hyper_shell(triangulation, Point<dim>(), .5, 1.0);

  dh.distribute_dofs(fe);

  std::vector<Vector<double>> solution(dh.n_dofs(),
                                       Vector<double>(dh.n_dofs()));

  DataOut<dim>          data_out;
  DataOutBase::VtkFlags flags;
  flags.write_higher_order_cells = true;
  data_out.set_flags(flags);

  std::cout << "Dofs: " << dh.n_dofs() << std::endl;
  std::cout << "Vertices: " << triangulation.n_vertices() << std::endl;

  std::ofstream out("solution.vtk");

  data_out.attach_dof_handler(dh);

  for (unsigned int i = 0; i < dh.n_dofs(); ++i)
    {
      solution[i][i] = 1;
      data_out.add_data_vector(solution[i], "solution_" + std::to_string(i));
    }

  data_out.build_patches(mapping, degree, DataOut<dim>::curved_inner_cells);
  data_out.write_vtk(out);
}
