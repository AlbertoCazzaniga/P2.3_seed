/* ---------------------------------------------------------------------
 *
 * Copyright (C) 1999 - 2016 by the deal.II authors
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

 *
 * Authors: Wolfgang Bangerth, 1999,
 *          Guido Kanschat, 2011
 */

#include <deal.II/base/function.h>
#include <deal.II/base/function_parser.h>
#include <deal.II/base/parsed_convergence_table.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>

#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <fstream>
#include <iostream>

using namespace dealii;


template <int dim>
class Step3
{
public:
  Step3();

  void
  run(const unsigned int n_cycles           = 1,
      const unsigned int initial_refinement = 3);


private:
  void
  make_grid(const unsigned int ref_level);
  void
  estimate_error();
  void
  mark_cells_for_refinement();
  void
  refine_grid();
  void
  setup_system();
  void
  assemble_system();
  void
  solve();
  void
  compute_error();
  void
  output_results(const unsigned int cycle) const;

  mutable TimerOutput timer;

  Triangulation<dim> triangulation;
  FE_Q<dim>          fe;
  DoFHandler<dim>    dof_handler;

  AffineConstraints<double> constraints;

  SparsityPattern      sparsity_pattern;
  SparseMatrix<double> system_matrix;

  Vector<double> solution;
  Vector<double> system_rhs;

  Vector<float> error_estimator;

  Vector<double> L2_error_per_cell;
  Vector<double> H1_error_per_cell;

  /** Exact solution (used to manufacture a rhs). */
  FunctionParser<dim> exact_solution;

  /** Manufactured right hand side. */
  FunctionParser<dim> rhs_function;

  /** Utility to compute error tables. */
  ParsedConvergenceTable error_table;
};

template <int dim>
Step3<dim>::Step3()
  : timer(std::cout, TimerOutput::summary, TimerOutput::cpu_and_wall_times)
  , fe(1)
  , dof_handler(triangulation)
  , exact_solution("exp(x)*exp(y)")
  , rhs_function("-2*exp(x)*exp(y)")
  , error_table({"u"}, {{VectorTools::H1_norm, VectorTools::L2_norm}})
{}


template <int dim>
void
Step3<dim>::make_grid(const unsigned int ref_level)
{
  TimerOutput::Scope timer_section(timer, "Make grid");
  triangulation.clear();
  GridGenerator::hyper_cube(triangulation, -1, 1);
  triangulation.refine_global(ref_level);

  std::cout << "Number of active cells: " << triangulation.n_active_cells()
            << std::endl;
}

template <int dim>
void
Step3<dim>::estimate_error()
{
  // We fill error_estimator with an indicator on when the grid needs refinement
  error_estimator.reinit(triangulation.n_active_cells());
  QGauss<dim - 1> face_quadrature(fe.degree + 1);

  KellyErrorEstimator<dim>::estimate(
    dof_handler, face_quadrature, {}, solution, error_estimator);
}

template <int dim>
void
Step3<dim>::mark_cells_for_refinement()
{
  GridRefinement::refine_and_coarsen_fixed_fraction(triangulation,
                                                    error_estimator,
                                                    0.33,
                                                    0.0);
}



template <int dim>
void
Step3<dim>::refine_grid()
{
  TimerOutput::Scope timer_section(timer, "Refine grid");
  triangulation.execute_coarsening_and_refinement();
  // triangulation.refine_global(1);
}


template <int dim>
void
Step3<dim>::setup_system()
{
  TimerOutput::Scope timer_section(timer, "Setup dofs");
  dof_handler.distribute_dofs(fe);

  constraints.clear();
  DoFTools::make_hanging_node_constraints(dof_handler, constraints);

  VectorTools::interpolate_boundary_values(dof_handler,
                                           0,
                                           exact_solution,
                                           constraints);
  constraints.close();

  DynamicSparsityPattern dsp(dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints);
  sparsity_pattern.copy_from(dsp);

  system_matrix.reinit(sparsity_pattern);

  solution.reinit(dof_handler.n_dofs());
  system_rhs.reinit(dof_handler.n_dofs());
}


template <int dim>
void
Step3<dim>::assemble_system()
{
  TimerOutput::Scope timer_section(timer, "Assemble system");
  QGauss<dim>        quadrature_formula(2);
  FEValues<dim>      fe_values(fe,
                          quadrature_formula,
                          update_quadrature_points | update_values |
                            update_gradients | update_JxW_values);

  const unsigned int dofs_per_cell = fe.dofs_per_cell;
  const unsigned int n_q_points    = quadrature_formula.size();

  FullMatrix<double>  cell_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double>      cell_rhs(dofs_per_cell);
  std::vector<double> rhs_values(n_q_points);

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  for (auto cell : dof_handler.active_cell_iterators())
    {
      fe_values.reinit(cell);

      cell_matrix = 0;
      cell_rhs    = 0;

      const auto &q_points = fe_values.get_quadrature_points();
      rhs_function.value_list(q_points, rhs_values);

      for (unsigned int q_index = 0; q_index < n_q_points; ++q_index)
        {
          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            for (unsigned int j = 0; j < dofs_per_cell; ++j)
              cell_matrix(i, j) +=
                (fe_values.shape_grad(i, q_index) *
                 fe_values.shape_grad(j, q_index) * fe_values.JxW(q_index));

          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            cell_rhs(i) += (fe_values.shape_value(i, q_index) *
                            rhs_values[q_index] * fe_values.JxW(q_index));
        }
      cell->get_dof_indices(local_dof_indices);

      constraints.distribute_local_to_global(
        cell_matrix, cell_rhs, local_dof_indices, system_matrix, system_rhs);
    }
}


template <int dim>
void
Step3<dim>::solve()
{
  TimerOutput::Scope timer_section(timer, "Solve system");
  SolverControl      solver_control(1000, 1e-12, false, false);
  SolverCG<>         solver(solver_control);

  solver.solve(system_matrix, solution, system_rhs, PreconditionIdentity());
  constraints.distribute(solution);
}



template <int dim>
void
Step3<dim>::compute_error()
{
  TimerOutput::Scope timer_section(timer, "Compute error");
  L2_error_per_cell.reinit(triangulation.n_active_cells());
  H1_error_per_cell.reinit(triangulation.n_active_cells());
  QGauss<dim> error_quadrature(2 * fe.degree + 1);

  VectorTools::integrate_difference(dof_handler,
                                    solution,
                                    exact_solution,
                                    L2_error_per_cell,
                                    error_quadrature,
                                    VectorTools::L2_norm);


  VectorTools::integrate_difference(dof_handler,
                                    solution,
                                    exact_solution,
                                    H1_error_per_cell,
                                    error_quadrature,
                                    VectorTools::H1_norm);

  std::cout << "L2 norm of error: " << L2_error_per_cell.l2_norm() << std::endl;
  std::cout << "H1 norm of error: " << H1_error_per_cell.l2_norm() << std::endl;
}


template <int dim>
void
Step3<dim>::output_results(const unsigned int cycle) const
{
  TimerOutput::Scope timer_section(timer, "Output results");
  DataOut<dim>       data_out;
  data_out.attach_dof_handler(dof_handler);
  data_out.add_data_vector(solution, "solution");
  data_out.add_data_vector(L2_error_per_cell, "L2_error");
  data_out.add_data_vector(H1_error_per_cell, "H1_error");
  data_out.add_data_vector(error_estimator, "Error_estimator");
  data_out.build_patches();

  std::ofstream output("solution_" + std::to_string(cycle) + ".vtu");
  data_out.write_vtu(output);
}


template <int dim>
void
Step3<dim>::run(const unsigned int n_cycles,
                const unsigned int initial_refinement)
{
  make_grid(initial_refinement);
  for (unsigned int cycle = 0; cycle < n_cycles; ++cycle)
    {
      std::cout << "Cycle " << cycle << std::endl;
      setup_system();
      assemble_system();
      solve();

      // Compute the actual error from the exact solution
      compute_error();
      // Compute an estimate of the error using Kelly error estimator
      estimate_error();
      error_table.error_from_exact(dof_handler, solution, exact_solution);
      output_results(cycle);

      if (cycle != n_cycles - 1)
        {
          // Mark and refine
          mark_cells_for_refinement();
          refine_grid();
        }
    }
  error_table.output_table(std::cout);
}



int
main()
{
  deallog.depth_console(2);

  Step3<2> laplace_problem;
  laplace_problem.run(8);

  return 0;
}
