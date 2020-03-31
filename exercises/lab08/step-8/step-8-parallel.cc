/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2000 - 2019 by the deal.II authors
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

 *
 * Author: Wolfgang Bangerth, University of Heidelberg, 2000
 */

// NEW: parallel writing
#include <deal.II/base/conditional_ostream.h>

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/tensor.h>

// NEW: timing
#include <deal.II/base/timer.h>

// NEW: mpi 
#include <deal.II/base/mpi.h>

// NEW: multi-thread
#include <deal.II/base/work_stream.h>

// NEW: utilities
#include <deal.II/base/utilities.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_q.h>

// NEW: distributed grid refinement and triangulation
#include <deal.II/distributed/grid_refinement.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>

// NEW: filtered iterator
#include <deal.II/grid/filtered_iterator.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/affine_constraints.h>

// NEW: generic linear algebra
#include <deal.II/lac/generic_linear_algebra.h>

// NEW (both): scratch and copy data for multi-thread approach
#include <deal.II/meshworker/copy_data.h>
#include <deal.II/meshworker/scratch_data.h>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>


#include <fstream>
#include <iostream>

namespace LA
{
  using namespace dealii::LinearAlgebraTrilinos;
} // namespace LA

namespace Step8
{
  using namespace dealii;

  
  template <int dim>
  class ElasticProblem
  {
  public:
    ElasticProblem();
    ~ElasticProblem();
    void run();

  private:
    void setup_system();
    void assemble_system();
    void solve();
    void refine_grid();
    void output_results(const unsigned int cycle) const;
    
    // MPI communication handler
    MPI_Comm communicator;
    
    // Conditional os-stream 
    ConditionalOStream pout; 
    
    // time output variable
    mutable TimerOutput timer;

    //Triangulation<dim> triangulation;
    parallel::distributed::Triangulation<dim> triangulation;
    DoFHandler<dim>    dof_handler;

    FESystem<dim> fe;
    
    // Locally owned dofs, and my locally relevant dofs
    IndexSet locally_owned_dofs;
    IndexSet locally_relevant_dofs;

    AffineConstraints<double> hanging_node_constraints;

    //SparsityPattern      sparsity_pattern;
    // LA::MPI::SparsityPattern      sparsity_pattern;
    LA::MPI::SparseMatrix system_matrix;

    LA::MPI::Vector solution;
    LA::MPI::Vector system_rhs;
    
    LA::MPI::Vector locally_relevant_solution;
  };


  template <int dim>
  void right_hand_side(const std::vector<Point<dim>> &points,
                       std::vector<Tensor<1, dim>> &  values)
  {
    Assert(values.size() == points.size(),
           ExcDimensionMismatch(values.size(), points.size()));
    Assert(dim >= 2, ExcNotImplemented());

    
    Point<dim> point_1, point_2;
    point_1(0) = 0.5;
    point_2(0) = -0.5;

    for (unsigned int point_n = 0; point_n < points.size(); ++point_n)
      {
        
        if (((points[point_n] - point_1).norm_square() < 0.2 * 0.2) ||
            ((points[point_n] - point_2).norm_square() < 0.2 * 0.2))
          values[point_n][0] = 1.0;
        else
          values[point_n][0] = 0.0;

        
        if (points[point_n].norm_square() < 0.2 * 0.2)
          values[point_n][1] = 1.0;
        else
          values[point_n][1] = 0.0;
      }
  }



  

  template <int dim>
  ElasticProblem<dim>::ElasticProblem()
    : communicator(MPI_COMM_WORLD)
    , pout(std::cout, Utilities::MPI::this_mpi_process(communicator) == 0)
    , timer(std::cout, TimerOutput::summary, TimerOutput::cpu_and_wall_times)
    , triangulation(communicator)
    , dof_handler(triangulation)
    , fe(FE_Q<dim>(1), dim)
  {}
  
  template <int dim>
  ElasticProblem<dim>::~ElasticProblem()
  {
    dof_handler.clear();
  }


  
  template <int dim>
  void ElasticProblem<dim>::setup_system()
  { 
    TimerOutput::Scope timer_section(timer, "Setup system");
    
    dof_handler.distribute_dofs(fe);
    
    // LOCAL DOFS
    locally_owned_dofs = dof_handler.locally_owned_dofs();
    DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);
    
    hanging_node_constraints.clear();
    DoFTools::make_hanging_node_constraints(dof_handler,
                                            hanging_node_constraints);
    
    // ADDED TO REPLACE ASSEMBLE LAST PART, setting boundary values here!!!
    VectorTools::interpolate_boundary_values(dof_handler,
                                             0,
                                             Functions::ZeroFunction<dim>(dim),
                                             hanging_node_constraints);
    
    hanging_node_constraints.close();

    DynamicSparsityPattern dsp(dof_handler.n_dofs(), dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler,
                                    dsp,
                                    hanging_node_constraints,
                                    /*keep_constrained_dofs = */ true);
    
    // DISTRIBUTE SPARSITY PATTERN
    SparsityTools::distribute_sparsity_pattern(
    dsp,
    dof_handler.n_locally_owned_dofs_per_processor(),
    communicator,
    locally_relevant_dofs);

    system_matrix.reinit(locally_owned_dofs, dsp, communicator);

    solution.reinit(locally_owned_dofs, communicator);
    system_rhs.reinit(locally_owned_dofs, communicator);

    locally_relevant_solution.reinit(locally_owned_dofs,
                                     locally_relevant_dofs,
                                     communicator);
  }


  
  template <int dim>
  void ElasticProblem<dim>::assemble_system()
  {
    
    TimerOutput::Scope timer_section(timer, "Assemble system");
    
    QGauss<dim> quadrature_formula(fe.degree + 1);
    
    // DEFINE SCRATCH OBJECT
    
    MeshWorker::ScratchData<dim> scratch(fe,
                                       quadrature_formula,
                                       update_quadrature_points |
                                         update_values | update_gradients |
                                         update_JxW_values);
    

    const unsigned int dofs_per_cell = fe.dofs_per_cell;
    const unsigned int n_q_points    = quadrature_formula.size();
  
    MeshWorker::CopyData<1, 1, 1> copy_data(dofs_per_cell);
    
    // DEFINE WORKER LAMBDA FUNCTION
    
    auto worker = [&](const decltype(dof_handler.begin_active()) &cell,
                      MeshWorker::ScratchData<dim> &              scratch,
                      MeshWorker::CopyData<1, 1, 1> &             copy_data) {
  
      std::vector<double> lambda_values(n_q_points);
      std::vector<double> mu_values(n_q_points);

      Functions::ConstantFunction<dim> lambda(1.), mu(1.);
      
      std::vector<Tensor<1, dim>> rhs_values(n_q_points);
      
      auto &fe_values = scratch.reinit(cell);

      copy_data.matrices[0] = 0;
      copy_data.vectors[0]  = 0;
      
      lambda.value_list(fe_values.get_quadrature_points(), lambda_values);
      mu.value_list(fe_values.get_quadrature_points(), mu_values);
      right_hand_side(fe_values.get_quadrature_points(), rhs_values);
      
      

          
      for (unsigned int i = 0; i < dofs_per_cell; ++i)
        {
          const unsigned int component_i =
              fe.system_to_component_index(i).first;

          for (unsigned int j = 0; j < dofs_per_cell; ++j)
            {
              const unsigned int component_j =
                fe.system_to_component_index(j).first;

              for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
                {
                  copy_data.matrices[0](i, j) +=
                    (                                                  //
                      (fe_values.shape_grad(i, q_point)[component_i] * //
                       fe_values.shape_grad(j, q_point)[component_j] * //
                       lambda_values[q_point])                         //
                      +                                                //
                      (fe_values.shape_grad(i, q_point)[component_j] * //
                       fe_values.shape_grad(j, q_point)[component_i] * //
                       mu_values[q_point])                             //
                      +                                                //
                      ((component_i == component_j) ?        //
                         (fe_values.shape_grad(i, q_point) * //
                          fe_values.shape_grad(j, q_point) * //
                          mu_values[q_point]) :              //
                          0)                                  //
                          ) *                                    //
                        fe_values.JxW(q_point);                  //
                    }
                }
            }
      for (unsigned int i = 0; i < dofs_per_cell; ++i)
        {
          const unsigned int component_i =
            fe.system_to_component_index(i).first;

          for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
            copy_data.vectors[0](i) += fe_values.shape_value(i, q_point) *
                           rhs_values[q_point][component_i] *
                           fe_values.JxW(q_point);
        }

      cell->get_dof_indices(copy_data.local_dof_indices[0]);
    };
        
        // ADDED COPY LAMBDA FUNCTION
        
    auto copier = [&](const MeshWorker::CopyData<1, 1, 1> &copy_data) {
    
    
      hanging_node_constraints.distribute_local_to_global(copy_data.matrices[0],
                                             copy_data.vectors[0],
                                             copy_data.local_dof_indices[0],
                                             system_matrix,
                                             system_rhs);
    };
      
    using CellFilter =
    FilteredIterator<typename DoFHandler<dim>::active_cell_iterator>;

    WorkStream::run(
      CellFilter(IteratorFilters::LocallyOwnedCell(), dof_handler.begin_active()),
      CellFilter(IteratorFilters::LocallyOwnedCell(), dof_handler.end()),
      worker,
      copier,
      scratch,
      copy_data);

    system_matrix.compress(VectorOperation::add);
    system_rhs.compress(VectorOperation::add);
  }


  template <int dim>
  void ElasticProblem<dim>::solve()
  {
    
    TimerOutput::Scope timer_section(timer, "Solve system");
    
    SolverControl solver_control(1000, 1e-12, false, false);
    LA::SolverCG    solver(solver_control);

    LA::MPI::PreconditionSSOR::AdditionalData data(1.2);
    LA::MPI::PreconditionSSOR preconditioner;
    preconditioner.initialize(system_matrix);

    solver.solve(system_matrix, solution, system_rhs, preconditioner);

    hanging_node_constraints.distribute(solution);
    locally_relevant_solution = solution;
  }


  
  template <int dim>
  void ElasticProblem<dim>::refine_grid()
  {
    
    TimerOutput::Scope timer_section(timer, "Refine grid");
    
    Vector<float> estimated_error_per_cell(triangulation.n_active_cells());

    KellyErrorEstimator<dim>::estimate(
      dof_handler,
      QGauss<dim - 1>(fe.degree + 1),
      std::map<types::boundary_id, const Function<dim> *>(),
      solution,
      estimated_error_per_cell);

    parallel::distributed::GridRefinement::refine_and_coarsen_fixed_fraction(
    triangulation, estimated_error_per_cell, 0.33, 0.0);

    triangulation.execute_coarsening_and_refinement();
  }


  
  template <int dim>
  void ElasticProblem<dim>::output_results(const unsigned int cycle) const
  {
  
    TimerOutput::Scope timer_section(timer, "Output results");
    
    DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler);

    std::vector<std::string> solution_names;
    switch (dim)
      {
        case 1:
          solution_names.emplace_back("displacement");
          break;
        case 2:
          solution_names.emplace_back("x_displacement");
          solution_names.emplace_back("y_displacement");
          break;
        case 3:
          solution_names.emplace_back("x_displacement");
          solution_names.emplace_back("y_displacement");
          solution_names.emplace_back("z_displacement");
          break;
        default:
          Assert(false, ExcNotImplemented());
      }

    
    data_out.add_data_vector(locally_relevant_solution, solution_names);
    data_out.build_patches();

    data_out.write_vtu_in_parallel("solution_" + std::to_string(cycle) + ".vtu",
                                 communicator);
  }



  
  template <int dim>
  void ElasticProblem<dim>::run()
  {
    for (unsigned int cycle = 0; cycle < 10; ++cycle)
      {
        pout << "Cycle " << cycle << ':' << std::endl;

        if (cycle == 0)
          {
            GridGenerator::hyper_cube(triangulation, -1, 1);
            triangulation.refine_global(2);
          }
        else
          refine_grid();

        pout << "   Number of active cells:       "
                  << triangulation.n_active_cells() << std::endl;

        setup_system();

        pout << "   Number of degrees of freedom: " << dof_handler.n_dofs()
                  << std::endl;

        assemble_system();
        solve();
        output_results(cycle);
      }
  }
} // namespace Step8


int main(int argc, char **argv)
{
  try
    { 
      dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, -1);
      
      
      Step8::ElasticProblem<2> elastic_problem_2d;
      elastic_problem_2d.run();
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;

      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }

  return 0;
}
