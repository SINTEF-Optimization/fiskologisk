import time
import sys
import gurobipy as gp
from GurobiProblemGenerator import GurobiProblemGenerator
from GurobiProblemGenerator import ObjectiveProfile
from GurobiMasterProblemGenerator import GurobiMasterProblemGenerator
from MasterColumn import MasterColumn
from SubproblemDP import solve_dp


class SubProblem:
    """Represents a subproblem used for column generation to the Master Problem in the Dantzig-Wolfe decomposition algorithm"""

    module_index: int
    """The index of the module in the subproblem"""

    problem_generator: GurobiProblemGenerator
    """Generator for the subproblem"""

    model: gp.Model
    """The Gurobi model of the Column Generation Subproblem"""

    print_level: int
    """Level indicating which solutions to write to the console.
    Level 0 means no solution output.
    Level 1 will output the final relaxed soluton and the final MIP solution of the Master Problem.
    Level 2 will also output the dual values of the constraints of the Master Problem.
    Level 3 will also print the solutions added from the column generation problems"""

    use_dp_heuristic: bool
    """If true, use the dynamic programming heuristic instead of a full MIP formulation for the subproblem."""

    polish_dp_with_mip: bool
    """If true, and `use_dp_heuristic` is true, then solve the subproblem with full MIP formulation after the DP 
    heuristic fails to provide positive reduced cost columns."""

    def __init__(
        self,
        module_index: int,
        problem_generator: GurobiProblemGenerator,
        drop_solution: int,
        use_dp_heuristic: bool,
    ) -> None:
        self.module_index = module_index
        self.problem_generator = problem_generator
        self.model = None
        self.print_level = drop_solution

        self.use_dp_heuristic = use_dp_heuristic
        self.polish_dp_with_mip = False
        self.verify_dp_solution = False
        self.bins = 1000

    def build_model(self) -> None:
        """Builds the MIP model for the column generation subproblem"""

        self.model = self.problem_generator.build_module_subproblemn(self.module_index)

    def solve(
        self,
        iteration: int,
        period_biomass_duals: dict[int, float],
        yearly_production_duals: dict[int, float],
        convex_dual: float,
        report_suboptimality,
    ) -> (list[MasterColumn], list[MasterColumn]):
        """Solves the column generation subproblem, and returns the columns generated from the solutions found by the solver.
        To expand the search space in the Master Problem, one column is added for each of the 10 last solution found in the Gurobi solver algorithm,
        as long as the objective is positive. In addition, for each of the columns found, some extra columns are added
        that maximize or minimize the extracted mass when locking all binary variables to the values from the original column.

        args:
            - period_biomass_duals: 'dict[int, float]' The dual values of the constraints for the total mass of salmon in the tanks at a period. Key is period index.
            - yearly_production_duals: 'dict[int, float]' The dual values of the constraints for the total mass of salmon extracted as post smolt or harvest within a production year. Key is year.
            - convex_dual 'float' The dual value of the constraint for the convexity of the column combination for the module of this sibproblem.
            - module_idx 'int' The index of the module to build the sub problem for

        returns:
            Two lists of columns.
            The first list is the columns from the latest solutions found when solving the column generation subproblem. Only columns from solutions with positive cost are added.
            The second list is the columns that have the same binary values as one of the solutions in the first list, but with the objective to maximize or
            minimize the weight of the extracted salmon. Combinations will also occur, when one deploy period production is minimized and the others are maximized.
        """

        self.problem_generator.set_subproblem_objective(self.model, period_biomass_duals, yearly_production_duals)

        profit_columns = []
        if self.use_dp_heuristic:
            solution = solve_dp(
                self.problem_generator.environment,
                self.module_index,
                period_biomass_duals,
                yearly_production_duals,
                self.bins
            )

            if solution is not None:
                module = self.problem_generator.environment.modules[self.module_index]
                production_plan_constraints = self.problem_generator.lock_production_plan_by_nontransfer_num_tanks(
                    self.model, module, solution.period_tanks
                )

                self.model.optimize()
                relaxdp_obj_value = self.problem_generator.calculate_core_objective(self.module_index)

                if self.model.ObjVal > convex_dual:
                    margin = 1e-4 * max(abs(self.model.PoolObjVal), abs(convex_dual))
                    if self.model.PoolObjVal - convex_dual > margin:
                        profit_columns.append(
                            self.problem_generator.get_master_column(self.module_index, relaxdp_obj_value, False)
                        )

                self.problem_generator.remove_constraints(self.model, production_plan_constraints)

                if self.verify_dp_solution:
                    self.model.optimize()
                    mip_obj_value = self.problem_generator.calculate_core_objective(self.module_index)
                    # print(f"relax_dp is suboptimal by {100.0 * (mip_obj_value / relaxdp_obj_value - 1.0):.2f}%")
                    if report_suboptimality is not None:
                        report_suboptimality(relaxdp_obj_value, mip_obj_value)


        polish = self.polish_dp_with_mip and len(profit_columns) == 0
        if not self.use_dp_heuristic or polish:

            self.model.Params.Cutoff = convex_dual
            self.model.optimize()
            self.model.Params.Cutoff = "default"

            # Solve and look for columns with positive cost from the 10 last feasible solutions found by the solver
            nmb_sol = min(10, self.model.SolCount)

            for sol_idx in range(nmb_sol):
                self.model.params.SolutionNumber = sol_idx
                if self.model.PoolObjVal > convex_dual:
                    margin = 1e-4 * max(abs(self.model.PoolObjVal), abs(convex_dual))
                    if self.model.PoolObjVal - convex_dual > margin:
                        # Solution with positive cost found, create column to be added to Master Problem
                        obj_value = self.problem_generator.calculate_core_objective(self.module_index)
                        profit_columns.append(
                            self.problem_generator.get_master_column(self.module_index, obj_value, False)
                        )

            self.model.params.SolutionNumber = 0

        # For each new column generated by the solver solutions, add extra columns with the same binary values, and with biomass maximized or minimized for the different deploy periods
        biomass_columns = []
        for column in profit_columns:
            if self.print_level >= 3:
                print("*** New column:")
                column.drop_positive_solution()
            production_plan_constraints = self.problem_generator.lock_binaries(self.model, column)

            # Add column that maximizes biomass from all deploy periods
            if self.problem_generator.objective_profile != ObjectiveProfile.BIOMASS:
                biom_col = self.problem_generator.biomass_objective_column(self.model, self.module_index, True)
                biomass_columns.append(biom_col)
                if self.print_level >= 3:
                    print("*** Column from max biomass:")
                    biom_col.drop_positive_solution()

            # Add column that minimizes biomass from all deploy periods
            biom_col = self.problem_generator.biomass_objective_column(self.model, self.module_index, False)
            biomass_columns.append(biom_col)
            if self.print_level >= 3:
                print("*** Column from min biomass:")
                biom_col.drop_positive_solution()

            for dep_p in column.deploy_periods():
                # Add column that minimizes biomass from one of the deploy periods and maximizes from the others
                biom_col = self.problem_generator.biomass_objective_column(self.model, self.module_index, True, dep_p)
                biomass_columns.append(biom_col)
                if self.print_level >= 3:
                    print("*** Column from max biomass except deploy in %s:" % dep_p)
                    biom_col.drop_positive_solution()

            self.problem_generator.remove_constraints(self.model, production_plan_constraints)

        return profit_columns, biomass_columns


class DecomposistionSolver:
    """A solver for the salmon production planning problem using the Dantzig-Wolfe decomposition algorithm by column generations.
    One subproblem is generated for each module in the production facility, while a Master Problem is used to combine the solutions from the
    subproblem to a solution on the global production problem for the facility that satisfies the cross-module production constraints.
    """

    problem_generator: GurobiMasterProblemGenerator
    """Generator for the Master Problem in the decomposistion algorithm"""

    master_model: gp.Model
    """The Gurobi model of the Master Problem. While the Master Problem is used to generate dual values from the constraints as input
    to the column generation problems, the problem is relaxed to an LP.
    When all columns are generated and added to the Master Problem, the problem can be changed to a MIP to look for a solution with binary decission values in the original problem."""

    sub_problems: dict[int, SubProblem]
    """The column generation subproblems, one for each module in the production facility. The key is the module index."""

    print_level: int
    """Level indicating which solutions to write to the console.
    Level 0 means no solution output.
    Level 1 will output the final relaxed soluton and the final MIP solution of the Master Problem.
    Level 2 will also output the dual values of the constraints of the Master Problem.
    Level 3 will also print the solutions added from the column generation problems"""

    def __init__(self, problem_generator: GurobiMasterProblemGenerator) -> None:
        self.problem_generator = problem_generator
        self.master_model = None
        self.sub_problems = {}
        self.print_level = 1

    def build_model(self, use_dp_heuristic: bool) -> None:
        """Builds the MIP models for the Master Problem and the column generation subproblems."""

        self.sub_problems = {}
        env = self.problem_generator.environment
        for m in env.modules:
            sub_prob = SubProblem(
                m.index,
                self.problem_generator.sub_problem_generator(),
                self.print_level,
                use_dp_heuristic,
            )
            sub_prob.build_model()
            self.sub_problems[m.index] = sub_prob

        self.master_model = self.problem_generator.build_model()

    def optimize(self) -> None:
        """Runs the decomposistion subproblem algorithm."""

        log_times = []
        log_objectives = []

        # Start with one initial column for each module by using the first feasible solution found for the original MIP problem of the entire production facility
        print("*** Generating initial columns for all modules")
        t0 = time.time()
        initial_cols = self.create_initial_columns()
        log_times.append("Initial columns (sec)\t%s" % (time.time() - t0))
        for initial_col in initial_cols:
            self.problem_generator.add_column(self.master_model, initial_col)

        # The main loop in the algorithm: As long as new columns have been added from any of the subproblems, the Master Problem is solved to update the dual values of the constraints,
        # these are applied to the subproblems in order to find new columns for the master problem.
        columns_added = True
        iteration = 0
        suboptimality = []
        while columns_added:
            # First solve the relaxed Master Problem with the currently added columns.
            iteration += 1
            print("*** Solving master problem, iteration " + str(iteration))
            t0 = time.time()
            (
                period_biomass_duals,
                yearly_production_duals,
                convex_duals,
            ) = self.solve_master()
            log_times.append("Iteration %s, LP Master (sec)\t%s" % (iteration, time.time() - t0))
            if self.print_level >= 2:
                self.problem_generator.drop_dual_values(self.master_model)
            columns_added = False

            # Run the column generation problems for each module
            for sp in self.sub_problems.values():
                print("*** Searching for new column, module " + str(sp.module_index) + ", iteration " + str(iteration))
                t0 = time.time()

                def report_suboptimality(dp_obj, mip_obj):
                    suboptimality.append((sp.module_index, dp_obj, mip_obj))

                new_cols, biomass_columns = sp.solve(
                    iteration,
                    period_biomass_duals,
                    yearly_production_duals,
                    convex_duals[sp.module_index],
                    report_suboptimality,
                )
                log_times.append(
                    "Iteration %s, subproblem %s, generated %s columns (sec)\t%s"
                    % (iteration, sp.module_index, len(new_cols), time.time() - t0)
                )
                print(
                    "*** "
                    + str(len(new_cols))
                    + " new columns found and added to master problem, module "
                    + str(sp.module_index)
                    + ", iteration "
                    + str(iteration)
                )

                # Add new columns to the Master Problem from solutions when solving the column generation subproblems.
                # The additional columns from the biomass maximization/minimization objectives are only collected now and added to the Master Problem when the subproblems no longer are able to produce more columns.
                for new_col in new_cols:
                    if self.print_level >= 3:
                        new_col.drop_positive_solution()
                    self.problem_generator.add_column(self.master_model, new_col)
                    columns_added = True

                for column in biomass_columns:
                    self.problem_generator.add_column(self.master_model, column)

        self.master_model.optimize()
        self.problem_generator.generate_solution()
        relaxed_objective = self.master_model.ObjVal
        if self.print_level >= 1:
            self.problem_generator.drop_positive_solution(self.master_model)

        # If the solution to the relaxed Master Problem does not give a solution with binary decission values, we change the Master Problem to a MIP by applying the binary constraints, and resulve using the same columns as before.
        if not self.problem_generator.validate_integer_values():
            print("*** Some relaxed binary variables failed to be integers, fixing to binary")
            log_objectives.append("Objective solved relaxed master problem\t%s" % relaxed_objective)
            self.problem_generator.lock_binaries()
            t0 = time.time()
            self.master_model.Params.MIPGap = 0.2 / 100.0
            self.master_model.optimize()
            log_times.append("MIP Master (sec)\t%s" % (time.time() - t0))
            mip_objective = self.master_model.ObjVal
            log_objectives.append("Objective solved MIP master problem\t%s" % mip_objective)
            percent_under = 100.0 * (relaxed_objective - mip_objective) / relaxed_objective
            log_objectives.append("MIP Objective below relaxed objective (%%)\t%s" % percent_under)
            self.problem_generator.generate_solution()
            print("*** MIP version of master problem solved after iteration " + str(iteration))
            if self.print_level >= 1:
                self.problem_generator.drop_positive_solution(self.master_model)
        else:
            log_objectives.append("Objective solved relaxed master problem, binaries satisfied\t%s" % relaxed_objective)
            print("*** All relaxed binary variables are integers in solution to relaxed master problem")

        " Finally output the times spent by the solvers and the the objective values of the solutions for the relaxed Master Problem and the MIP Master Problem"
        for msg in log_times:
            print(msg)
        for msg in log_objectives:
            print(msg)

        if len(suboptimality) > 0:
            print("Suboptimality of DP approach:")
            for mod, dp_obj, mip_obj in suboptimality:
                print(f" - module{mod} dp={dp_obj} mip={mip_obj} suboptimality={100.0 * ( mip_obj / dp_obj  - 1.0 ):.2f}%")

    def create_initial_columns(self) -> list[MasterColumn]:
        """Generates the initial columns to be added to the Master Problem.
        This is done by using the first feasible solution found when trying to solve the production planning problem as one big MIP problem,
        and then splitting the solution into one column for each module.

        returns:
            A list of the columns from the initial solution, one for each module
        """

        gpg = self.problem_generator.sub_problem_generator()
        model = gpg.build_model()
        model.params.SolutionLimit = 1
        obj_expr = {}
        for m in gpg.environment.modules:
            obj_expr[m.index] = gpg.create_core_objective_expression(m.index)
        model.optimize()
        if self.print_level >= 3:
            gpg.drop_positive_solution(model)
        columns = []
        for m_idx, expr in obj_expr.items():
            obj_value = expr.getValue()
            columns.append(gpg.get_master_column(m_idx, obj_value, True))

        return columns

    def solve_master(self) -> (dict[int, float], dict[int, float], dict[int, float]):
        """Solves the Master Problem with the currently added columns and returns get the dual values of the constraints.

        returns:
            Three dictionaries with the dual values.
            The first dictionary holds the dual values of the constraints for the total mass of salmon in the tanks at a period. Key is period index.
            The second dictionary holds the dual values of the constraints for the total mass of salmon extracted as post smolt or harvest within a production year. Key is year.
            The third dictionary holds the dual values of the constraints for the convexity of the column combination for each module. Key is module index.
        """

        self.master_model.optimize()
        return self.problem_generator.get_dual_values()
