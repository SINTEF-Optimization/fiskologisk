import sys
import time
import getopt
import gurobipy as gp
import json
import os
import math
from fiskologisk.solvers.DecompCyclesSolver import decomp_cycles_solve
from fiskologisk.SolutionProvider import SolutionProvider
from fiskologisk.solvers.GurobiProblemGenerator import GurobiProblemGenerator
from fiskologisk.solvers.GurobiProblemGenerator import ObjectiveProfile
from fiskologisk.solvers.GurobiMasterProblemGenerator import GurobiMasterProblemGenerator
from fiskologisk.solvers.DecomposistionSolver import DecomposistionSolver
from fiskologisk.domain.Environment import Environment
from fiskologisk.read_problem import read_core_problem

class Iteration:

    def __init__(self, current_iteration, max_iteration, unextended_planning_years, core_setup_file, input_file, solution_output_file, initial_populations) -> None:
        self.core_setup_file = core_setup_file
        self.current_iteration = current_iteration
        self.max_iteration = max_iteration
        self.unextended_planning_years = unextended_planning_years
        self.input_file = input_file
        self.solution_output_file = solution_output_file
        self.initial_populations = initial_populations

def run_iteration(file_path: str, objective: ObjectiveProfile, allow_transfer: bool, use_decomposistion: int, add_symmetry_breaks: bool, max_single_modules: int, fixed_values_file: str, use_dp_heuristic :bool) -> None:

    file_dir = os.path.dirname(file_path)
    iteration = read_iteration_setup(file_path)

    environment = read_core_problem(file_dir, iteration.core_setup_file)
    environment.add_initial_populations(iteration.initial_populations)

    sol_prov: SolutionProvider = None
    model :gp.Model = None
    time0 = time.time()

    # Decomposition 0 is the full MIP formulation
    # Decomposition 1 is the column generation model where the columns are single-module production plans
    # Decomposition 2 is the column generation model where the columns are production cycles (deploy to harvest of a single module).

    if use_decomposistion == 2:
        print(" ** Starting single-module production-cycle decomposition solver.")
        sol_prov, model = decomp_cycles_solve(environment, objective, allow_transfer, add_symmetry_breaks)

    elif use_decomposistion == 1:
        print(" ** Starting single-module full-horizon decomposition solver.")
        gmpg = GurobiMasterProblemGenerator(environment, objective_profile = objective, allow_transfer = allow_transfer, add_symmetry_breaks = add_symmetry_breaks, max_single_modules = max_single_modules)
        sol_prov = gmpg

        decomp_solver = DecomposistionSolver(gmpg)
        decomp_solver.build_model(use_dp_heuristic)
        decomp_solver.optimize()
        model = decomp_solver.master_model
        
    else:
        print(" ** Starting full MIP solve.")
        gpg = GurobiProblemGenerator(environment, objective_profile = objective, allow_transfer = allow_transfer, add_symmetry_breaks = add_symmetry_breaks, max_single_modules = max_single_modules)
        sol_prov = gpg
        model = gpg.build_model()

        if fixed_values_file != "":
            fixed_values_file_path = os.path.join(file_dir.replace("\\","/"), fixed_values_file.replace("\\","/"))
            with open(fixed_values_file_path, "r") as input_fixed_values_file:
                fixed_values_json = json.load(input_fixed_values_file)
                gpg.add_fixed_values(model, fixed_values_json)

        model.optimize()

    time1 = time.time()

    if model.SolCount == 0:
        print(f" ** No solution found.")
    else:
        print(f" ** Best solution objective value: {model.ObjVal:.2f}")

    print(f" ** Solver finished after {time1-time0:.3f}s")

    if iteration.current_iteration < iteration.max_iteration:

        next_initial_populations = []
        first_planning_idx = environment.periods[0].index
        unextended_planning_periods = 12 * iteration.unextended_planning_years
        first_extended_idx = first_planning_idx + unextended_planning_periods

        if (environment.periods[-1].index >= first_extended_idx):

            first_extended_period = next(p for p in environment.periods if p.index == first_extended_idx)
            last_horizon_period = next(p for p in environment.periods if p.index == first_extended_idx - 1)

            for t in environment.tanks:
                init_in_use = sol_prov.contains_salmon_value(t, last_horizon_period) > 0.5
                init_dep_p = 0
                init_weight = 0.0

                for dep_p in first_extended_period.deploy_periods:
                    if dep_p != first_extended_period:
                        pop_val = sol_prov.population_weight_value(dep_p, t, first_extended_period)
                        if pop_val > 0.5:
                            init_dep_p = dep_p.index - unextended_planning_periods
                            init_weight = pop_val

                if init_in_use or init_weight > 0.0:
                    next_init_pop = { "tank": t.index }
                    if init_weight > 0.0:
                        red_init_weight = float(math.floor(init_weight - 1e-7))
                        if red_init_weight > 0.0:
                            next_init_pop["deploy_period"] = init_dep_p
                            next_init_pop["weight"] = red_init_weight
                    if init_in_use:
                        next_init_pop["in_use"] = True
                    next_initial_populations.append(next_init_pop)


        next_iteration = iteration.current_iteration + 1
        iteration_setup = { "current_iteration": next_iteration, "max_iteration": iteration.max_iteration, "unextended_planning_years": iteration.unextended_planning_years, "input_file": iteration.input_file }
        if iteration.solution_output_file != None:
            iteration_setup["solution_output_file"] = iteration.solution_output_file
        run_file_setup = { "core_setup": iteration.core_setup_file, "iteration_setup": iteration_setup, "initial_populations": next_initial_populations }

        outfile_path_local = iteration.input_file.replace("%N", str(next_iteration))
        outfile_path = os.path.join(file_dir, outfile_path_local)
        json_object = json.dumps(run_file_setup, indent=4)
        print(f" ** Non-final iteration {iteration.current_iteration}/{iteration.max_iteration}. Writing initial tank contents to {outfile_path}")
        with open(outfile_path, "w") as outfile:
            outfile.write(json_object)

    if iteration.solution_output_file != None:
        solution_output_file_local = iteration.solution_output_file.replace("%N", str(iteration.current_iteration))
        output_filename = os.path.join(file_dir, solution_output_file_local)
        print(f" ** Writing solution to {output_filename}")
        write_solution_file(output_filename, environment, iteration.unextended_planning_years, sol_prov, allow_transfer)

def print_variables(variables: list[gp.Var], min_val: float) -> None:
    for v in variables:
        if v.X > min_val:
            print(v.VarName + " = " + str(v.X))

def read_iteration_setup(file_path: str) -> Iteration:
    file_path = file_path.replace("\\","/")
    with open(file_path, "r") as input_file:
        data = json.load(input_file)
        core_setup_file = data["core_setup"]
        iteration_setup = data["iteration_setup"]
        current_iteration = iteration_setup["current_iteration"]
        max_iteration = iteration_setup["max_iteration"]
        unextended_planning_years = iteration_setup["unextended_planning_years"]
        input_file = iteration_setup["input_file"]
        if "solution_output_file" in iteration_setup:
            solution_output_file = iteration_setup["solution_output_file"]
        else:
            solution_output_file = None
        if "initial_populations" in data:
            initial_populations = data["initial_populations"]
        else:
            initial_populations = []

    return Iteration(current_iteration, max_iteration, unextended_planning_years, core_setup_file, input_file, solution_output_file, initial_populations)

def write_solution_file(file_path: str, environment: Environment, planning_years: int, sol_prov: SolutionProvider, allow_transfer: bool) -> Iteration:
    modules = []
    for m in environment.modules:
        tank_indices = []
        tank_transfers = []
        for t in m.tanks:
            tank_indices.append(t.index)
            for to_t in t.transferable_to:
                tank_transfers.append({"from": t.index, "to": to_t.index})
        modules.append({"module_index": m.index, "tank_indices": tank_indices, "tank_transfers": tank_transfers})

    first_plan_year = environment.years[0].year
    first_plan_period = environment.periods[0].index
    last_plan_period = environment.periods[-1].index
    last_horizon_period = first_plan_period + 12 * planning_years - 1
    first_preplan_year = first_plan_year - first_plan_period // 12
    first_preplan_period = 0
    preplan_deploy_periods = []
    for p in environment.preplan_release_periods:
        preplan_deploy_periods.append(p.index)
    plan_deploy_periods = []
    for p in environment.plan_release_periods:
        plan_deploy_periods.append(p.index)
    pre_planning_horizon = {"first_year": first_preplan_year, "first_period": first_preplan_period, "last_period": first_plan_period - 1, "deploy_periods": preplan_deploy_periods}
    planning_horizon = {"first_year": first_plan_year, "first_period": first_plan_period, "last_ordinary_horizon_period": last_horizon_period, "last_period": last_plan_period, "deploy_periods": plan_deploy_periods}

    module_by_tank = {}
    for m in environment.modules:
        for t in m.tanks:
            module_by_tank[t.index] = m.index
    prod_cyles_by_deploy = {}

    # Start of tank cycles initiated before planning horizon
    for t in environment.tanks:
        if t.initial_weight > 0:
            add_tank_cycle_start(prod_cyles_by_deploy, t.initial_deploy_period, module_by_tank[t.index], t.index, first_plan_period, "pre_planning_deploy")

    # Start of tank cycles initiated by deploy
    for dep_p in environment.plan_release_periods:
        if len(dep_p.periods_after_deploy) > 0 and dep_p.periods_after_deploy[0] == dep_p:
            for t in environment.tanks:
                value = sol_prov.population_weight_value(dep_p, t, dep_p)
                if value > 0.5:
                    add_tank_cycle_start(prod_cyles_by_deploy, dep_p.index, module_by_tank[t.index], t.index, dep_p.index, "deploy")

    # Start of tank cycles initiated by transfer
    if allow_transfer:
        for dep_p in environment.release_periods:
            for p in dep_p.transfer_periods:
                for from_t in environment.tanks:
                    for to_t in from_t.transferable_to:
                        value = sol_prov.transfer_weight_value(dep_p, from_t, to_t, p)
                        if value > 0.5:
                            add_tank_cycle_start(prod_cyles_by_deploy, dep_p.index, module_by_tank[to_t.index], to_t.index, p.index + 1, "transfer", from_t.index, value)

    # Add tank populations
    for dep_p in environment.release_periods:
        for p in dep_p.periods_after_deploy:
            for t in environment.tanks:
                value = sol_prov.population_weight_value(dep_p, t, p)
                if value > 0.5:
                    tank_cycle = get_tank_cycle(prod_cyles_by_deploy, dep_p.index, module_by_tank[t.index], t.index, p.index)
                    add_tank_cycle_weight(tank_cycle, p.index, value)

    # Add extractions
    for dep_p in environment.release_periods:
        for extr_idx in range(2):
            extract_periods = dep_p.postsmolt_extract_periods if extr_idx == 0 else dep_p.harvest_periods
            for p in extract_periods:
                for t in environment.tanks:
                    value = sol_prov.extract_weight_value(dep_p, t, p)
                    if value > 0.5:
                        tank_cycle = get_tank_cycle(prod_cyles_by_deploy, dep_p.index, module_by_tank[t.index], t.index, p.index)
                        tank_cycle["end_period"] = p.index
                        tank_cycle["end_cause"] = "post_smolt" if extr_idx == 0 else "harvest"

    production_cycles = []
    for dep_p_idx, mod_deploy_cycles in sorted(prod_cyles_by_deploy.items()):
        for mod_idx, cycle_tank_cycles in sorted(mod_deploy_cycles.items()):
            tank_cycles = []
            for _, tank_cycles_for_tank in sorted(cycle_tank_cycles.items()):
                for _, tank_cycle in sorted(tank_cycles_for_tank.items()):
                    tank_cycles.append(tank_cycle)
            production_cycles.append({"deploy_period": dep_p_idx, "module": mod_idx, "tank_cycles": tank_cycles})

    solution = {"modules": modules, "pre_planning_horizon": pre_planning_horizon, "planning_horizon": planning_horizon, "production_cycles": production_cycles}
    json_object = json.dumps(solution, indent=4)
    with open(file_path, "w") as outfile:
        outfile.write(json_object)


def add_tank_cycle_start(prod_cyles_by_deploy, dep_p_idx: int, mod_idx: int, tank_idx: int, start_period_idx: int, start_cause: str, from_tank_idx: int = 0, transfer_weight: float = 0.0) -> None:

    if not dep_p_idx in prod_cyles_by_deploy:
        prod_cyles_by_deploy[dep_p_idx] = {}
    deploy_prod_cycles = prod_cyles_by_deploy[dep_p_idx]
    if not mod_idx in deploy_prod_cycles:
        deploy_prod_cycles[mod_idx] = {}
    module_cycles = deploy_prod_cycles[mod_idx]
    if not tank_idx in module_cycles:
        module_cycles[tank_idx] = {}
    tank_cycles = module_cycles[tank_idx]
    tank_cycle = {"tank": tank_idx, "start_period": start_period_idx, "start_cause": start_cause}
    tank_cycles[start_period_idx] = tank_cycle
    if transfer_weight > 0.0:
        tank_cycle["transfer"] = {"period": start_period_idx - 1, "from_tank": from_tank_idx, "biomass": transfer_weight}
    tank_cycle["end_period"] = None
    tank_cycle["end_cause"] = None
    tank_cycle["period_biomasses"] = []

def get_tank_cycle(prod_cyles_by_deploy, dep_p_idx: int, mod_idx: int, tank_idx: int, period_idx: int):

    tank_cycles = prod_cyles_by_deploy[dep_p_idx][mod_idx][tank_idx]
    start_p = None
    for sp in tank_cycles.keys():
        if sp <= period_idx and (start_p == None or start_p < sp):
            start_p = sp
    return tank_cycles[start_p]

def add_tank_cycle_weight(tank_cycle, period_idx: int, weight: float) -> None:

    tank_cycle["period_biomasses"].append({"period": period_idx, "biomass": weight})

def parse_objective(value: str, default: ObjectiveProfile) -> ObjectiveProfile:

    value_up = value.upper()

    if value_up == "PROFIT":
        return ObjectiveProfile.PROFIT
    elif value_up == "BIOMASS":
        return ObjectiveProfile.BIOMASS
    else:
        return default

def parse_bool(value: str, default: bool) -> bool:

    value_up = value.upper()

    if value_up in ("T", "TRUE", "1", "Y", "YES"):
        return True
    elif value_up in ("F", "FALSE", "0", "N", "NO"):
        return False
    else:
        return default

def parse_int(value: str, default: int) -> int:

    try:
        result = int(value)
    except ValueError:
        result = default
    return result

if __name__ == "__main__":
    print("** Fiskologisk production planning v0.2")

    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <PROBLEMFILE> [OPTIONS]*")
        print("  The options are:")
        print("   --Objective OBJECTIVE  - where OBJECTIVE is \"profit\" or \"biomass\"")
        print("   --Symmetry_break BOOL")
        print("   --Transfer BOOL")
        print("   --Decomposition 0  - solve the full MIP formulation")
        print("   --Decomposition 1  - solve the column generation model with single-module full-horizon subproblems")
        print("   --Decomposition 2  - solve the column generation model with single-module single-production-cycle subproblems")
        print("   --Max_single_modules INT")
        print("   --Fixed FILE  - file containing additional restrictions to fixed values for some variables")
        print("   --Heuristic BOOL  - use the DP heuristic in decomposition 1")
        print("")
        print("See the README file for an explanation of iterations and example use of this program.")
        print("")
        sys.exit(1)

    file_path = sys.argv[1]
    objective = ObjectiveProfile.PROFIT
    allow_transfer = True
    add_symmetry_breaks = False

    # Decomposition 0 is the full MIP formulation
    # Decomposition 1 is the column generation model where the columns are single-module production plans
    # Decomposition 2 is the column generation model where the columns are production cycles (deploy to harvest of a single module).
    use_decomposistion = 0
    max_single_modules = 0
    fixed_values_file = ""
    use_dp_heuristic = False

    opt_arguments = sys.argv[2:]
    options = "d:f:m:o:s:t:h:"
    long_options = ["Decomposition=", "Fixed=", "Objective=", "Symmetry_break=", "Transfer=", "Max_single_modules=", "Heuristic="]

    try:
        arguments, values = getopt.getopt(opt_arguments, options, long_options)

        for argument, value in arguments:

            if argument in ("-o", "--Objective"):
                objective = parse_objective(value, ObjectiveProfile.PROFIT)
            
            elif argument in ("-s", "--Symmetry_break"):
                add_symmetry_breaks = parse_bool(value, True)

            elif argument in ("-t", "--Transfer"):
                allow_transfer = parse_bool(value, True)

            elif argument in ("-d", "--Decomposition"):
                use_decomposistion = parse_int(value, 0)

            elif argument in ("-m", "--Max_single_modules"):
                max_single_modules = parse_int(value, 0)

            elif argument in ("-f", "--Fixed"):
                fixed_values_file = value

            elif argument in ("-h", "--Heuristic"):
                use_dp_heuristic = parse_bool(value, False)

        run_iteration(file_path, objective, allow_transfer, use_decomposistion, add_symmetry_breaks, max_single_modules, fixed_values_file, use_dp_heuristic)

    except getopt.error as err:
        print(str(err))
