import sys
import getopt
import gurobipy as gp
import json
import os
from GurobiProblemGenerator import GurobiProblemGenerator
from GurobiProblemGenerator import ObjectiveProfile
from Environment import Environment
from read_problem import read_core_problem

class Iteration:

    def __init__(self, current_iteration, max_iteration, unextended_planning_years, core_setup_file, input_file, solution_output_file, initial_populations) -> None:
        self.core_setup_file = core_setup_file
        self.current_iteration = current_iteration
        self.max_iteration = max_iteration
        self.unextended_planning_years = unextended_planning_years
        self.input_file = input_file
        self.solution_output_file = solution_output_file
        self.initial_populations = initial_populations

def run_iteration(file_path: str, objective: ObjectiveProfile, allow_transfer: bool, add_symmetry_breaks: bool) -> None:

    file_dir = os.path.dirname(file_path)
    iteration = read_iteration_setup(file_path)

    environment = read_core_problem(file_dir, iteration.core_setup_file)
    environment.add_initial_populations(iteration.initial_populations)

    gpm = GurobiProblemGenerator(environment, objective_profile = objective, allow_transfer = allow_transfer)
    model = gpm.build_model()

    model.optimize()

    if iteration.current_iteration < iteration.max_iteration:

        next_initial_populations = []
        first_planning_idx = environment.periods[0].index
        unextended_planning_periods = 12 * iteration.unextended_planning_years
        first_extended_idx = first_planning_idx + unextended_planning_periods

        if (environment.periods[-1].index >= first_extended_idx):

            first_extended_period = next(p for p in environment.periods if p.index == first_extended_idx)
            last_horizon_period = next(p for p in environment.periods if p.index == first_extended_idx - 1)

            for t in environment.tanks:
                init_in_use = gpm.contains_salmon_variable(t, last_horizon_period).X > 0.5
                init_dep_p = 0
                init_weight = 0.0

                for dep_p in first_extended_period.deploy_periods:
                    if dep_p != first_extended_period:
                        pop_var = gpm.population_weight_variable(dep_p, t, first_extended_period).X
                        if pop_var > 0.5:
                            init_dep_p = dep_p.index - unextended_planning_periods
                            init_weight = pop_var

                if init_in_use or init_weight > 0.0:
                    next_init_pop = { "tank": t.index }
                    if init_weight > 0.0:
                        next_init_pop["deploy_period"] = init_dep_p
                        next_init_pop["weight"] = init_weight
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
        with open(outfile_path, "w") as outfile:
            outfile.write(json_object)

    if iteration.solution_output_file != None:
        solution_output_file_local = iteration.solution_output_file.replace("%N", str(iteration.current_iteration))
        write_solution_file(os.path.join(file_dir, solution_output_file_local), environment, iteration.unextended_planning_years, gpm)

def read_iteration_setup(file_path: str) -> Iteration:

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

def write_solution_file(file_path: str, environment: Environment, planning_years: int, gpm: GurobiProblemGenerator) -> Iteration:

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
    last_plan_period = first_plan_period + 12 * planning_years - 1
    first_preplan_year = first_plan_year - first_plan_period // 12
    first_preplan_period = 0
    preplan_deploy_periods = []
    for p in environment.preplan_release_periods:
        preplan_deploy_periods.append(p.index)
    plan_deploy_periods = []
    for p in environment.plan_release_periods:
        if p.index <= last_plan_period:
            plan_deploy_periods.append(p.index)
    pre_planning_horizon = {"years": first_plan_year - first_preplan_year, "first_year": first_preplan_year, "first_period": first_preplan_period, "deploy_periods": preplan_deploy_periods}
    planning_horizon = {"years": planning_years, "first_year": first_plan_year, "first_period": first_plan_period, "deploy_periods": plan_deploy_periods}

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
        if dep_p.index <= last_plan_period and len(dep_p.deploy_periods) > 0 and dep_p.periods_after_deploy[0] == dep_p:
            for t in environment.tanks:
                var = gpm.population_weight_variable(dep_p, t, dep_p)
                if var.X > 0.5:
                    add_tank_cycle_start(prod_cyles_by_deploy, dep_p.index, module_by_tank[t.index], t.index, dep_p.index, "deploy")

    # Start of tank cycles initiated by transfer. Must also include those starting after normal planning horizon to recognize correct tank cycle when detecting those that completed after planning horizon end.
    if gpm.allow_transfer:
        for dep_p in environment.release_periods:
            if dep_p.index <= last_plan_period:
                for p in dep_p.transfer_periods:
                    for from_t in environment.tanks:
                        for to_t in from_t.transferable_to:
                            var = gpm.transfer_weight_variable(dep_p, from_t, to_t, p)
                            if var.X > 0.5:
                                add_tank_cycle_start(prod_cyles_by_deploy, dep_p.index, module_by_tank[to_t.index], to_t.index, p.index + 1, "transfer", from_t.index, var.X)

    # Add tank populations. Must also include those after normal planning horizon to recognize correct tank cycle when detecting those that completed after planning horizon end.
    for dep_p in environment.release_periods:
        if dep_p.index <= last_plan_period:
            for p in dep_p.periods_after_deploy:
                for t in environment.tanks:
                    var = gpm.population_weight_variable(dep_p, t, p)
                    if var.X > 0.5:
                        tank_cycle = get_tank_cycle(prod_cyles_by_deploy, dep_p.index, module_by_tank[t.index], t.index, p.index)
                        add_tank_cycle_weight(tank_cycle, p.index, var.X)

    # Add extractions
    for dep_p in environment.release_periods:
        if dep_p.index <= last_plan_period:
            for extr_idx in range(2):
                extract_periods = dep_p.postsmolt_extract_periods if extr_idx == 0 else dep_p.harvest_periods
                for p in extract_periods:
                    for t in environment.tanks:
                        var = gpm.extract_weight_variable(dep_p, t, p)
                        if var.X > 0.5:
                            tank_cycle = get_tank_cycle(prod_cyles_by_deploy, dep_p.index, module_by_tank[t.index], t.index, p.index)
                            if p.index <= last_plan_period:
                                tank_cycle["end_period"] = p.index
                                tank_cycle["end_cause"] = "post_smolt" if extr_idx == 0 else "harvest"
                            else:
                                tank_cycle["end_period"] = last_plan_period
                                tank_cycle["end_cause"] = "planning_horizon_extension"

    production_cycles = []
    for dep_p_idx, mod_deploy_cycles in sorted(prod_cyles_by_deploy.items()):
        for mod_idx, cycle_tank_cycles in sorted(mod_deploy_cycles.items()):
            tank_cycles = []
            for tank_idx, tank_cycles_for_tank in sorted(cycle_tank_cycles.items()):
                for start_p_idx, tank_cycle in sorted(tank_cycles_for_tank.items()):
                    if start_p_idx <= last_plan_period:
                        # In final output, only include periods inside orinary planning horizon.
                        relevant_periods = []
                        for biomass_data in tank_cycle["period_biomasses"]:
                            if biomass_data["period"] <= last_plan_period:
                                relevant_periods.append(biomass_data)
                        tank_cycle["period_biomasses"] = relevant_periods
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



if __name__ == "__main__":
    file_path = sys.argv[1]
    objective = ObjectiveProfile.PROFIT
    allow_transfer = True
    add_symmetry_breaks = True

    opt_arguments = sys.argv[2:]
    options = "o:s:t:"
    long_options = ["Objective=", "Symmetry_break=", "Transfer="]

    try:
        arguments, values = getopt.getopt(opt_arguments, options, long_options)

        for argument, value in arguments:

            if argument in ("-o", "--Objective"):
                objective = parse_objective(value, ObjectiveProfile.PROFIT)
            
            elif argument in ("-s", "--Symmetry_break"):
                add_symmetry_breaks = parse_bool(value, True)

            elif argument in ("-t", "--Transfer"):
                allow_transfer = parse_bool(value, True)

        run_iteration(file_path, objective, allow_transfer, add_symmetry_breaks)

    except getopt.error as err:
        print(str(err))
