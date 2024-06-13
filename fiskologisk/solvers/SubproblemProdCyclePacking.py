from dataclasses import dataclass, field
from math import prod
import time
from typing import Any, Dict, List, Optional, Set, Tuple
import gurobipy as gp
from fiskologisk.domain.Environment import Environment
from fiskologisk.solvers.GurobiProblemGenerator import GurobiProblemGenerator
from fiskologisk.domain.Module import Module

#
# PACKING OF PRODUCTION CYCLES
#
# This is an experiment from 2023-12-18 that was not successful, and is not
# currently used anywhere.
#
# This solves the single-module full-horizon production planning by planning all
# combinations of fixed deploy and harvest times and then selecting an optimal
# subset of non-overlapping production plans (by dynamic programming).
# 
#

@dataclass
class DPSolution:
    objective_value: float
    period_tanks: List[Tuple[int, int]]


@dataclass
class Node:
    time: int
    deploy_period: int
    cost: float
    prev: Any = field(repr=False)


@dataclass
class PeriodInfo:
    growth_periods: Set[int]
    extract_periods: Set[int]


def solve_prod_cycle_packing(
    module: Module,
    problem_generator: GurobiProblemGenerator,
    model: gp.Model,
) -> Optional[DPSolution]:
    
    t0 = time.time()
    n_evals = 0

    problem_generator.remove_initial_value_constraints(model)

    environment = problem_generator.environment
    initial_deploy_period = set(tank.initial_deploy_period for tank in module.tanks if tank.initial_weight > 1e-5)
    assert len(initial_deploy_period) == 0 or len(initial_deploy_period) == 1
    initial_deploy_period = -1 if len(initial_deploy_period) == 0 else next(iter(initial_deploy_period))
    assert initial_deploy_period != 0

    planning_start_time = min(p.index for p in environment.periods)
    planning_end_time = max(p.index for p in environment.periods)

    prev_time = planning_start_time if initial_deploy_period > 0 else planning_start_time - 1
    prev_states = {initial_deploy_period: Node(prev_time, initial_deploy_period, 0.0, None)}

    release_periods: Dict[int, PeriodInfo] = {
        p.index: PeriodInfo(
            set(other_p.index for other_p in p.periods_after_deploy),
            set(other_p.index for other_p in p.extract_periods),
        )
        for p in environment.release_periods
    }

    def get_cost(p1, p2) -> float:
        nonlocal n_evals
        n_evals += 1
        # print("GET COST ", p1, p2)
        use_initial_values = p1 <= planning_start_time
        if use_initial_values:
            problem_generator.add_initial_value_constraints(model, module.index)

        constraints = []
        for period in environment.periods:
            if period.index < p1 or period.index > p2:
                # print("PERIOD ", period.index, " is outside", p1, p2)
                for t in module.tanks:
                    constraints.append(
                        model.addConstr(
                            problem_generator.contains_salmon_variable(t, period) == 0,
                            name=f"lock_noproduction_t{t.index}_p{period.index}",
                        )
                    )

            # TODO test if it's faster to force deployment in `p1`
            # TODO test if it's faster to force salmon in tanks in periods inside the interval

        model.optimize()
        if model.status != gp.GRB.OPTIMAL:
            model.computeIIS()
            model.write("iis.ilp")
            raise Exception()
        value = model.ObjVal
        problem_generator.remove_constraints(model, constraints)

        if use_initial_values:
            problem_generator.remove_initial_value_constraints(model)
        return value

    def succ(node: Node):
        if node.deploy_period == -1 and node.time + 1 in release_periods:
            yield Node(node.time + 1, node.time + 1, 0.0, node)
        if node.deploy_period >= 0:
            period_info = release_periods[node.deploy_period]
            # print("  PINFO", period_info)
            if node.time in period_info.extract_periods:
                cost = get_cost(node.deploy_period, node.time)
                yield Node(node.time + 1, -1, cost, node)

            if node.time + 1 in period_info.growth_periods:
                yield Node(node.time + 1, node.deploy_period, 0.0, node)

    n_nodes = 1
    while prev_time < planning_end_time:
        next_states: Dict[int, Node] = {}
        next_time = prev_time + 1
        for prev_node in prev_states.values():
            # print("PREV", prev_node)
            for next_node in succ(prev_node):
                # print("ARC", prev_node, next_node)
                if (
                    next_node.deploy_period not in next_states
                    or next_states[next_node.deploy_period].cost > next_node.cost
                ):
                    next_states[next_node.deploy_period] = next_node
                    n_nodes += 1

        prev_states = next_states
        prev_time = next_time

    final_states: Dict[int, Node] = prev_states
    assert len(final_states) > 0

    final_state = min(final_states.values(), key=lambda n: n.cost)
    print("AGEDP COST ", final_state.cost)


    problem_generator.add_initial_value_constraints(model, module.index)

    print("TIME", time.time()-t0)
    print("NODES", n_nodes)
    print("MIPS", n_evals)

    raise Exception("implementation unfinished")
