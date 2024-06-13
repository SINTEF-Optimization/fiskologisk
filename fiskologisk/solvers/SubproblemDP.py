from dataclasses import dataclass
import json
from math import prod
import time
from typing import List, Optional, Tuple
import dp_heur
from fiskologisk.domain.Environment import Environment

#
# This is a solver for the single-module full-horizon planning problem
# that calls out to a library implemented in Rust. The Rust module `dp_heur`
# must be installed with maturin, e.g. `maturin develop --release`, for this
# to work. The approach is to discretize biomass into a fixed number of levels
# and do dynamic programming over the resulting state space graph.
#


@dataclass
class DPSolution:
    objective_value: float
    period_tanks: List[Tuple[int, int]]


def solve_dp(
    environment: Environment,
    module_idx: int,
    period_biomass_duals: dict[int, float],
    yearly_production_duals: dict[int, float],
    bins: int,
) -> Optional[DPSolution]:
    #
    # Check some problem definition limits for the algorithm.
    #

    module_tanks = environment.modules[module_idx].tanks
    if len(module_tanks) >= 6:
        print("Warning: DP heuristic solver maybe unsuitable with a large number of tanks in a module")

    planning_start_time = min(p.index for p in environment.periods)
    planning_end_time = max(p.index for p in environment.periods)
    assert len(set(p.index for p in environment.periods)) == planning_end_time - planning_start_time + 1

    initial_tanks_in_use = [tank for tank in module_tanks if tank.initial_use and tank.initial_weight >= 0.01]

    initial_biomass = sum(tank.initial_weight for tank in initial_tanks_in_use)
    initial_tanks_cleaning = [tank for tank in module_tanks if tank.initial_use and tank.initial_weight < 0.01]

    min_initial_age = min(
        (planning_start_time - tank.initial_deploy_period for tank in initial_tanks_in_use), default=0
    )

    max_initial_age = max(
        (planning_start_time - tank.initial_deploy_period for tank in initial_tanks_in_use), default=0
    )

    assert all(planning_start_time - tank.initial_deploy_period > 0 for tank in initial_tanks_in_use)
    if min_initial_age != max_initial_age:
        print("Warning: initial tank deploy period", [tank.initial_deploy_period for tank in initial_tanks_in_use])
    assert min_initial_age == max_initial_age

    initial_age = min_initial_age

    min_tank_volume = min(1.0 / t.inverse_volume for t in module_tanks)
    max_tank_volume = max(1.0 / t.inverse_volume for t in module_tanks)

    if max_tank_volume / min_tank_volume >= 1.2:
        raise Exception("DP heuristic solver does not support uneven tank sizes")

    #
    # Compute problem parameters in algorithm-specific format for DP heuristic.
    #

    avg_tank_volume = sum(1.0 / t.inverse_volume for t in module_tanks) / len(module_tanks)

    period_production_duals = {}
    for year, price in yearly_production_duals.items():
        for env_year in environment.years:
            if env_year.year == year:
                for period in env_year.periods:
                    assert period.index not in period_production_duals
                    period_production_duals[period.index] = price

    # A map from deploy period and age to price,
    # where the price is the integral of the probability distribution of
    # weight classes multiplied by the post-smolt price for that weight class.
    # NOTE: post-smolt prices for a weight class are not time-variant,
    #       (though the problem definition here supports it)
    post_smolt_sell_price = {
        dep_p.index: {
            p.index: 1.0
            * (
                -period_production_duals[p.index]
                + sum(
                    frac * cls.post_smolt_revenue
                    for cls, frac in zip(
                        environment.weight_classes,
                        dep_p.periods_after_deploy_data[p.index].weight_distribution,
                    )
                )
            )
            for p in dep_p.postsmolt_extract_periods
        }
        for dep_p in environment.release_periods
    }

    # Same, but the harvest yield (being <100%) is included in the price.
    harvest_sell_price = {
        dep_p.index: {
            p.index: -period_production_duals[p.index]
            + environment.parameters.harvest_yield
            * sum(
                frac * cls.harvest_revenue
                for cls, frac in zip(
                    environment.weight_classes,
                    dep_p.periods_after_deploy_data[p.index].weight_distribution,
                )
            )
            for p in dep_p.harvest_periods
        }
        for dep_p in environment.release_periods
    }

    biomass_costs = {
        dep_p.index: {
            p.index: dep_p.periods_after_deploy_data[p.index].oxygen_cost
            + environment.parameters.marginal_increase_pumping_cost
            + period_biomass_duals[p.index]
            for p in dep_p.periods_after_deploy
        }
        for dep_p in environment.release_periods
    }

    biomass_costs_feed = {
        dep_p.index: {p.index: dep_p.periods_after_deploy_data[p.index].feed_cost for p in dep_p.periods_after_deploy}
        for dep_p in environment.release_periods
    }

    transfer_periods = {
        dep_p.index: {p.index: True for p in dep_p.transfer_periods} for dep_p in environment.release_periods
    }

    loss_factor = 1.0 - environment.parameters.monthly_loss

    growth_factors = {
        dep_p.index: {
            p.index: loss_factor * dep_p.periods_after_deploy_data[p.index].growth_factor
            for p in dep_p.periods_after_deploy
        }
        for dep_p in environment.release_periods
    }

    transfer_growth_factors = {
        dep_p.index: {
            p.index: loss_factor * dep_p.periods_after_deploy_data[p.index].transferred_growth_factor
            for p in dep_p.periods_after_deploy
        }
        for dep_p in environment.release_periods
    }

    max_biomass_per_tank = environment.parameters.max_tank_density * avg_tank_volume

    max_module_use_length = max((
        p.index - dep_p.index
        for dep_p in environment.release_periods
        for p in dep_p.periods_after_deploy
    )) + 1

    minimum_growth_factor = min((
        loss_factor * dep_p.periods_after_deploy_data[p.index].growth_factor
        for dep_p in environment.release_periods
        for p in dep_p.periods_after_deploy
    ))

    maximum_growth_factor = max((
        loss_factor * dep_p.periods_after_deploy_data[p.index].growth_factor
        for dep_p in environment.release_periods
        for p in dep_p.periods_after_deploy
    ))

    print("MIN GROWTH",minimum_growth_factor)
    print("MAX GROWTH", maximum_growth_factor)

    problem_json = {
        # PARAMETERS
        "volume_bins": bins,
        "max_module_use_length": max_module_use_length,
        "num_tanks": len(module_tanks),
        "planning_start_time": planning_start_time,
        "planning_end_time": planning_end_time,
        "initial_biomass": initial_biomass,
        "initial_tanks_in_use": len(initial_tanks_in_use),
        "initial_tanks_cleaning": len(initial_tanks_cleaning),
        "initial_age": initial_age,
        "smolt_deploy_price": environment.parameters.smolt_price,
        "max_deploy": environment.parameters.max_deploy_smolt,
        "min_deploy": environment.parameters.min_deploy_smolt,
        "tank_const_cost": environment.parameters.min_tank_cost,
        "max_biomass_per_tank": max_biomass_per_tank,
        "minimum_growth_factor": minimum_growth_factor,
        "maximum_growth_factor": maximum_growth_factor,

        # TABLES INDEXED ON BY TUPLE (DEPLOY_TIME,AGE)
        # TODO: clean this up a bit by putting all the tables into `deploy_period_data`.

        "post_smolt_sell_price": post_smolt_sell_price,
        "harvest_sell_price": harvest_sell_price,
        "biomass_costs": biomass_costs,
        "biomass_costs_feed": biomass_costs_feed,
        "transfer_periods": transfer_periods,
        "monthly_growth_factors": growth_factors,
        "monthly_growth_factors_transfer": transfer_growth_factors,
        "deploy_period_data": {},
        "logarithmic_bins": False,
    }

    problem_str = json.dumps(problem_json)

    print("starting DP solve....")

    t0 = time.time()
    solution_jsonstr = dp_heur.solve_module_json(problem_str)
    t1 = time.time()

    print(f"DP heuristic solver finished in {t1-t0:.2f}")

    solution = json.loads(solution_jsonstr)

    objective_value = -solution["objective"]
    retval = DPSolution(
        objective_value,
        [(state["period"], state["num_tanks"]) for state in solution["states"]],
    )

    return retval
