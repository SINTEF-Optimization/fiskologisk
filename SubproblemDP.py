import json
from math import prod
import time
from typing import Optional

import dp_heur
from Environment import Environment
from MasterColumn import MasterColumn


def solve_dp(
    environment: Environment,
    module_idx: int,
    period_biomass_duals: dict[int, float],
    yearly_production_duals: dict[int, float],
) -> Optional[MasterColumn]:
    #
    # Check some problem definition limits for the algorithm.
    #

    module_tanks = environment.modules[module_idx].tanks
    if len(module_tanks) >= 6:
        print("Warning: DP heuristic solver maybe unsuitable with a large number of tanks in a module")

    planning_start_time = min(p.index for p in environment.periods)
    planning_end_time = max(p.index for p in environment.periods)

    initial_biomass = sum(tank.initial_weight for tank in module_tanks)
    initial_tanks = sum(1 if tank.initial_use else 0 for tank in module_tanks)
    print([f"Initial deploy period {tank.index} {tank.initial_deploy_period}" for tank in module_tanks])

    min_initial_age = min(planning_start_time - tank.initial_deploy_period for tank in module_tanks)
    max_initial_age = max(planning_start_time - tank.initial_deploy_period for tank in module_tanks)

    assert all(planning_start_time - tank.initial_deploy_period > 0 for tank in module_tanks)
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

    assert(sum(period_production_duals.values()) < 1e-5)
    assert(sum(yearly_production_duals.values()) < 1e-5)
    assert(sum(period_biomass_duals.values()) < 1e-5)

    # A map from deploy period and age to price,
    # where the price is the integral of the probability distribution of
    # weight classes multiplied by the post-smolt price for that weight class.
    # NOTE: post-smolt prices for a weight class are not time-variant,
    #       (though the problem definition here supports it)
    post_smolt_sell_price = {
        dep_p.index: {
            p.index: 1.0*(-period_production_duals[p.index]
            + sum(
                frac * cls.post_smolt_revenue
                for cls, frac in zip(
                    environment.weight_classes,
                    dep_p.periods_after_deploy_data[p.index].weight_distribution,
                ))
            )
            for p in dep_p.postsmolt_extract_periods
        }
        for dep_p in environment.release_periods
    }

    # Same, but the harvest yield (being <100%) is included in the price.
    print("## harvest yield", environment.parameters.harvest_yield)
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
            + dep_p.periods_after_deploy_data[p.index].feed_cost
            for p in dep_p.periods_after_deploy
        }
        for dep_p in environment.release_periods
    }

    biomass_costs_survival = {
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

    accumulated_growth_factors = {
        dep_p.index: {
            p.index: prod(
                (
                    loss_factor * dep_p.periods_after_deploy_data[prev_p.index].growth_factor
                    for prev_p in dep_p.periods_after_deploy
                    if prev_p.index < p.index
                )
            )
            for p in dep_p.periods_after_deploy
        }
        for dep_p in environment.release_periods
    }

    accumulated_minimum_growth_adjustment_factors = {
        dep_p.index: {
            p.index: min(
                (
                    dep_p.periods_after_deploy_data[prev_p.index].transferred_growth_factor
                    / dep_p.periods_after_deploy_data[prev_p.index].growth_factor
                    for prev_p in dep_p.periods_after_deploy
                    if prev_p.index < p.index
                    if p.index >= min((x.index for x in dep_p.transfer_periods), default=-1)
                ),
                default=1.0,
            )
            for p in dep_p.periods_after_deploy
        }
        for dep_p in environment.release_periods
    }

    print(f"# Montlyloss = {loss_factor}")
    print("# Valid periods")
    for dep_p in environment.release_periods:
        print(" - Release at ", dep_p.index)
        for p in dep_p.periods_after_deploy:
            data = dep_p.periods_after_deploy_data[p.index]
            print(
                f"   - p={p.index} G={growth_factors[dep_p.index][p.index]} G^T={data.transferred_growth_factor}",
                f"accum_G={accumulated_growth_factors[dep_p.index][p.index]}",
                f"accum_G_min={accumulated_minimum_growth_adjustment_factors[dep_p.index][p.index]}",
                f"costs={biomass_costs[dep_p.index][p.index]}",
                "DEPLOY" if p.index == dep_p.index else "",
                f"POSTSMOLT p={post_smolt_sell_price[dep_p.index][p.index]}"
                if p in dep_p.postsmolt_extract_periods
                else "",
                f"HARVEST  p={harvest_sell_price[dep_p.index][p.index]}" if p in dep_p.harvest_periods else "",
                "TRANSFER" if p in dep_p.transfer_periods else "",
            )

    max_biomass_per_tank = environment.parameters.max_tank_density * avg_tank_volume
    print("## max biomass per tank", max_biomass_per_tank)
    print("## deploy limits", environment.parameters.min_deploy_smolt, environment.parameters.max_deploy_smolt)

    problem_json = {
        # PARAMETERS
        "volume_bins": 1000,
        "max_module_use_length": 25,
        "num_tanks": len(module_tanks),
        "planning_start_time": planning_start_time,
        "planning_end_time": planning_end_time,
        "initial_biomass": initial_biomass,
        "initial_tanks": initial_tanks,
        "initial_age": initial_age,
        "smolt_deploy_price": environment.parameters.smolt_price,
        "max_deploy": environment.parameters.max_deploy_smolt,
        "min_deploy": environment.parameters.min_deploy_smolt,
        "tank_const_cost": environment.parameters.min_tank_cost,
        "max_biomass_per_tank": max_biomass_per_tank,
        # TABLES INDEXED ON BY TUPLE (DEPLOY_TIME,AGE)
        # TODO: clean this up a bit by putting all the tables into `deploy_period_data`.
        "post_smolt_sell_price": post_smolt_sell_price,
        "harvest_sell_price": harvest_sell_price,
        "biomass_costs": biomass_costs,
        "biomass_costs_survival": biomass_costs_survival,
        "transfer_periods": transfer_periods,
        "monthly_growth_factors": growth_factors,
        "monthly_growth_factors_transfer": transfer_growth_factors,
        "accumulated_growth_factors": accumulated_growth_factors,
        "accumulated_minimum_growth_adjustment_factors": accumulated_minimum_growth_adjustment_factors,
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

    print("SOLUTION", solution)

    for action in solution["actions"]:
        print(f"Action: {action}")

    return None
