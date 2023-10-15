from collections import defaultdict
import os
import sys
import json
import dp_heur
from math import prod
from GurobiProblemGenerator import ObjectiveProfile
from read_problem import read_core_problem
from run_iteration import read_iteration_setup


if __name__ == "__main__":
    file_path = sys.argv[1]
    file_dir = os.path.dirname(file_path)

    iteration = read_iteration_setup(file_path)
    environment = read_core_problem(file_dir, iteration.core_setup_file)
    environment.add_initial_populations(iteration.initial_populations)

    #
    # Check some problem definition limits for the algorithm.
    #

    if len(iteration.initial_populations) > 0:
        raise Exception("unimplemented")

    if len(environment.modules) != 1:
        raise Exception("DP heuristic solver supports only 1 module")

    if len(environment.tanks) >= 6:
        print(
            "Warning: DP heuristic solver maybe unsuitable with a large number of tanks in a module"
        )

    min_tank_volume = min(1.0 / t.inverse_volume for t in environment.tanks)
    max_tank_volume = max(1.0 / t.inverse_volume for t in environment.tanks)

    if max_tank_volume / min_tank_volume >= 1.2:
        raise Exception("DP heuristic solver does not support uneven tank sizes")

    #
    # Compute problem parameters in algorithm-specific format for DP heuristic.
    #

    avg_tank_volume = sum(1.0 / t.inverse_volume for t in environment.tanks) / len(
        environment.tanks
    )

    # A map from deploy period and age to price,
    # where the price is the integral of the probability distribution of
    # weight classes multiplied by the post-smolt price for that weight class.
    # NOTE: post-smolt prices for a weight class are not time-variant,
    #       (though the problem definition here supports it)
    post_smolt_sell_price = {
        dep_p.index: {
            p.index: sum(
                frac * cls.post_smolt_revenue
                for cls, frac in zip(
                    environment.weight_classes,
                    dep_p.periods_after_deploy_data[p.index].weight_distribution,
                )
            )
            for p in dep_p.postsmolt_extract_periods
        }
        for dep_p in environment.release_periods
    }

    # Same, but the harvest yield (being <100%) is included in the price.
    harvest_sell_price = {
        dep_p.index: {
            p.index: environment.parameters.harvest_yield
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

    # TODO: here we can add shadow prices for biomass in the period
    biomass_costs = {
        dep_p.index: {
            p.index: dep_p.periods_after_deploy_data[p.index].feed_cost
            + dep_p.periods_after_deploy_data[p.index].oxygen_cost
            + environment.parameters.marginal_increase_pumping_cost
            for p in dep_p.periods_after_deploy
        }
        for dep_p in environment.release_periods
    }

    transfer_periods = {
        dep_p.index: {p.index: True for p in dep_p.transfer_periods}
        for dep_p in environment.release_periods
    }

    growth_factors = {
        dep_p.index: {
            p.index: environment.parameters.monthly_loss
            * dep_p.periods_after_deploy_data[p.index].growth_factor
            for p in dep_p.periods_after_deploy
        }
        for dep_p in environment.release_periods
    }

    transfer_growth_factors = {
        dep_p.index: {
            p.index: environment.parameters.monthly_loss
            * dep_p.periods_after_deploy_data[p.index].transferred_growth_factor
            for p in dep_p.periods_after_deploy
        }
        for dep_p in environment.release_periods
    }

    accumulated_growth_factors = {
        dep_p.index: {
            p.index: prod(
                (
                    environment.parameters.monthly_loss
                    * dep_p.periods_after_deploy_data[prev_p.index].growth_factor
                    for prev_p in dep_p.periods_after_deploy
                    if prev_p.index <= p.index
                )
            )
            for p in dep_p.periods_after_deploy
        }
        for dep_p in environment.release_periods
    }

    
    accumulated_minimum_growth_factors = {
        dep_p.index: {
            p.index: min(
                (
                    dep_p.periods_after_deploy_data[
                        prev_p.index
                    ].transferred_growth_factor
                    / dep_p.periods_after_deploy_data[prev_p.index].growth_factor
                    for prev_p in dep_p.periods_after_deploy
                    if prev_p.index <= p.index
                ),
                default=1.0,
            )
            for p in dep_p.periods_after_deploy
            if p.index >= min((x.index for x in dep_p.transfer_periods), default=-1)
        }
        for dep_p in environment.release_periods
    }

    print("# Valid periods")
    for dep_p in environment.release_periods:
        print(" - Release at ", dep_p.index)
        for p in dep_p.periods_after_deploy:
            print("   - after:  ", p.index)


    problem_json = {

        # PARAMETERS
        "volume_bins": 25,
        "max_module_use_length": 25,
        "num_tanks": len(environment.tanks),
        "planning_start_time": min(p.index for p in environment.periods),
        "planning_end_time": max(p.index for p in environment.periods),
        "smolt_deploy_price": environment.parameters.smolt_price,
        "max_deploy_mass": environment.parameters.max_deploy_smolt,
        "min_deploy_mass": environment.parameters.min_deploy_smolt,
        "tank_const_cost": environment.parameters.min_tank_cost,
        "max_biomass_per_tank": environment.parameters.max_tank_density
        * avg_tank_volume,

        # TABLES INDEXED ON BY TUPLE (DEPLOY_TIME,AGE)
        "post_smolt_sell_price": post_smolt_sell_price,
        "harvest_sell_price": harvest_sell_price,
        "biomass_costs": biomass_costs,
        "transfer_periods": transfer_periods,
        "monthly_growth_factors": growth_factors,
        "monthly_growth_factors_transfer": transfer_growth_factors,
        "accumulated_growth_factors": accumulated_growth_factors,
        "accumulated_minimum_growth_factors": accumulated_minimum_growth_factors,
    }

    solution_jsonstr = dp_heur.solve_module_json(json.dumps(problem_json))
    solution = json.loads(solution_jsonstr)

    print("SOLUTION", solution)

    for action in solution.actions:
        print(f"Action: {action}")
