import json
import csv
import os
from Environment import Environment
from Parameters import Parameters
from Module import Module
from Tank import Tank
from Year import Year
from Period import Period
from PeriodAfterDeploy import PeriodAfterDeploy
from WeightClass import WeightClass
from weight_distribution import get_weight_distributions

def read_problem(file_path : str) -> Environment:

    environment = Environment()
    file_dir = os.path.dirname(file_path)

    with open(file_path, "r") as input_file:
        data = json.load(input_file)

        read_parameters(environment.parameters, data["parameters"])
        read_modules(environment, data["modules"])
        read_weight_classes(environment, data["weight_classes"])
        read_periods(environment, data["periods"])
        read_post_deploy_relations(environment, file_dir, data["post_deploy"])

    return environment

def read_parameters(env_params : Parameters, param_json) -> None:
    env_params.smolt_price = param_json["smolt_price"]
    env_params.min_tank_cost = param_json["min_tank_cost"]
    env_params.marginal_increase_pumping_cost = param_json["marginal_increase_pumping_cost"]
    env_params.max_deploy_smolt = param_json["max_deploy_smolt"]
    env_params.min_deploy_smolt = param_json["min_deploy_smolt"]
    env_params.max_extract_weight = param_json["max_extract_weight"]
    env_params.max_transfer_weight = param_json["max_transfer_weight"]
    env_params.min_transfer_weight = param_json["min_transfer_weight"]
    env_params.max_tank_density = param_json["max_tank_density"]
    env_params.max_total_biomass = param_json["max_total_biomass"]
    env_params.max_yearly_production = param_json["max_yearly_production"]
    env_params.monthly_loss = param_json["monthly_loss"]
    env_params.harvest_yield = param_json["harvest_yield"]

def read_modules(environment : Environment, modules_json) -> None:

    mod_type = modules_json["type"]

    if mod_type == "FourTank":
        modules = modules_json["modules"]
        tank_volume = modules_json["tank_volume"]
        add_four_tank_modules(environment, modules, 1.0 / tank_volume)

    else:
        raise ValueError("Unknown module setup type: " + mod_type)

def read_weight_classes(environment : Environment, w_classes_json) -> None:

    min_weight = w_classes_json["min_weight"]
    weight_step = w_classes_json["weight_step"]
    classes = w_classes_json["classes"]
    post_smolt_revenue = w_classes_json["post_smolt_revenue"]
    harvest_revenue_pr_kg = w_classes_json["harvest_revenue_pr_kg"]
    post_smolt_idx = 0
    harvest_idx = 0

    for i in range(classes):
        weight = min_weight + i * weight_step
        while post_smolt_idx + 1 < len(post_smolt_revenue) and post_smolt_revenue[post_smolt_idx + 1][0] <= weight:
            post_smolt_idx += 1
        while harvest_idx + 1 < len(harvest_revenue_pr_kg) and harvest_revenue_pr_kg[harvest_idx + 1][0] <= weight:
            harvest_idx += 1
        environment.weight_classes.append(WeightClass(weight, post_smolt_revenue[post_smolt_idx][1] / weight, harvest_revenue_pr_kg[harvest_idx][1]))

def read_periods(environment : Environment, periods_json) -> None:

    first_planning_year = periods_json["first_planning_year"]
    planning_periods = periods_json["planning_periods"]
    pre_planning_periods = periods_json["pre_planning_periods"]
    latest_deploy = periods_json["latest_deploy"]
    deploy_months = periods_json["deploy_months"]

    dy = (pre_planning_periods + 11) // 12
    month_in_year = 12 * dy - pre_planning_periods
    year_idx = first_planning_year - dy
    year = None

    for month_idx in range(pre_planning_periods + planning_periods):
        is_planning = month_idx >= pre_planning_periods
        is_deploy = month_in_year in deploy_months and month_idx <= latest_deploy + pre_planning_periods

        if is_planning or is_deploy:
            period = Period(month_idx, month_in_year, is_deploy, is_planning)

            if is_planning:
                if month_in_year == 0:
                    year = Year(year_idx)
                    environment.years.append(year)
                year.periods.append(period)
                environment.periods.append(period)
                if is_deploy:
                    environment.release_periods.append(period)
            else:
                environment.preplan_release_periods.append(period)

        month_in_year += 1
        if month_in_year == 12:
            month_in_year = 0
            year_idx += 1

def read_post_deploy_relations(environment : Environment, file_dir : str, post_deploy_json) -> None:
    transfer_weight = post_deploy_json["transfer_weight"]
    min_transfer_weight = transfer_weight["minimum"]
    max_transfer_weight = transfer_weight["maximum"]
    post_smolt_weight = post_deploy_json["post_smolt_weight"]
    min_post_smolt_weight = post_smolt_weight["minimum"]
    max_post_smolt_weight = post_smolt_weight["maximum"]
    harvest_weight = post_deploy_json["harvest_weight"]
    min_harvest_weight = harvest_weight["minimum"]
    max_harvest_weight = harvest_weight["maximum"]
    weight_variance_portion = post_deploy_json["weight_variance_portion"]
    expected_weights = read_csv_table(file_dir, post_deploy_json["expected_weights_file"])
    feed_costs = read_csv_table(file_dir, post_deploy_json["feed_costs_file"])

    # Build tables of distributions into weight classes for different deploy months
    weight_distributions = get_weight_distributions(environment.weight_classes, expected_weights, weight_variance_portion, max_harvest_weight)

    # Connect deploy periods with production periods afterwards
    for deploy_period in environment.preplan_release_periods + environment.release_periods:
        deploy_month = deploy_period.month
        max_since_deploy = len(weight_distributions[deploy_month])
        for period in environment.periods:
            since_deploy = period.index - deploy_period.index
            if since_deploy >= 0 and since_deploy < max_since_deploy:

                expected_weight = expected_weights[deploy_month][since_deploy]
                feed_cost = feed_costs[deploy_month][since_deploy]
                can_extract_post_smolt = expected_weight > min_post_smolt_weight and expected_weight < max_post_smolt_weight
                can_transfer = expected_weight > min_transfer_weight and expected_weight < max_transfer_weight
                can_harvest = expected_weight > min_harvest_weight and expected_weight < max_harvest_weight
                growth_factor = expected_weights[deploy_month][since_deploy + 1] / expected_weight
                transfer_growth_factor = 1.0 + 0.5 * (growth_factor - 1)
                
                period.deploy_periods.append(deploy_period)
                deploy_period.periods_after_deploy[period.index] = PeriodAfterDeploy(feed_cost, 0.0, growth_factor, expected_weight, transfer_growth_factor, weight_distributions[deploy_month][since_deploy])
                if can_harvest or can_extract_post_smolt:
                    period.deploy_periods_for_extract.append(deploy_period)
                    deploy_period.extract_periods.append(period)
                else:
                    deploy_period.nonextract_periods.append(period)
                if can_transfer:
                    period.deploy_periods_for_transfer.append(deploy_period)
                    deploy_period.transfer_periods.append(period)
                if can_extract_post_smolt:
                    deploy_period.postsmolt_extract_periods.append(period)
                if can_harvest:
                    deploy_period.harvest_periods.append(period)

def read_csv_table(dir : str, local_file_path : str) -> list[list[float]]:

    result = []

    file_path = os.path.join(dir, local_file_path)
    with open(file_path, "r") as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in list(csv_reader)[1:]:
            float_row = []
            for cell in row[1:]:
                float_row.append(float(cell))
            result.append(float_row)

    return result

def add_four_tank_modules(environment : Environment, modules : int, inv_tank_volume : float) -> None:

    for mod_idx in range(modules):
        module = Module(mod_idx)
        environment.modules.append(module)

        for tank_idx in range(4):
            tank = Tank(4 * mod_idx + tank_idx, inv_tank_volume)
            module.tanks.append(tank)

        module.connect_transfer_tanks(0, 1)
        module.connect_transfer_tanks(0, 2)
        module.connect_transfer_tanks(1, 3)