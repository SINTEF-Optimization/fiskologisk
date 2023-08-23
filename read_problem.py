import json
import csv
import os
from Environment import Environment
from Parameters import Parameters
from Module import Module
from Tank import Tank
from Year import Year
from Period import Period
from Period import PeriodAfterDeployData
from WeightClass import WeightClass
from weight_distribution import get_weight_distributions

def read_problem(file_path : str) -> Environment:
    """Reads the model used for building the MIP problem for landbased salmon farming

    args:
        - file_path : 'str' The path to the json-file setting up the model

    returns:
        An environment object with the model
    """

    file_dir = os.path.dirname(file_path)

    with open(file_path, "r") as input_file:
        data = json.load(input_file)
        environment = read_core_problem(file_dir, data["core_setup"])
        if "initial_tank_setup" in data:
            read_initial_tank_setup(environment, data["initial_tank_setup"])

    return environment

def read_core_problem(dir : str, local_file_path : str) -> Environment:
    """Reads the model with empty initial conditions used for building the MIP problem for landbased salmon farming

    args:
        - dir : 'str' The path to the directory of the json file refering to the file with the core model
        - local_file_path : 'str' The path (relative to dir) to the json-file setting up the core model

    returns:
        An environment object with the core model
    """

    environment = Environment()
    file_path = os.path.join(dir, local_file_path)
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
    """Reads the global parameters for the MIP model

    args:
        - env_params : 'Parameters' Where the global parameters are stored
        - param_json The deserialized json object with the parameters
    """

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
    """Reads the setup of modules and tanks for the MIP model

    args:
        - environment : 'Environment' The environment object the MIP problem is built from
        - modules_json The deserialized json object with the module setup
    """

    mod_type = modules_json["type"]

    if mod_type == "FourTank":
        modules = modules_json["modules"]
        tank_volume = modules_json["tank_volume"]
        add_four_tank_modules(environment, modules, 1.0 / tank_volume)

    else:
        raise ValueError("Unknown module setup type: " + mod_type)

def read_weight_classes(environment : Environment, w_classes_json) -> None:
    """Reads the setup of the weight classes for the MIP model, with weights, post-smolt revenues and harvest revenues.
    The weight class weigths are expected to be equally separated.
    Post-smolt revenue is given by a list of revenue pr individ for different weight intervals.
    Harvest revenue is given by a list of revenue pr kg for different weight intervals.

    args:
        - environment : 'Environment' The environment object the MIP problem is built from
        - w_classes_json The deserialized json object with the setup of the weight classes
    """

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

        # Post-smolt revenue is given at NOK pr individ in the table, therefore we divide with weight class weight to get NOK/kg
        environment.weight_classes.append(WeightClass(weight, post_smolt_revenue[post_smolt_idx][1] / weight, harvest_revenue_pr_kg[harvest_idx][1]))

def read_periods(environment : Environment, periods_json) -> None:
    """Reads the setup of the periods for the MIP model.
    The time span of periods consist of a planning horizon plus some extension, starting at March in a given year,
    and some deploy periods for a preplanning horizon before the planning horizon.
    The MIP problem will only solve the salmon setup in each tank in the (extended) planning horizon,
    while the preplanning horizon is only for giving the initial setup of the tanks at the beginning of the planning horizon.

    args:
        - environment : 'Environment' The environment object the MIP problem is built from
        - periods_json The deserialized json object with the setup of the periods
    """

    # The first year in the planning horizon. The first period in the planning horizon will be March of this yhear.
    first_planning_year = periods_json["first_planning_year"]
    # The number of periods in the planning horizon, including extended planning horizon.
    planning_periods = periods_json["planning_periods"]
    # The number of periods from the first possible period in the preplanning horizon to the first period in the planning horizon.
    # This will also be the index of the first period in the planning horizon since indexing starts at 0 in preplanning horizon.
    pre_planning_periods = periods_json["pre_planning_periods"]
    # The number of periods from the start of the planning horizon to the last deploy period
    latest_deploy = periods_json["latest_deploy"]
    # The list of month numbers within each year of the deploy periods
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

            if is_deploy:
                environment.release_periods.append(period)
                if is_planning:
                    environment.plan_release_periods.append(period)
                else:
                    environment.preplan_release_periods.append(period)

            if is_planning:
                if month_in_year == 0:
                    year = Year(year_idx)
                    environment.years.append(year)
                year.periods.append(period)
                environment.periods.append(period)

        month_in_year += 1
        if month_in_year == 12:
            month_in_year = 0
            year_idx += 1

def read_post_deploy_relations(environment : Environment, file_dir : str, post_deploy_json) -> None:
    """Builds the necessary connections and data for periods with salmon and the periods when the salmon was deployed.
    This will connect possible transfer, harvest and post-smolt extraction periods after a deploy period,
    and store data about weight, growth factors, costs and weight class distribujtions in a period.

    args:
        - environment : 'Environment' The environment object the MIP problem is built from
        - dir : 'str' The path to the directory of the json source file of the setup
        - post_deploy_json The deserialized json object with the setup for building the connections and data
    """

    # Weight intervals for when salmon can be transfered, extracted as post-smolt and harvested
    transfer_weight = post_deploy_json["transfer_weight"]
    min_transfer_weight = transfer_weight["minimum"]
    max_transfer_weight = transfer_weight["maximum"]
    post_smolt_weight = post_deploy_json["post_smolt_weight"]
    min_post_smolt_weight = post_smolt_weight["minimum"]
    max_post_smolt_weight = post_smolt_weight["maximum"]
    harvest_weight = post_deploy_json["harvest_weight"]
    min_harvest_weight = harvest_weight["minimum"]
    max_harvest_weight = harvest_weight["maximum"]

    # Variance in the normal distribution function of individual salmon weights
    weight_variance_portion = post_deploy_json["weight_variance_portion"]

    # Table of expected weights (kg), feed costs (NOK/kg) and oxygen consumptions (g oxygen/kg salmon) depending on deploy period and number of periods since deploy
    expected_weights = read_csv_table(file_dir, post_deploy_json["expected_weights_file"])
    feed_costs = read_csv_table(file_dir, post_deploy_json["feed_costs_file"])
    oxygen_price = post_deploy_json["oxygen_price"]
    oxygen_consumptions = read_csv_table(file_dir, post_deploy_json["oxygen_consumption_file"])

    # Build tables of distributions into weight classes for different deploy months
    weight_distributions = get_weight_distributions(environment.weight_classes, expected_weights, weight_variance_portion, max_harvest_weight)

    # Connect deploy periods with production periods afterwards
    for deploy_period in environment.release_periods:
        deploy_month = deploy_period.month
        max_since_deploy = len(weight_distributions[deploy_month])

        # First find index of last possible extraction period
        last_extract_index = -1
        for period in environment.periods:
            since_deploy = period.index - deploy_period.index
            if since_deploy >= 0 and since_deploy < max_since_deploy:
                expected_weight = expected_weights[deploy_month][since_deploy]

                can_extract_post_smolt = expected_weight > min_post_smolt_weight and expected_weight < max_post_smolt_weight
                can_harvest = expected_weight > min_harvest_weight and expected_weight < max_harvest_weight
                if can_extract_post_smolt or can_harvest:
                    last_extract_index = period.index

        for period in environment.periods:
            since_deploy = period.index - deploy_period.index
            if since_deploy >= 0 and since_deploy < max_since_deploy and period.index <= last_extract_index:

                expected_weight = expected_weights[deploy_month][since_deploy]
                feed_cost = feed_costs[deploy_month][since_deploy]
                oxygen_cost = oxygen_consumptions[deploy_month][since_deploy] * oxygen_price
                growth_factor = expected_weights[deploy_month][since_deploy + 1] / expected_weight
                transfer_growth_factor = 1.0 + 0.5 * (growth_factor - 1)
                weight_distribution = weight_distributions[deploy_month][since_deploy]
                period_after_deploy_data = PeriodAfterDeployData(period, expected_weight, feed_cost, oxygen_cost, growth_factor, transfer_growth_factor, weight_distribution)

                can_extract_post_smolt = expected_weight > min_post_smolt_weight and expected_weight < max_post_smolt_weight
                can_transfer = expected_weight > min_transfer_weight and expected_weight < max_transfer_weight
                can_harvest = expected_weight > min_harvest_weight and expected_weight < max_harvest_weight
                
                deploy_period.add_after_deploy(period, period_after_deploy_data, can_harvest or can_extract_post_smolt)
                if can_transfer:
                    deploy_period.add_transfer_period(period)
                if can_extract_post_smolt:
                    deploy_period.add_postsmolt_extract_period(period)
                if can_harvest:
                    deploy_period.add_harvest_period(period)

def read_csv_table(dir : str, local_file_path : str) -> list[list[float]]:
    """Reads a csv file and returns the entries with the first row and first column removed.
    The entries are expected to be floats, except for the first row and column.

        args:
        - dir : 'str' The path to the directory of the source file refering to the csv file
        - local_file_path : 'str' The path (relative to dir) to the csv file

    returns:
        The contents of the csv file, outermost list is the rows, innermost list is the columns
    """

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
    """Adds modules and tanks to the MIP model, where each module has four tanks, all tanks have the same size,
    and salmon transfers between tanks are 0 -> 1 -> 3 and 0 -> 2

    args:
        - environment : 'Environment' The environment object the MIP problem is built from
        - modules : 'int' The number of modules added
        - inv_tank_volume : 'float' 1.0 divided by the tank volume (1/m3)
    """

    for mod_idx in range(modules):
        module = Module(mod_idx)
        environment.modules.append(module)

        for tank_idx in range(4):
            tank = Tank(4 * mod_idx + tank_idx, inv_tank_volume)
            module.tanks.append(tank)
            environment.tanks.append(tank)

        module.connect_transfer_tanks(0, 1)
        module.connect_transfer_tanks(0, 2)
        module.connect_transfer_tanks(1, 3)

def read_initial_tank_setup(environment: Environment, tank_setups) -> None:
    """Reads the initial setup of the tanks starting with salmon in the planning horizon

    args:
        - environment : 'Environment' The environment object the MIP problem is built from
        - tank_setups The deserialized json object with the initial tanks setup
    """

    all_tanks = [ t for module in environment.modules for t in module.tanks ]    

    for tank_setup in tank_setups["initial_tank_setup"]:
        tank_index = tank_setup["tank"]
        deploy_period_index = tank_setup["deploy_period"]
        weight = tank_setup["weight"]

        tank = next(t for t in environment.tanks if t.index == tank_index)
        tank.initial_weight = weight
        tank.initial_deploy_period = deploy_period_index
