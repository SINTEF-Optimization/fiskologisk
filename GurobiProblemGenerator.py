from __future__ import annotations
import gurobipy as gp
from gurobipy import GRB
from enum import Enum
from SolutionProvider import SolutionProvider
from Environment import Environment
from Period import Period
from Year import Year
from Module import Module
from Tank import Tank
from MasterColumn import MasterColumn

class ObjectiveProfile(Enum):
    """
    Profile for setup of objective in MIP problem
    """

    PROFIT = 1
    """Maximize revenues minus costs (NOK)"""

    BIOMASS = 2
    """Maximize weight of extracted post-smolt and harvested salmon from production facility"""

class GurobiProblemGenerator(SolutionProvider):
    """
    MIP problem generator for landbased salmon farming.
    The generator takes in the setup of the production facility and time periods for the planning horizon,
    adds the variables and constraints and builds the objective function into a Gurobi model.
    The generator can be used both for generating the single MIP problem of the entire production facility,
    or for generating the column generation subproblem of a single module in the Dantzig-Wolfe decomposistion
    algorithm.
    """

    environment: Environment
    """Holds the model used for building the MIP problem"""

    objective_profile: ObjectiveProfile
    """The profile of the objective in the problem. Default is PROFIT"""

    allow_transfer: bool
    """Whether transfer between tanks in a module is allowed. Default is True"""

    add_symmetry_breaks: bool
    """Whether extra constraints for breaking symmetries should be added. Default is False"""

    max_single_modules: int
    """If positive, each module has maximum one production cycle, they are deployed in chronological order, and the number sets a maximum limit on the number of simultaneously active deployments.
    Only used in the entire production facility problem, not in a column generation subproblem."""

    extract_weight_variables: dict[(int, int, int), gp.Var]
    """The continous MIP variables for weight of salmon extracted from the tanks for post-smolt or harvesting. Key is deploy period, tank and extract period"""

    population_weight_variables: dict[(int, int, int), gp.Var]
    """The continous MIP variables for salmon weight in a tank at a period. Key is deploy period, tank and period the mass is given for"""

    transfer_weight_variables: dict[(int, int, int, int), gp.Var]
    """The continous MIP variables for weight of salmon transfered at a period. Key is deploy period, tank transferred from, tank transferred to and period when the transfer takes place"""

    contains_salmon_variables: dict[(int, int), gp.Var]
    """The binary MIP variables for whether tanks hold salmon at a given period. Key is tank and period"""

    smolt_deployed_variables: dict[(int, int), gp.Var]
    """The binary MIP variables for whether salmon has been deployed in a module at a given period. Key is module and deploy period within the planning horizon"""

    salmon_extracted_variables: dict[(int, int), gp.Var]
    """The binary MIP variables for whether salmon was extracted from a tank at the end of a given period. Key is tank and period"""

    salmon_transferred_variables: dict[(int, int), gp.Var]
    """The binary MIP variables for whether salmon was transferred to a tank at a given period. Key is tank transferred to and period"""

    module_active_variables: dict[(int, int), gp.Var]
    """The binary MIP variables for whether the module was active in a production cycle this or previous period. Key is module and deploy period within the planning horizon"""

    core_objective_expression: gp.LinExpr
    """The linear expression for the objective without added constraints weighted by dual values of the relaxed regulatory constraints. Only used in a column generation subproblem"""

    period_biomass_expressions: dict[int, gp.LinExpr]
    """The expressions for the total mass of salmon in the tanks at a period. Key is period"""

    yearly_production_expressions: dict[int, gp.LinExpr]
    """The expressions for the total mass of salmon extracted as post smolt or harvest within a production year. Key is year"""

    def __init__(self, environment: Environment, objective_profile: ObjectiveProfile = ObjectiveProfile.PROFIT, allow_transfer: bool = True, add_symmetry_breaks: bool = False, max_single_modules: int = 0) -> None:
        self.environment = environment
        self.objective_profile = objective_profile
        self.allow_transfer = allow_transfer
        self.add_symmetry_breaks = add_symmetry_breaks
        self.max_single_modules = max_single_modules

        self.extract_weight_variables = {}
        self.population_weight_variables = {}
        self.transfer_weight_variables = {}
        self.contains_salmon_variables = {}
        self.smolt_deployed_variables = {}
        self.salmon_extracted_variables = {}
        self.salmon_transferred_variables = {}
        self.module_active_variables = {}

        self.core_objective_expression = None
        self.period_biomass_expressions = {}
        self.yearly_production_expressions = {}

    def build_model(self) -> gp.Model:
        """Builds the MIP model

        returns:
            The generated MIP model
        """

        model = gp.Model()
        self.add_variables(model)
        self.add_objective(model)
        self.add_constraints(model)
        return model
    
    def build_module_subproblemn(self, module_idx: int) -> gp.Model:
        """Builds the column generationg sub problem of the given module

        args:
            - module_idx 'int' The index of the module to build the sub problem for

        returns:
            The generated MIP model
        """

        model = gp.Model()
        self.add_variables(model, module_idx)
        self.core_objective_expression = self.create_core_objective_expression(module_idx)
        self.add_constraints(model, module_idx)
        
        return model

    def get_master_column(self, module_idx: int, objective_value: float, best_sol: bool) -> MasterColumn:
        """Builds a column for the Master Problem from the current solution in the problem generated by this generator.
        Should only be called if this generator is used for a column generation subproblem, and after a solution is found.

        args:
            - module_idx 'int' The index of the module the subproblem is built for
            - objective_value 'float' The column objective value to be used in the Master Problem.
            - best_sol 'bool' If True, the column is generated from the best solution found. If False, the column is generated from the current selection among the last solutions found in the MIP branching algorithm.

        returns:
            The generated master column
        """

        # Continous variable: Extracted salmon from deploy period from tank at period
        extract_weight_values = {}
        for dep_p in self.environment.release_periods:
            for p in dep_p.extract_periods:
                for t in self.environment.get_tanks(module_idx):
                    key = (dep_p.index, t.index, p.index)
                    var = self.extract_weight_variables[key]
                    extract_weight_values[key] = var.X if best_sol else var.Xn

        # Continous variable: Population weight from deploy period in tank at period
        population_weight_values = {}
        for dep_p in self.environment.release_periods:
            for p in dep_p.periods_after_deploy:
                for t in self.environment.get_tanks(module_idx):
                    key = (dep_p.index, t.index, p.index)
                    var = self.population_weight_variables[key]
                    population_weight_values[key] = var.X if best_sol else var.Xn

        # Continous variable: Transferred salmon from deploy period from tank to tank in period
        transfer_weight_values = {}
        if self.allow_transfer:
            for dep_p in self.environment.release_periods:
                for p in dep_p.transfer_periods:
                    for from_t in self.environment.get_tanks(module_idx):
                        for to_t in from_t.transferable_to:
                            key = (dep_p.index, from_t.index, to_t.index, p.index)
                            var = self.transfer_weight_variables[key]
                            transfer_weight_values[key] = var.X if best_sol else var.Xn

        # Binary variable: Tank contains salmon in period
        contains_salmon_values = {}
        for t in self.environment.get_tanks(module_idx):
            for p in self.environment.periods:
                key = (t.index, p.index)
                var = self.contains_salmon_variables[key]
                contains_salmon_values[key] = round(var.X if best_sol else var.Xn)

        # Binary variable: Smolt is deployed in module in period
        smolt_deployed_values = {}
        for m in self.environment.get_modules(module_idx):
            for dep_p in self.environment.plan_release_periods:
                key = (m.index, dep_p.index)
                var = self.smolt_deployed_variables[key]
                smolt_deployed_values[key] = round(var.X if best_sol else var.Xn)

        # Binary variable: Salmon is extracted from tank in period
        salmon_extracted_values = {}
        for t in self.environment.get_tanks(module_idx):
            for p in self.environment.periods:
                key = (t.index, p.index)
                var = self.salmon_extracted_variables[key]
                salmon_extracted_values[key] = round(var.X if best_sol else var.Xn)

        # Binary variable: Salmon is transferred to tank in period
        salmon_transferred_values = {}
        if self.allow_transfer:
            for t in self.environment.get_tanks(module_idx):
                if len(t.transferable_from) > 0:
                    for p in self.environment.periods:
                        key = (t.index, p.index)
                        var = self.salmon_transferred_variables[key]
                        salmon_transferred_values[key] = round(var.X if best_sol else var.Xn)

        return MasterColumn(module_idx, objective_value, extract_weight_values, population_weight_values, transfer_weight_values, contains_salmon_values, smolt_deployed_values, salmon_extracted_values, salmon_transferred_values)

    def add_fixed_values(self, model: gp.Model, fixed_values_json) -> None:
        """Reads information on the fixed values to be set for some of the variables in the MIP problem for landbased salmon farming

        args:
            - fixed_values_json The deserialized json object with the setup fixed values
        """

        if "deploy_periods" in fixed_values_json:
            end_period = self.environment.periods[-1].index + 1
            for depl_p in fixed_values_json["deploy_periods"]:
                module_idx = depl_p["module"]
                first_period = -1
                active_periods = [False] * end_period
                for dep_period in depl_p["deploy_periods"]:
                    start_p = dep_period["start_period"]
                    self.add_fixed_deploy_period(model, module_idx, start_p)
                    if first_period == -1:
                        first_period = start_p

                    end_p = dep_period["end_period"]
                    for p_idx in range(start_p, end_p + 1):
                        active_periods[p_idx] = True

                if first_period != -1:
                    for p_idx in range(first_period, end_period):
                        if active_periods[p_idx]:
                            self.add_fixed_active_period(model, module_idx, p_idx)
                        else:
                            self.add_fixed_inactive_period(model, module_idx, p_idx)

    def add_variables(self, model: gp.Model, module_idx: int = -1) -> None:
        """Generates all variables for the MIP problem
        
        args:
            - model: 'gp.Model' The MIP model to add the variables into.
            - module_idx: 'int' The index of the module to build the subproblem for, or -1 for all modules.
        """

        # Continous variable: Extracted salmon from deploy period from tank at period
        self.extract_weight_variables = {}
        for dep_p in self.environment.release_periods:
            for p in dep_p.extract_periods:
                for t in self.environment.get_tanks(module_idx):
                    key = (dep_p.index, t.index, p.index)
                    var = model.addVar(name = "e_%s,%s,%s"%key)
                    self.extract_weight_variables[key] = var

        # Continous variable: Population weight from deploy period in tank at period
        self.population_weight_variables = {}
        for dep_p in self.environment.release_periods:
            for p in dep_p.periods_after_deploy:
                for t in self.environment.get_tanks(module_idx):
                    key = (dep_p.index, t.index, p.index)
                    var = model.addVar(name = "x_%s,%s,%s"%key)
                    self.population_weight_variables[key] = var

        # Continous variable: Transferred salmon from deploy period from tank to tank in period
        self.transfer_weight_variables = {}
        if self.allow_transfer:
            for dep_p in self.environment.release_periods:
                for p in dep_p.transfer_periods:
                    for from_t in self.environment.get_tanks(module_idx):
                        for to_t in from_t.transferable_to:
                            key = (dep_p.index, from_t.index, to_t.index, p.index)
                            var = model.addVar(name = "y_%s,%s,%s,%s"%key)
                            self.transfer_weight_variables[key] = var

        # Binary variable: Tank contains salmon in period
        self.contains_salmon_variables = {}
        for t in self.environment.get_tanks(module_idx):
            for p in self.environment.periods:
                key = (t.index, p.index)
                var = model.addVar(name = "alpha_%s,%s"%key, vtype = GRB.BINARY)
                self.contains_salmon_variables[key] = var

        # Binary variable: Smolt is deployed in module in period
        self.smolt_deployed_variables = {}
        for m in self.environment.get_modules(module_idx):
            for dep_p in self.environment.plan_release_periods:
                key = (m.index, dep_p.index)
                var = model.addVar(name = "delta_%s,%s"%key, vtype = GRB.BINARY)
                self.smolt_deployed_variables[key] = var

        # Binary variable: Salmon is extracted from tank in period
        self.salmon_extracted_variables = {}
        for t in self.environment.get_tanks(module_idx):
            for p in self.environment.periods:
                key = (t.index, p.index)
                var = model.addVar(name = "epsilon_%s,%s"%key, vtype = GRB.BINARY)
                self.salmon_extracted_variables[key] = var

        # Binary variable: Salmon is transferred to tank in period
        self.salmon_transferred_variables = {}
        if self.allow_transfer:
            for t in self.environment.get_tanks(module_idx):
                if len(t.transferable_from) > 0:
                    for p in self.environment.periods:
                        key = (t.index, p.index)
                        var = model.addVar(name = "sigma_%s,%s"%key, vtype = GRB.BINARY)
                        self.salmon_transferred_variables[key] = var

        # Binary variable: Module is active in period or previous period
        self.module_active_variables = {}
        if self.max_single_modules > 0:
            for m in self.environment.modules:
                for dep_p in self.environment.plan_release_periods:
                    key = (m.index, dep_p.index)
                    var = model.addVar(name = "phi_%s,%s"%key, vtype = GRB.BINARY)
                    self.module_active_variables[key] = var

    def add_objective(self, model: gp.Model) -> None:
        """Builds and sets the objective function of the MIP problem
        
        args:
            - model: 'gp.Model' The MIP model to set the objective function for
        """

        obj_expr = self.create_core_objective_expression()
        model.setObjective(obj_expr, GRB.MAXIMIZE)

    def set_subproblem_objective(self, model: gp.Model, period_biomass_dual_values: dict[int, float], yearly_production_dual_values: dict[int, float]) -> None:
        """Builds and sets the objective function of the column generation subproblem
        
        args:
            - model: 'gp.Model' The MIP model to set the objective function for
            - period_biomass_dual_values: 'dict[int, float]' The dual values from the master problem of the expressions for the total mass of salmon in the tanks at a period. Key is period
            - yearly_production_dual_values: 'dict[int, float]' The dual values from the master problem of the expressions for the total mass of salmon extracted as post smolt or harvest within a production year. Key is year
        """

        obj_expr = gp.LinExpr(self.core_objective_expression)

        if period_biomass_dual_values != None:
            for p in self.environment.periods:
                obj_expr -= period_biomass_dual_values[p.index] * self.period_biomass_expressions[p.index]

        if yearly_production_dual_values != None:
            for y in self.environment.years:
                obj_expr -= yearly_production_dual_values[y.year] * self.yearly_production_expressions[y.year]

        model.setObjective(obj_expr, GRB.MAXIMIZE)

    def create_core_objective_expression(self, module_idx: int = -1) -> gp.LinExpr:
        """Creates and returns the core expression for the objective function of the MIP problem.
        For a column generation problem, this is the same as the objective without added constraints weighted by dual values of the relaxed regulatory constraints
        
        args:
            - module_idx: 'int' The index of the module to build the subproblem for, or -1 for all modules.
        """

        match self.objective_profile:
            case ObjectiveProfile.PROFIT:
                return self.create_profit_objective_expression(module_idx)
            case ObjectiveProfile.BIOMASS:
                return self.create_biomass_objective_expression(module_idx)
            case _:
                raise ValueError("Unknown objective profile")

    def create_profit_objective_expression(self, module_idx: int = -1) -> gp.LinExpr:
        """Creates the expression for the profit maximizing objective function of the MIP problem
        
        args:
            - module_idx: 'int' The index of the module to build the subproblem for, or -1 for all modules.
        """

        obj_expr = gp.LinExpr()
        parameters = self.environment.parameters

        # Revenue for selling post-smolt and harvestable salmon
        weight_classes = self.environment.weight_classes
        for dep_p in self.environment.release_periods:
            for t in self.environment.get_tanks(module_idx):

                # Post-smolt
                for p in dep_p.postsmolt_extract_periods:
                    weight_distribution = dep_p.periods_after_deploy_data[p.index].weight_distribution
                    coef = 0
                    for idx in range(len(weight_classes)):
                        coef += weight_classes[idx].post_smolt_revenue * weight_distribution[idx]
                    obj_expr.addTerms(coef, self.extract_weight_variable(dep_p, t, p))

                # Harvest
                for p in dep_p.harvest_periods:
                    weight_distribution = dep_p.periods_after_deploy_data[p.index].weight_distribution
                    coef = 0
                    for idx in range(len(weight_classes)):
                        coef += weight_classes[idx].harvest_revenue * weight_distribution[idx]
                    coef *= parameters.harvest_yield
                    obj_expr.addTerms(coef, self.extract_weight_variable(dep_p, t, p))

        # Smolt cost
        for dep_p in self.environment.plan_release_periods:
            for t in self.environment.get_tanks(module_idx):
                obj_expr.addTerms(-parameters.smolt_price, self.population_weight_variable(dep_p, t, dep_p))

        # Feeding cost
        for dep_p in self.environment.release_periods:
            for t in self.environment.get_tanks(module_idx):

                # Population weight variable for all periods
                for p in dep_p.periods_after_deploy:
                    cost = dep_p.periods_after_deploy_data[p.index].feed_cost
                    obj_expr.addTerms(-cost, self.population_weight_variable(dep_p, t, p))

                # Extract weight variable for extract periods
                for p in dep_p.extract_periods:
                    cost = dep_p.periods_after_deploy_data[p.index].feed_cost
                    obj_expr.addTerms(cost, self.extract_weight_variable(dep_p, t, p))

        # Oxygen cost
        for dep_p in self.environment.release_periods:
            for p in dep_p.periods_after_deploy:
                for t in self.environment.get_tanks(module_idx):
                    cost = dep_p.periods_after_deploy_data[p.index].oxygen_cost
                    obj_expr.addTerms(-cost, self.population_weight_variable(dep_p, t, p))

        # Tank operation
        for p in self.environment.periods:
            for t in self.environment.get_tanks(module_idx):

                # Minimum cost for operating a tank
                obj_expr.addTerms(-parameters.min_tank_cost, self.contains_salmon_variable(t, p))

                for dep_p in p.deploy_periods:
                    obj_expr.addTerms(-parameters.marginal_increase_pumping_cost, self.population_weight_variable(dep_p, t, p))

        return obj_expr

    def create_biomass_objective_expression(self, module_idx: int = -1, neg_deploy_period: int = -1) -> gp.LinExpr:
        """Creates the expression for the extracted mass maximizing objective function of the MIP problem
        
        args:
            - module_idx: 'int' The index of the module to build the subproblem for, or -1 for all modules.
            - neg_deploy_period: 'int' If not -1, extracted mass of salmon deployed in this period is subtracted in the objective.
        """

        obj_expr = gp.LinExpr()

        for dep_p in self.environment.release_periods:
            coef = -1.0 if neg_deploy_period == dep_p.index else 1.0
            for p in dep_p.extract_periods:
                for t in self.environment.get_tanks(module_idx):
                    obj_expr.addTerms(coef, self.extract_weight_variable(dep_p, t, p))

        return obj_expr

    def calculate_core_objective(self, module_idx: int) -> float:
        """Calculates and returns the core objective of the currently selected solution to the MIP problem.
        For a column generation problem, this is the same as the objective without added constraints weighted by dual values of the relaxed regulatory constraints
        
        args:
            - module_idx: 'int' The index of the module to calculate the objective for
        """

        match self.objective_profile:
            case ObjectiveProfile.PROFIT:
                return self.calculate_profit_objective(module_idx)
            case ObjectiveProfile.BIOMASS:
                return self.calculate_biomass_objective(module_idx)
            case _:
                raise ValueError("Unknown objective profile")

    def calculate_profit_objective(self, module_idx: int) -> float:
        """Calculates and returns the profit maximizing objective of the currently selected solution to the MIP problem.
        
        args:
            - module_idx: 'int' The index of the module to calculate the objective for
        """

        obj = 0
        parameters = self.environment.parameters

        # Revenue for selling post-smolt and harvestable salmon
        weight_classes = self.environment.weight_classes
        for dep_p in self.environment.release_periods:
            for t in self.environment.get_tanks(module_idx):

                # Post-smolt
                for p in dep_p.postsmolt_extract_periods:
                    weight_distribution = dep_p.periods_after_deploy_data[p.index].weight_distribution
                    coef = 0
                    for idx in range(len(weight_classes)):
                        coef += weight_classes[idx].post_smolt_revenue * weight_distribution[idx]
                    obj += coef * self.extract_weight_variable(dep_p, t, p).Xn

                # Harvest
                for p in dep_p.harvest_periods:
                    weight_distribution = dep_p.periods_after_deploy_data[p.index].weight_distribution
                    coef = 0
                    for idx in range(len(weight_classes)):
                        coef += weight_classes[idx].harvest_revenue * weight_distribution[idx]
                    coef *= parameters.harvest_yield
                    obj += coef * self.extract_weight_variable(dep_p, t, p).Xn

        # Smolt cost
        for dep_p in self.environment.plan_release_periods:
            for t in self.environment.get_tanks(module_idx):
                obj -= parameters.smolt_price * self.population_weight_variable(dep_p, t, dep_p).Xn

        # Feeding cost
        for dep_p in self.environment.release_periods:
            for t in self.environment.get_tanks(module_idx):

                # Population weight variable for all periods
                for p in dep_p.periods_after_deploy:
                    cost = dep_p.periods_after_deploy_data[p.index].feed_cost
                    obj -= cost * self.population_weight_variable(dep_p, t, p).Xn

                # Extract weight variable for extract periods
                for p in dep_p.extract_periods:
                    cost = dep_p.periods_after_deploy_data[p.index].feed_cost
                    obj += cost * self.extract_weight_variable(dep_p, t, p).Xn

        # Oxygen cost
        for dep_p in self.environment.release_periods:
            for p in dep_p.periods_after_deploy:
                for t in self.environment.get_tanks(module_idx):
                    cost = dep_p.periods_after_deploy_data[p.index].oxygen_cost
                    obj -= cost * self.population_weight_variable(dep_p, t, p).Xn

        # Tank operation
        for p in self.environment.periods:
            for t in self.environment.get_tanks(module_idx):

                # Minimum cost for operating a tank
                obj -= parameters.min_tank_cost * self.contains_salmon_variable(t, p).Xn

                for dep_p in p.deploy_periods:
                    obj -= parameters.marginal_increase_pumping_cost * self.population_weight_variable(dep_p, t, p).Xn

        return obj

    def calculate_biomass_objective(self, module_idx: int) -> float:
        """Calculates and returns the extracted mass maximizing objective of the currently selected solution to the MIP problem.
        
        args:
            - module_idx: 'int' The index of the module to calculate the objective for
        """

        obj = 0

        for dep_p in self.environment.release_periods:
            for p in dep_p.extract_periods:
                for t in self.environment.get_tanks(module_idx):
                    obj += self.extract_weight_variable(dep_p, t, p).Xn

        return obj

    def add_constraints(self, model: gp.Model, module_idx: int = -1) -> None:
        """Adds all the constraints to the MIP problem
        
        args:
            - model: 'gp.Model' The MIP model to add the constraints into
            - module_idx: 'int' The index of the module to build the subproblem for, or -1 for all modules.
        """

        self.add_initial_value_constraints(model, module_idx)
        self.add_smolt_deployment_constraints(model, module_idx)
        self.add_extraction_constraints(model, module_idx)
        if self.allow_transfer:
            self.add_salmon_transfer_constraints(model, module_idx)
        self.add_salmon_density_constraints(model, module_idx)

        # TODO why not add the regulatory constraints in the B&P subproblem as well?
        #      This could fix the problem of non-convergence of the one-module problem.
        
        if module_idx == -1:
            self.add_regulatory_constraints(model)
        else:
            self.create_regulatory_expressions(module_idx)
        self.add_biomass_development_constraints(model, module_idx)
        self.add_improving_constraints(model, module_idx)
        if self.add_symmetry_breaks:
            self.add_symmetry_break_constraints(model, module_idx)
        if self.max_single_modules > 0:
            self.add_max_single_modules_constraints(model)

    def add_initial_value_constraints(self, model: gp.Model, module_idx: int = -1) -> None:
        """Adds the smolt deployment constraints to the MIP problem
        
        args:
            - model: 'gp.Model' The MIP model to add the constraints into
            - module_idx: 'int' The index of the module to build the subproblem for, or -1 for all modules.
        """

        first_p = self.environment.periods[0]
        for dep_p in first_p.deploy_periods:
            if dep_p != first_p:
                for t in self.environment.get_tanks(module_idx):
                    init_w = t.initial_weight if t.initial_weight > 0 and t.initial_deploy_period == dep_p.index else 0.0
                    model.addConstr(self.population_weight_variable(dep_p, t, first_p) == init_w, name = "initial_population_%s,%s"%(dep_p.index, t.index))

    def add_smolt_deployment_constraints(self, model: gp.Model, module_idx: int = -1) -> None:
        """Adds the smolt deployment constraints to the MIP problem
        
        args:
            - model: 'gp.Model' The MIP model to add the constraints into
            - module_idx: 'int' The index of the module to build the subproblem for, or -1 for all modules.
        """

        max_w = self.environment.parameters.max_deploy_smolt
        min_w = self.environment.parameters.min_deploy_smolt
        for dep_p in self.environment.plan_release_periods:
            first = dep_p == self.environment.periods[0]
            prev_p = None if first else next(pp for pp in self.environment.periods if pp.index == dep_p.index - 1)
            for m in self.environment.get_modules(module_idx):
                delta = self.smolt_deployed_variable(m, dep_p)
                deployed_smolt_expr = gp.LinExpr()

                # Only deploy smolt in module if all tanks were empty previous period (5.4, 6.3)
                for t in m.tanks:
                    deployed_smolt_expr.addTerms(1.0, self.population_weight_variable(dep_p, t, dep_p))
                    if not first:
                        model.addConstr(delta + self.contains_salmon_variable(t, prev_p) <= 1, name = "empty_deploy_tank_%s,%s,%s"%(m.index, t.index, dep_p.index))
                    elif t.initial_use:
                        model.addConstr(delta == 0, name = "empty_deploy_tank_%s,%s,%s"%(m.index, t.index, dep_p.index))

                # Set weight range for deployed smolt (5.3, 6.2)
                model.addConstr(min_w * delta <= deployed_smolt_expr, name = "min_smolt_dep_%s,%s"%(m.index, dep_p.index))
                model.addConstr(deployed_smolt_expr <= max_w * delta, name = "max_smolt_dep_%s,%s"%(m.index, dep_p.index))

    def add_extraction_constraints(self, model: gp.Model, module_idx: int = -1) -> None:
        """Adds the extraction constraints to the MIP problem
        
        args:
            - model: 'gp.Model' The MIP model to add the constraints into
            - module_idx: 'int' The index of the module to build the subproblem for, or -1 for all modules.
        """

        max_w = self.environment.parameters.max_extract_weight
        for t in self.environment.get_tanks(module_idx):
            for p in self.environment.periods:
                epsilon = self.salmon_extracted_variable(t, p)
                extr_w_expr = gp.LinExpr()

                # Set max limit for extracted salmon (5.5, 6.4)
                for dep_p in p.deploy_periods_for_extract:
                    extr_w_expr.addTerms(1.0, self.extract_weight_variable(dep_p, t, p))
                model.addConstr(extr_w_expr <= max_w * epsilon, name = "max_extract_%s,%s"%(t.index, p.index))

                # Empty tank after extraction (5.6, 6.5)
                if p != self.environment.periods[-1]:
                    next_p = next(np for np in self.environment.periods if np.index == p.index + 1)
                    model.addConstr(self.contains_salmon_variable(t, next_p) + epsilon <= 1, name = "empty_extract_tank_%s,%s"%(t.index, p.index))

    def add_salmon_transfer_constraints(self, model: gp.Model, module_idx: int = -1) -> None:
        """Adds the salmon transfer constraints to the MIP problem, only used when transfer is allowed
        
        args:
            - model: 'gp.Model' The MIP model to add the constraints into
            - module_idx: 'int' The index of the module to build the subproblem for, or -1 for all modules.
        """

        max_w = self.environment.parameters.max_transfer_weight
        min_w = self.environment.parameters.min_transfer_weight
        for p in self.environment.periods:
            first = p == self.environment.periods[0]
            prev_p = None if first else next(pp for pp in self.environment.periods if pp.index == p.index - 1)
            last = p == self.environment.periods[-1]
            next_p = None if last else next(np for np in self.environment.periods if np.index == p.index + 1)
            for t in self.environment.get_tanks(module_idx):
                if len(t.transferable_from) > 0:
                    sigma = self.salmon_transferred_variable(t, p)
                    min_w_expr = gp.LinExpr(min_w, sigma)
                    max_w_expr = gp.LinExpr(max_w, sigma)

                    # Only transfer to empty tanks (5.7, 6.6)
                    if not first:
                        model.addConstr(sigma + self.contains_salmon_variable(t, prev_p) <= 1, name = "transfer_to_empty_%s,%s"%(t.index, p.index))
                    elif t.initial_use:
                        model.addConstr(sigma == 0, name = "transfer_to_empty_%s,%s"%(t.index, p.index))

                    # Tanks can not receive and extract same period (5.8, 6.7)
                    model.addConstr(sigma + self.salmon_extracted_variable(t, p) <= 1, name = "no_extract_if_transfer_%s,%s"%(t.index, p.index))

                    for from_t in t.transferable_from:

                        # Do not empty tank transferred from (5.9, 6.8)
                        if not last:
                            model.addConstr(sigma - self.contains_salmon_variable(from_t, next_p) <= 0, name = "transfer_from_not_empty_%s,%s,%s"%(t.index, from_t.index, p.index))

                        # Set weight range for transferred salmon (5.10, 6.9)
                        transf_w_expr = gp.LinExpr()
                        for dep_p in p.deploy_periods_for_transfer:
                            transf_w_expr.addTerms(1.0, self.transfer_weight_variable(dep_p, from_t, t, p))

                        # model.addConstr(gp.quicksum([1,0 * self.transfer_weight_variable(dep_p, from_t, t, p) for dep_p in p.deploy_periods_for_transfer]) <= transf_w_expr, name = "min_transfer_%s,%s,%s"%(t.index, from_t.index, p.index))
                        model.addConstr(min_w_expr <= transf_w_expr, name = "min_transfer_%s,%s,%s"%(t.index, from_t.index, p.index))
                        model.addConstr(transf_w_expr <= max_w_expr, name = "max_transfer_%s,%s,%s"%(t.index, from_t.index, p.index))

    def add_salmon_density_constraints(self, model: gp.Model, module_idx: int = -1) -> None:
        """Adds the salmon density and tank activation constraints to the MIP problem
        
        args:
            - model: 'gp.Model' The MIP model to add the constraints into
            - module_idx: 'int' The index of the module to build the subproblem for, or -1 for all modules.
        """

        max_den = self.environment.parameters.max_tank_density
        for t in self.environment.get_tanks(module_idx):
            inv_vol = t.inverse_volume
            for p in self.environment.periods:

                # Set limit on tank density (5.11, 6.10)
                weight_expr = gp.LinExpr()
                for dep_p in p.deploy_periods:
                    weight_expr.addTerms(inv_vol, self.population_weight_variable(dep_p, t, p))
                if self.allow_transfer:
                    for dep_p in p.deploy_periods_for_transfer:
                        for from_t in t.transferable_from:
                            weight_expr.addTerms(inv_vol, self.transfer_weight_variable(dep_p, from_t, t, p))
                model.addConstr(weight_expr <= max_den * self.contains_salmon_variable(t, p), name = "max_density_%s,%s"%(t.index, p.index))

    def add_regulatory_constraints(self, model: gp.Model) -> None:
        """Adds the regulatory constraints to the MIP problem.
        Should not be called for the column generation sub problem
        
        args:
            - model: 'gp.Model' The MIP model to add the constraints into
        """

        # Set limit on total biomass each period (5.12)
        max_production_tanks = len(self.environment.tanks) if self.max_single_modules == 0 else self.max_single_modules * len(self.environment.modules[0].tanks)
        regulation_rescale = max_production_tanks / self.environment.parameters.tanks_in_regulations
        max_mass = self.environment.parameters.max_total_biomass * regulation_rescale
        for p in self.environment.periods:
            weight_expr = self.create_period_biomass_expression(p)
            model.addConstr(weight_expr <= max_mass, name = "max_biomass_%s"%p.index)

        # Set limit on yearly production (5.13)
        max_prod = self.environment.parameters.max_yearly_production * regulation_rescale
        for y in self.environment.years:
            extr_w_expr = self.create_yearly_production_expression(y)
            model.addConstr(extr_w_expr <= max_prod, name = "max_year_prod_%s"%y.year)

    def create_regulatory_expressions(self, module_idx: int) -> None:
        """Creates and stores the expressions for the relaxed regulatory constraints in the column generation subproblem.
        
        args:
            - model: 'gp.Model' The MIP model to add the constraints into
            - module_idx: 'int' The index of the module the subproblem is built for
        """
        self.period_biomass_expressions = {}
        for p in self.environment.periods:
            self.period_biomass_expressions[p.index] = self.create_period_biomass_expression(p, module_idx)

        self.yearly_production_expressions = {}
        for y in self.environment.years:
            self.yearly_production_expressions[y.year] = self.create_yearly_production_expression(y, module_idx)

    def create_period_biomass_expression(self, period: Period, module_idx: int = -1) -> gp.LinExpr:
        """Creates the expression for the total mass of salmon in the tanks at a period
        
        args:
            - period: 'Period' The period to build the expression for
            - module_idx: 'int' The index of the module of the tanks to sum the mass over, or -1 for all modules.
        """

        weight_expr = gp.LinExpr()
        for dep_p in period.deploy_periods:
            for t in self.environment.get_tanks(module_idx):
                weight_expr.addTerms(1.0, self.population_weight_variable(dep_p, t, period))
        return weight_expr

    def create_yearly_production_expression(self, year: Year, module_idx: int = -1) -> gp.LinExpr:
        """Creates the expression for the total mass of salmon extracted as post smolt or harvest within a production year
        
        args:
            - year: 'Year' The year to build the expression for
            - module_idx: 'int' The index of the module of the tanks to sum the mass over, or -1 for all modules.
        """

        extr_w_expr = gp.LinExpr()
        for p in year.periods:
            for dep_p in p.deploy_periods_for_extract:
                for t in self.environment.get_tanks(module_idx):
                    extr_w_expr.addTerms(1.0, self.extract_weight_variable(dep_p, t, p))
        return extr_w_expr

    def add_biomass_development_constraints(self, model: gp.Model, module_idx: int = -1) -> None:
        """Adds the biomass development constraints to the MIP problem
        
        args:
            - model: 'gp.Model' The MIP model to add the constraints into
            - module_idx: 'int' The index of the module to build the subproblem for, or -1 for all modules.
        """

        loss = self.environment.parameters.monthly_loss
        for dep_p in self.environment.release_periods:
            if len(dep_p.periods_after_deploy) > 0:
                for p in dep_p.periods_after_deploy[:-1]:
                    p_data = dep_p.periods_after_deploy_data[p.index]
                    growth_factor = (1.0 - loss) * p_data.growth_factor
                    transf_growth_factor = (1.0 - loss) * p_data.transferred_growth_factor

                    is_transfer = p in dep_p.transfer_periods
                    is_extract = p in dep_p.extract_periods

                    for t in self.environment.get_tanks(module_idx):
                        next_w_expr = gp.LinExpr()

                        # Add updated weight from salmon at start of period (5.14, 5.15, 5.16, 5.17, 6.11, 6.12, 6.13, 6.14)
                        next_w_expr.addTerms(growth_factor, self.population_weight_variable(dep_p, t, p))
                        
                        if is_transfer and self.allow_transfer:
                            # Subtract weight from salmon transfered out during period (5.15, 5.16, 6.12, 6.13)
                            for to_t in t.transferable_to:
                                next_w_expr.addTerms(-growth_factor, self.transfer_weight_variable(dep_p, t, to_t, p))

                            # Add weight from salmon transfered in during period (5.15, 5.16, 6.12, 6.13)
                            for from_t in t.transferable_from:
                                next_w_expr.addTerms(transf_growth_factor, self.transfer_weight_variable(dep_p, from_t, t, p))
                        
                        if is_extract:
                            # Subtract weight from salmon extracted during period (5.16, 5.17, 6.13, 6.14)
                            next_w_expr.addTerms(-growth_factor, self.extract_weight_variable(dep_p, t, p))
                        next_p = next(np for np in dep_p.periods_after_deploy if np.index == p.index + 1)
                        model.addConstr(self.population_weight_variable(dep_p, t, next_p) == next_w_expr, name = "next_period_mass_%s,%s,%s"%(dep_p.index, t.index, p.index))

                # All salmon in last possible period must be extracted (5.18, 6.15)
                last_p = dep_p.periods_after_deploy[-1]
                for t in self.environment.get_tanks(module_idx):
                    model.addConstr(self.population_weight_variable(dep_p, t, last_p) == self.extract_weight_variable(dep_p, t, last_p), name = "extract_last_period_%s,%s"%(dep_p.index, t.index))

    def add_improving_constraints(self, model: gp.Model, module_idx: int = -1) -> None:
        """Adds valid inequalities to the MIP problem that can improve the linear relaxation in the model
        
        args:
            - model: 'gp.Model' The MIP model to add the constraints into
            - module_idx: 'int' The index of the module to build the subproblem for, or -1 for all modules.
        """

        prev_p = self.environment.periods[0]
        for p in self.environment.periods[1:]:
            is_deploy = p in self.environment.plan_release_periods
            for m in self.environment.get_modules(module_idx):
                for t in m.tanks:

                    # Add improving constraint: Tank is empty if empty previous period and no transfer or deploy this period (5.20, 5.21, 6.17, 6.18)
                    expr = gp.LinExpr()
                    if is_deploy:
                        expr.addTerms(1, self.smolt_deployed_variable(m, p))
                    if self.allow_transfer and len(t.transferable_from) > 0:
                        expr.addTerms(1, self.salmon_transferred_variable(t, p))
                    expr.addTerms(1, self.contains_salmon_variable(t, prev_p))
                    expr.addTerms(-1, self.salmon_extracted_variable(t, prev_p))
                    model.addConstr(expr >= self.contains_salmon_variable(t, p), name = "no_transfer_deploy_%s,%s"%(p.index, t.index))
            prev_p = p

        for t in self.environment.get_tanks(module_idx):

            # Can not extract from empty tanks last period (similar to 5.20/5.21/6.17/6.18 for a period after planning horizon)
            model.addConstr(self.salmon_extracted_variable(t, prev_p) <= self.contains_salmon_variable(t, prev_p), name = "no_extract_from_empty_last_period_%s"%t.index)

        for dep_p in self.environment.plan_release_periods:
            first_extract_idx = dep_p.extract_periods[0].index
            up_to_extract = [p for p in dep_p.periods_after_deploy if p.index <= first_extract_idx ]
            for m in self.environment.get_modules(module_idx):
                delta = self.smolt_deployed_variable(m, dep_p)
                for p in up_to_extract:

                    # Add improving constraint: At least one tank in module has salmon from deploy up to first possible extract period (5.22, 6.19)
                    sum_alpha = gp.LinExpr()
                    for t in m.tanks:
                        sum_alpha.addTerms(1.0, self.contains_salmon_variable(t, p))
                    model.addConstr(gp.LinExpr(1.0, delta) <= sum_alpha, name = "before_first_extract__%s,%s,%s"%(m.index, dep_p.index, p.index))
                
                for t in m.tanks:

                    # Add improving constraint: Force alpha to be one from deploy to extraction: (5.23, 6.20)
                    expr_lhs_1 = gp.LinExpr()
                    expr_lhs_1.addTerms(2, delta)
                    expr_lhs_1.addConstant(-2)
                    expr_lhs_1.addTerms(1, self.contains_salmon_variable(t, dep_p))
                    for p_idx in range(1, len(dep_p.periods_after_deploy)):
                        p = dep_p.periods_after_deploy[p_idx]
                        expr_lhs_2 = gp.LinExpr()
                        for pp_idx in range(1, p_idx + 1):
                            pp = dep_p.periods_after_deploy[pp_idx]
                            if self.allow_transfer and len(t.transferable_from) > 0:
                                expr_lhs_2.addTerms(1, self.salmon_transferred_variable(t, pp))
                            if pp_idx != p_idx:
                                expr_lhs_2.addTerms(-1, self.salmon_extracted_variable(t, pp))
                        model.addConstr(expr_lhs_1 + expr_lhs_2 <= self.contains_salmon_variable(t, p), name = "force_alpha_%s,%s,%s"%(t.index, dep_p.index, p.index))

    def add_symmetry_break_constraints(self, model: gp.Model, module_idx: int = -1) -> None:
        """Adds extra constraints for breaking symmetries
        
        args:
            - model: 'gp.Model' The MIP model to add the constraints into
            - module_idx: 'int' The index of the module to build the subproblem for, or -1 for all modules.
        """

        mod_type = self.environment.parameters.modules_type

        if mod_type == "FourTanks":
            for dep_p in self.environment.plan_release_periods:
                for m in self.environment.get_modules(module_idx):
                    delta = self.smolt_deployed_variable(m, dep_p)
                    nmb_tanks = 4 if self.allow_transfer else 3
                    for t in m.tanks[:nmb_tanks]:
                        model.addConstr(delta <= self.contains_salmon_variable(t, dep_p), name = "force_deploy_tank_%s,%s"%(t.index, dep_p.index))

    def add_max_single_modules_constraints(self, model: gp.Model) -> None:
        """Adds extra constraints for modules with single production cycle
        
        args:
            - model: 'gp.Model' The MIP model to add the constraints into
        """

        for m in self.environment.modules:

            sum_delta = gp.LinExpr()
            num_tanks = len(m.tanks)
            is_last_mod = m == self.environment.modules[-1]
            next_m = None if is_last_mod else next(nm for nm in self.environment.modules if nm.index == m.index + 1)
            sum_delta_next = gp.LinExpr()

            for dep_p in self.environment.plan_release_periods:
                phi = self.module_active_variable(m, dep_p)
                delta = self.smolt_deployed_variable(m, dep_p)
                sum_delta.addTerms(1.0, delta)

                # Set value of phi variables
                if dep_p == self.environment.periods[0]:
                    for t in m.tanks:
                        if t.initial_use:
                            model.addConstr(phi == 1, name = "set_phi_first_period_%s"%m.index)
                            break
                    else:
                        model.addConstr(phi == delta, name = "set_phi_first_period_%s"%m.index)
                else:
                    prev_p = next (pp for pp in self.environment.periods if pp.index == dep_p.index - 1)
                    use_expr = gp.LinExpr()
                    use_expr.addTerms(1.0, delta)
                    for t in m.tanks:
                        use_expr.addTerms(1.0, self.contains_salmon_variable(t, prev_p))
                    model.addConstr(gp.LinExpr(1.0, phi) <= use_expr, name = "set_max_phi_%s,%s"%(m.index, dep_p.index))
                    model.addConstr(use_expr <= gp.LinExpr(num_tanks, phi), name = "set_min_phi_%s,%s"%(m.index, dep_p.index))

                if not is_last_mod:

                    # Ensure module is not deployed after next in list of modules
                    sum_delta_next.addTerms(1.0, self.smolt_deployed_variable(next_m, dep_p))
                    model.addConstr(sum_delta_next <= sum_delta, name = "modules_chronologically_%s,%s"%(m.index, dep_p.index))

            # Ensure module is only deployed once
            model.addConstr(sum_delta == 1, name = "module_deployed_once_%s"%m.index)

        for dep_p in self.environment.plan_release_periods:

            # Limit number of active modules each period
            sum_mod = gp.LinExpr()
            for m in self.environment.modules:
                sum_mod.addTerms(1.0, self.module_active_variable(m, dep_p))
            model.addConstr(sum_mod <= self.max_single_modules, name = "maximum_modules_%s"%dep_p.index)

    def add_fixed_deploy_period(self, model: gp.Model, m_idx: int, dep_p_idx: int) -> None:
        """Adds extra constraint forcing a module to be deployed at a given deploy period
        
        args:
            - model: 'gp.Model' The MIP model to add the constraint into
            - m_idx: 'int' The index of the module with a ficed deploy
            - dep_p_idx: 'int' The index of the deploy period when the module has a fixed deploy
        """

        model.addConstr(self.smolt_deployed_variables[(m_idx, dep_p_idx)] == 1.0, name = "fix_deploy_period_%s,%s"%(m_idx, dep_p_idx))

    def add_fixed_active_period(self, model: gp.Model, m_idx: int, p_idx: int) -> None:
        """Adds extra constraint forcing a module to have at least one tank with salmon at a given period
        
        args:
            - model: 'gp.Model' The MIP model to add the constraint into
            - m_idx: 'int' The index of the module
            - p_idx: 'int' The index of the period when at least one tank in the module has salmon
        """

        m = next(mm for mm in self.environment.modules if mm.index == m_idx)
        sum_alpha = gp.LinExpr()
        for t in m.tanks:
            sum_alpha.addTerms(1.0, self.contains_salmon_variables[(t.index, p_idx)])
        model.addConstr(sum_alpha >= 1.0, name = "fix_module_active_%s,%s"%(m_idx, p_idx))

    def add_fixed_inactive_period(self, model: gp.Model, m_idx: int, p_idx: int) -> None:
        """Adds extra constraints forcing a module to have no tanks with salmon at a given period
        
        args:
            - model: 'gp.Model' The MIP model to add the constraints into
            - m_idx: 'int' The index of the module
            - p_idx: 'int' The index of the period when no tanks in the module have salmon
        """

        m = next(mm for mm in self.environment.modules if mm.index == m_idx)
        for t in m.tanks:
            model.addConstr(self.contains_salmon_variables[(t.index, p_idx)] == 0.0, name = "fix_tank_inactive_%s,%s"%(t.index, p_idx))


    def lock_binaries(self, model: gp.Model, column: MasterColumn) -> list[gp.Constr]:
        """Adds constraints that lock all binary variables to their values in the given column from an earlier column generation.
        This will in principle make the problem into an LP

        args:
            - model: 'gp.Model' The MIP model to add the constraints into
            - column 'MasterColumn' The column holding the values to lock the binary variables to

        returns:
            The new added constraints
        """

        constraints = []
        for key, val in column.smolt_deployed_values.items():
            constraints.append(model.addConstr(self.smolt_deployed_variables[key] == val, name = "lock_smolt_deployed_" + str(key)))
        for key, val in column.salmon_extracted_values.items():
            constraints.append(model.addConstr(self.salmon_extracted_variables[key] == val, name = "lock_salmon_extracted_" + str(key)))
        for key, val in column.salmon_transferred_values.items():
            constraints.append(model.addConstr(self.salmon_transferred_variables[key] == val, name = "lock_salmon_transferred_" + str(key)))

        return constraints

    def lock_num_tanks(self, model: gp.Model, period :Period, module :Module, num_tanks :int) -> gp.Constr:
        return model.addConstr(
            sum(self.contains_salmon_variable(t,period) 
                for t in module.tanks) == num_tanks, 
            name = f"lock_num_tanks_p{period.index}_m{module.index}_eq{num_tanks}")

    def remove_constraints(self, model: gp.Model, constraints: list[gp.Constr]) -> None:
        """Removes the given constraints from the MIP model.

        args:
            - model: 'gp.Model' The MIP model to remove the constraints from
            - constraints: 'list[gp.Constr]' The constraints to be removed
        """

        for constr in constraints:
            model.remove(constr)

    def biomass_objective_column(self, model: gp.Model, module_idx: int, maximize: bool, neg_deploy_period: int = -1) -> MasterColumn:

        """Builds a column for the Master Problem by solving the generated MIP problem without changing the constraints, and with an objective that minimizes or maximizes the extracted biomass.
        One of the deploy periods might be treated with oposite sign, meaning that the objective will be to minimize the extracted biomass of salmon deployed in that period if the
        general objective for the other deploy periods is to maximize the biomass, and vice versa.

        args:
            - model: 'gp.Model' The MIP model holding the MIP problem
            - module_idx 'int' The index of the module the subproblem is built for
            - maximize 'bool' If True, the objective is to maximize the extracted biomass. If False, the objective is to minimize the extracted biomass
            - neg_deploy_period: If this is the index of a deploy period, extracted salmon from that period will be minimized in the objective if the rest is maximized, and vice versa.

        returns:
            The generated master column
        """

        obj_expr = self.create_biomass_objective_expression(module_idx, neg_deploy_period)
        model.setObjective(obj_expr, GRB.MAXIMIZE if maximize else GRB.MINIMIZE)
        model.optimize()
        obj_value = self.core_objective_expression.getValue()
        return self.get_master_column(module_idx, obj_value, True)

    def extract_weight_variable(self, depl_period: Period, tank: Tank, period: Period) -> gp.Var:
        """Returns the continous MIP variable for weight of extracted salmon

        args:
            - depl_period: 'Period' The period when the extracted salmon was deployed
            - tank: 'Tank' The tank the salmon was extracted from
            - period: 'Period' The period when the salmon was extracted
        """

        return self.extract_weight_variables[(depl_period.index, tank.index, period.index)]

    def population_weight_variable(self, depl_period: Period, tank: Tank, period: Period) -> gp.Var:
        """Returns the continous MIP variable for salmon weight in a tank

        args:
            - depl_period: 'Period' The period when the salmon was deployed
            - tank: 'Tank' The tank
            - period: 'Period' The period to get the salmon weight for
        """

        return self.population_weight_variables[(depl_period.index, tank.index, period.index)]
    
    def transfer_weight_variable(self, depl_period: Period, from_tank: Tank, to_tank: Tank, period: Period) -> gp.Var:
        """Returns the continous MIP variable for weight of transferred salmon

        args:
            - depl_period: 'Period' The period when the transferred salmon was deployed
            - from_tank: 'Tank' The tank the salmon was transferred from
            - to_tank: 'Tank' The tank the salmon was transferred to
            - period: 'Period' The period when the salmon was transferred
        """

        return self.transfer_weight_variables[(depl_period.index, from_tank.index, to_tank.index, period.index)]

    def contains_salmon_variable(self, tank: Tank, period: Period) -> gp.Var:
        """Returns the binary MIP variable for whether a tanks holds salmon at a given period

        args:
            - tank: 'Tank' The tank
            - period: 'Period' The period
        """

        return self.contains_salmon_variables[(tank.index, period.index)]

    def smolt_deployed_variable(self, module: Module, depl_period: Period) -> gp.Var:
        """Returns the binary MIP variable for whether salmon has been deployed in a module at a given period

        args:
            - module: 'Module' The module
            - depl_period: 'Period' The deploy period
        """

        return self.smolt_deployed_variables[(module.index, depl_period.index)]

    def salmon_extracted_variable(self, tank: Tank, period: Period) -> gp.Var:
        """Returns the binary MIP variable for whether salmon was extracted from a tank at the end of a given period

        args:
            - tank: 'Tank' The tank
            - period: 'Period' The period
        """

        return self.salmon_extracted_variables[(tank.index, period.index)]

    def salmon_transferred_variable(self, tank: Tank, period: Period) -> gp.Var:
        """Returns the binary MIP variable for whether salmon was transferred to a tank at a given period

        args:
            - tank: 'Tank' The tank
            - period: 'Period' The period
        """

        return self.salmon_transferred_variables[(tank.index, period.index)]

    def module_active_variable(self, module: Module, depl_period: Period) -> gp.Var:
        """The binary MIP variables for whether the module was active in a production cycle this or previous period

        args:
            - module: 'Module' The module
            - depl_period: 'Period' The deploy period
        """

        return self.module_active_variables[(module.index, depl_period.index)]

    def extract_weight_value(self, depl_period: Period, tank: Tank, period: Period) -> float:
        """Returns the value of the continous MIP variable for weight of extracted salmon

        args:
            - depl_period: 'Period' The period when the extracted salmon was deployed
            - tank: 'Tank' The tank the salmon was extracted from
            - period: 'Period' The period when the salmon was extracted
        """

        return self.extract_weight_variable(depl_period, tank, period).X

    def population_weight_value(self, depl_period: Period, tank: Tank, period: Period) -> float:
        """Returns the value of the continous MIP variable for salmon weight in a tank

        args:
            - depl_period: 'Period' The period when the salmon was deployed
            - tank: 'Tank' The tank
            - period: 'Period' The period to get the salmon weight for
        """

        return self.population_weight_variable(depl_period, tank, period).X
    
    def transfer_weight_value(self, depl_period: Period, from_tank: Tank, to_tank: Tank, period: Period) -> float:
        """Returns the value of the continous MIP variable for weight of transferred salmon

        args:
            - depl_period: 'Period' The period when the transferred salmon was deployed
            - from_tank: 'Tank' The tank the salmon was transferred from
            - to_tank: 'Tank' The tank the salmon was transferred to
            - period: 'Period' The period when the salmon was transferred
        """

        return self.transfer_weight_variable(depl_period, from_tank, to_tank, period).X

    def contains_salmon_value(self, tank: Tank, period: Period) -> float:
        """Returns the value of the binary MIP variable for whether a tanks holds salmon at a given period

        args:
            - tank: 'Tank' The tank
            - period: 'Period' The period
        """

        return self.contains_salmon_variable(tank, period).X

    def smolt_deployed_value(self, module: Module, depl_period: Period) -> float:
        """Returns the value of the binary MIP variable for whether salmon has been deployed in a module at a given period

        args:
            - module: 'Module' The module
            - depl_period: 'Period' The deploy period
        """

        return self.smolt_deployed_variable(module, depl_period).X

    def salmon_extracted_value(self, tank: Tank, period: Period) -> float:
        """Returns the value of the binary MIP variable for whether salmon was extracted from a tank at the end of a given period

        args:
            - tank: 'Tank' The tank
            - period: 'Period' The period
        """

        return self.salmon_extracted_variable(tank, period).X

    def salmon_transferred_value(self, tank: Tank, period: Period) -> float:
        """Returns the value of the binary MIP variable for whether salmon was transferred to a tank at a given period

        args:
            - tank: 'Tank' The tank
            - period: 'Period' The period
        """

        return self.salmon_transferred_variable(tank, period).X

    def drop_positive_solution(self, model: gp.Model) -> None:
        """Prints the objective and the variable values for the last solution. Only positive values are printed.

        args:
            - model: 'gp.Model' The MIP model holding the MIP problem
        """
    
        for key, var in self.extract_weight_variables.items():
            if var.X > 0.5:
                print("extract_weight(%s) = %s"%(key, var.X))
        for key, var in self.population_weight_variables.items():
            if var.X > 0.5:
                print("population_weight(%s) = %s"%(key, var.X))
        for key, var in self.transfer_weight_variables.items():
            if var.X > 0.5:
                print("transfer_weight(%s) = %s"%(key, var.X))
        for key, var in self.contains_salmon_variables.items():
            if var.X > 0.001:
                print("contains_salmon(%s) = %s"%(key, var.X))
        for key, var in self.smolt_deployed_variables.items():
            if var.X > 0.001:
                print("smolt_deployed(%s) = %s"%(key, var.X))
        for key, var in self.salmon_extracted_variables.items():
            if var.X > 0.001:
                print("salmon_extracted(%s) = %s"%(key, var.X))
        for key, var in self.salmon_transferred_variables.items():
            if var.X > 0.001:
                print("salmon_transferred(%s) = %s"%(key, var.X))
        print("objective = %s"%model.ObjVal)
