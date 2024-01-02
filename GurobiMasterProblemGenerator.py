import gurobipy as gp
from gurobipy import GRB
from Environment import Environment
from GurobiProblemGenerator import GurobiProblemGenerator
from GurobiProblemGenerator import ObjectiveProfile
from MasterColumn import MasterColumn
from Period import Period
from SolutionProvider import SolutionProvider
from Tank import Tank
from Module import Module

class GurobiMasterProblemGenerator(SolutionProvider):
    """
    Problem generator for the Master Problem in the Dantzig-Wolfe decomposition algorithm applied on the landbased salmon farming MIP problem.
    The generator takes in the setup of the production facility and time periods for the planning horizon,
    builds the objective function and constraints across the subproblem variables, adds new columns from the column generations in the sub problems
    solves the master problem and provides the master problem solution and dual values from the constraints to be applied to the subproblems.
    The Master Problem can be solved as a relaxed LP problem, or with MIP constraints on the binary variables in the subproblems.
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
    """If positive, each module has maximum one production cycle, they are deployed in chronological order, and the number sets a maximum limit on the number of simultaneously active deployments"""

    columns: dict[int, list[MasterColumn]]
    """The modules in the problem, i.e. the extreme points in the variable domains for each subproblem. Key is module index"""

    column_variables: dict[(int, int), gp.Var]
    """The variables for the contribution from each column in the master problem.Key is module index and a running index within the columns from the same module"""

    contains_salmon_variables: dict[(int, int), gp.Var]
    """The relaxed binary variables for whether tanks hold salmon at a given period. Key is tank and period. Can become unrelaxed if relaxed master problem fails to get integer binaries."""

    smolt_deployed_variables: dict[(int, int), gp.Var]
    """The relaxed binary variables for whether salmon has been deployed in a module at a given period. Key is module and deploy period within the planning horizon. Can become unrelaxed if relaxed master problem fails to get integer binaries."""

    salmon_extracted_variables: dict[(int, int), gp.Var]
    """The relaxed binary variables for whether salmon was extracted from a tank at the end of a given period. Key is tank and period. Can become unrelaxed if relaxed master problem fails to get integer binaries."""

    salmon_transferred_variables: dict[(int, int), gp.Var]
    """The relaxed binary variables for whether salmon was transferred to a tank at a given period. Key is tank transferred to and period. Can become unrelaxed if relaxed master problem fails to get integer binaries."""

    period_biomass_constraints: dict[int, gp.Constr]
    """The constraints for the total mass of salmon in the tanks at a period. Key is period index"""

    yearly_production_constraints: dict[int, gp.Constr]
    """The constraints for the total mass of salmon extracted as post smolt or harvest within a production year. Key is year"""

    module_convexity_constraints: dict[int, gp.Constr]
    """The constraints for the convexity of the column combination for each module. Key is module index"""

    set_contains_salmon_constraints: dict[(int, int), gp.Constr]
    """The constraints setting the values of the relaxed binary variables for whether tanks hold salmon at a given period. Key is tank and period"""

    set_smolt_deployed_constraints: dict[(int, int), gp.Constr]
    """The constraints setting the values of the relaxed binary variables for whether salmon has been deployed in a module at a given period. Key is module and deploy period within the planning horizon"""

    set_salmon_extracted_constraints: dict[(int, int), gp.Constr]
    """The constraints setting the values of the relaxed binary variables for whether salmon was extracted from a tank at the end of a given period. Key is tank and period"""

    set_salmon_transferred_constraints: dict[(int, int), gp.Constr]
    """The constraints setting the values of the relaxed binary variables for whether salmon was transferred to a tank at a given period. Key is tank transferred to and period"""

    extract_weight_values: dict[(int, int, int), float]
    """The values of the continous MIP variables for weight of salmon extracted from the tanks for post-smolt or harvesting. Key is deploy period, tank and extract period"""

    population_weight_values: dict[(int, int, int), float]
    """The values of the continous MIP variables for salmon weight in a tank at a period. Key is deploy period, tank and period the mass is given for"""

    transfer_weight_values: dict[(int, int, int, int), float]
    """The values of the continous MIP variables for weight of salmon transfered at a period. Key is deploy period, tank transferred from, tank transferred to and period when the transfer takes place"""

    contains_salmon_values: dict[(int, int), int]
    """The values of the binary MIP variables for whether tanks hold salmon at a given period. Key is tank and period"""

    smolt_deployed_values: dict[int, int]
    """The values of the binary MIP variables for whether salmon has been deployed in a module at a given period. Key is module and deploy period within the planning horizon"""

    salmon_extracted_values: dict[(int, int), int]
    """The values of the binary MIP variables for whether salmon was extracted from a tank at the end of a given period. Key is tank and period"""

    salmon_transferred_values: dict[(int, int), int]
    """The values of the binary MIP variables for whether salmon was transferred to a tank at a given period. Key is tank transferred to and period"""

    def __init__(self, environment: Environment, objective_profile: ObjectiveProfile = ObjectiveProfile.PROFIT, allow_transfer: bool = True, add_symmetry_breaks: bool = False, max_single_modules: int = 0) -> None:

        self.environment = environment
        self.objective_profile = objective_profile
        self.allow_transfer = allow_transfer
        self.add_symmetry_breaks = add_symmetry_breaks
        self.max_single_modules = max_single_modules

        self.columns = {}
        self.column_variables = {}
        self.period_biomass_constraints = {}
        self.yearly_production_constraints = {}
        self.module_convexity_constraints = {}

    def sub_problem_generator(self) -> GurobiProblemGenerator:
        """Cretes a problem generator for a single module subproblem with the same environment setup

        returns:
            The generator for the subproblem
        """

        return GurobiProblemGenerator(self.environment, objective_profile = self.objective_profile, allow_transfer = self.allow_transfer, add_symmetry_breaks = self.add_symmetry_breaks, max_single_modules = self.max_single_modules)

    def build_model(self) -> gp.Model:
        """Builds the MIP model

        returns:
            The generated MIP model
        """

        model = gp.Model()
        model.Params.Threads = 2

        # Empty set of columns and variables
        self.columns = {}
        self.column_variables = {}
        for m in self.environment.modules:
            self.columns[m.index] = []

        # Add relaxed binary variables
        self.add_variables(model)

        # Set empty objective
        model.setObjective(gp.LinExpr(), GRB.MAXIMIZE)

        # Add constraints
        self.add_constraints(model)

        return model

    def add_variables(self, model: gp.Model) -> None:
        """Generates all variables for the relaxed Master Problem
        
        args:
            - model: 'gp.Model' The MIP model to add the variables into.
        """

        # Relaxed binary variable: Tank contains salmon in period
        self.contains_salmon_variables = {}
        for t in self.environment.tanks:
            for p in self.environment.periods:
                key = (t.index, p.index)
                var = model.addVar(name = "alpha_%s,%s"%key)
                self.contains_salmon_variables[key] = var

        # Relaxed binary variable: Smolt is deployed in module in period
        self.smolt_deployed_variables = {}
        for m in self.environment.modules:
            for dep_p in self.environment.plan_release_periods:
                key = (m.index, dep_p.index)
                var = model.addVar(name = "delta_%s,%s"%key)
                self.smolt_deployed_variables[key] = var

        # Relaxed binary variable: Salmon is extracted from tank in period
        self.salmon_extracted_variables = {}
        for t in self.environment.tanks:
            for p in self.environment.periods:
                key = (t.index, p.index)
                var = model.addVar(name = "epsilon_%s,%s"%key)
                self.salmon_extracted_variables[key] = var

        # Relaxed binary variable: Salmon is transferred to tank in period
        self.salmon_transferred_variables = {}
        if self.allow_transfer:
            for t in self.environment.tanks:
                if len(t.transferable_from) > 0:
                    for p in self.environment.periods:
                        key = (t.index, p.index)
                        var = model.addVar(name = "sigma_%s,%s"%key)
                        self.salmon_transferred_variables[key] = var

    def add_constraints(self, model: gp.Model) -> None:
        """Adds all the constraints to the MIP problem
        
        args:
            - model: 'gp.Model' The MIP model to add the constraints into
        """

        # Set limit on total biomass each period (6.22)
        self.period_biomass_constraints = {}
        max_production_tanks = len(self.environment.tanks) if self.max_single_modules == 0 else self.max_single_modules * len(self.environment.modules[0].tanks)
        regulation_rescale = max_production_tanks / self.environment.parameters.tanks_in_regulations
        max_mass = self.environment.parameters.max_total_biomass * regulation_rescale
        for p in self.environment.periods:
            constr = model.addConstr(gp.LinExpr() <= max_mass, name = "max_biomass_%s"%p.index)
            self.period_biomass_constraints[p.index] = constr

        # Set limit on yearly production (6.23)
        self.yearly_production_constraints = {}
        max_prod = self.environment.parameters.max_yearly_production * regulation_rescale
        for y in self.environment.years:
            constr = model.addConstr(gp.LinExpr() <= max_prod, name = "max_year_prod_%s"%y.year)
            self.yearly_production_constraints[y.year] = constr

        # Set convexity of column combinations (6.24)
        self.module_convexity_constraints = {}
        for m in self.environment.modules:
            constr = model.addConstr(gp.LinExpr() == 1, name = "convexity_%s"%m.index)
            self.module_convexity_constraints[m.index] = constr

        # Set relaxed binary variable: Tank contains salmon in period (6.26)
        self.set_contains_salmon_constraints = {}
        for t in self.environment.tanks:
            for p in self.environment.periods:
                key = (t.index, p.index)
                self.set_contains_salmon_constraints[key] = model.addConstr(self.contains_salmon_variables[key] == 0.0, name = "set_alpha_%s,%s"%key)

        # Set relaxed binary variable: Smolt is deployed in module in period (6.25)
        self.set_smolt_deployed_constraints = {}
        for m in self.environment.modules:
            for dep_p in self.environment.plan_release_periods:
                key = (m.index, dep_p.index)
                self.set_smolt_deployed_constraints[key] = model.addConstr(self.smolt_deployed_variables[key] == 0.0, name = "set_delta_%s,%s"%key)

        # Set relaxed binary variable: Salmon is extracted from tank in period (6.27)
        self.set_salmon_extracted_constraints = {}
        for t in self.environment.tanks:
            for p in self.environment.periods:
                key = (t.index, p.index)
                self.set_salmon_extracted_constraints[key] = model.addConstr(self.salmon_extracted_variables[key] == 0.0, name = "set_epsilon_%s,%s"%key)

        # Set relaxed binary variable: Salmon is transferred to tank in period (6.28)
        self.set_salmon_transferred_constraints = {}
        if self.allow_transfer:
            for t in self.environment.tanks:
                if len(t.transferable_from) > 0:
                    for p in self.environment.periods:
                        key = (t.index, p.index)
                        self.set_salmon_transferred_constraints[key] = model.addConstr(self.salmon_transferred_variables[key] == 0.0, name = "set_sigma_%s,%s"%key)

    def add_column(self, model: gp.Model, column: MasterColumn) -> None:
        """Adds a submodel solution column to the Master Problem.
        This will add a new lambda value for the specific column, and update its contribution in the constraints and objective.
        
        args:
            - model: 'gp.Model' The MIP model of the Master Problem
            - column: 'MasterColumn' The column to be added
        """

        module_index = column.module_index
        idx = len(self.columns[module_index])
        self.columns[module_index].append(column)
        key = (module_index, idx)
        var = model.addVar(name = "lambda_%s,%s"%key)
        self.column_variables[key] = var

        # Update coefficient in objective and constraints for the new column variable
        var.Obj = column.objective_value

        for p in self.environment.periods:
            constr = self.period_biomass_constraints[p.index]
            coef = 0
            for dep_p in p.deploy_periods:
                for t in self.environment.get_tanks(module_index):
                    key = (dep_p.index, t.index, p.index)
                    coef += column.population_weight_values[key]
            model.chgCoeff(constr, var, coef)

        for y in self.environment.years:
            constr = self.yearly_production_constraints[y.year]
            coef = 0
            for p in y.periods:
                for dep_p in p.deploy_periods_for_extract:
                    for t in self.environment.get_tanks(module_index):
                        key = (dep_p.index, t.index, p.index)
                        coef += column.extract_weight_values[key]
            model.chgCoeff(constr, var, coef)

        constr = self.module_convexity_constraints[module_index]
        model.chgCoeff(constr, var, 1.0)

        for t in self.environment.get_tanks(module_index):
            for p in self.environment.periods:
                key = (t.index, p.index)
                if column.contains_salmon_values[key] == 1:
                    constr = self.set_contains_salmon_constraints[key]
                    model.chgCoeff(constr, var, -1.0)

        for m in self.environment.get_modules(module_index):
            for dep_p in self.environment.plan_release_periods:
                key = (m.index, dep_p.index)
                if column.smolt_deployed_values[key] == 1:
                    constr = self.set_smolt_deployed_constraints[key]
                    model.chgCoeff(constr, var, -1.0)

        for t in self.environment.get_tanks(module_index):
            for p in self.environment.periods:
                key = (t.index, p.index)
                if column.salmon_extracted_values[key] == 1:
                    constr = self.set_salmon_extracted_constraints[key]
                    model.chgCoeff(constr, var, -1.0)

        if self.allow_transfer:
            for t in self.environment.get_tanks(module_index):
                if len(t.transferable_from) > 0:
                    for p in self.environment.periods:
                        key = (t.index, p.index)
                        if column.salmon_transferred_values[key] == 1:
                            constr = self.set_salmon_transferred_constraints[key]
                            model.chgCoeff(constr, var, -1.0)

    def get_dual_values(self) -> (dict[int, float], dict[int, float], dict[int, float]):
        """Returns the dual values of the constraints in the Master Problem from the last solution.
        
        returns:
            Three dictionaries with the dual values.
            The first dictionary holds the dual values of the constraints for the total mass of salmon in the tanks at a period. Key is period index.
            The second dictionary holds the dual values of the constraints for the total mass of salmon extracted as post smolt or harvest within a production year. Key is year.
            The third dictionary holds the dual values of the constraints for the convexity of the column combination for each module. Key is module index.
        """

        period_biomass_duals = {}
        for p in self.environment.periods:
            period_biomass_duals[p.index] = self.period_biomass_constraints[p.index].Pi

        yearly_production_duals = {}
        for y in self.environment.years:
            yearly_production_duals[y.year] = self.yearly_production_constraints[y.year].Pi

        convex_duals = {}
        for m in self.environment.modules:
            convex_duals[m.index] = self.module_convexity_constraints[m.index].Pi

        print("CONVEX DUALS")
        for k,v in convex_duals.items():
            print(f" - {k} {v}")


        other_constraint_sets = [
            self.set_contains_salmon_constraints, 
            self.set_smolt_deployed_constraints, 
            self.set_salmon_extracted_constraints, 
            self.set_salmon_transferred_constraints]
        
        for c_set in other_constraint_sets:
            for _k,v in c_set.items():
                if abs(v.Pi) > 1e-5:
                    print(f"Warning: constraint {v.Name} has shadow price {v.Pi}")

        return period_biomass_duals, yearly_production_duals, convex_duals

    def validate_integer_values(self) -> bool:
        """Returns whether the solution of the Master Problem holds binary values for the relaxed binary variables, within a tolerance.
        Only smolt deployment, extraction and salmon transfer variables are tested, since the values of the indication variables on whether tanks contain salmon are
        uniquely determined by the values of the other binary variables.
        
        returns:
            Whether all tested variables have binary value within the tolerance.
        """
    
        for val in self.smolt_deployed_values.values():
            if abs(val) > 1e-6 and abs(1-val) > 1e-6:
                return False
        for val in self.salmon_extracted_values.values():
            if abs(val) > 1e-6 and abs(1-val) > 1e-6:
                return False
        for val in self.salmon_transferred_values.values():
            if abs(val) > 1e-6 and abs(1-val) > 1e-6:
                return False
        return True

    def lock_binaries(self) -> None:
        """Requires all relaxed binary variables from the original problem to be binary (non-relaxed).
        The lambda variables might still be non-binary, as long as they combine columns into a solution with binary values.
        """

        for var in self.contains_salmon_variables.values():
            var.VType = GRB.BINARY
        for var in self.smolt_deployed_variables.values():
            var.VType = GRB.BINARY
        for var in self.salmon_extracted_variables.values():
            var.VType = GRB.BINARY
        for var in self.salmon_transferred_variables.values():
            var.VType = GRB.BINARY

    def generate_solution(self) -> None:
        """Generates the values for the solution of the original problem from the current solution to the Master Problem"""

        self.extract_weight_values = {}
        self.population_weight_values = {}
        self.transfer_weight_values = {}
        self.contains_salmon_values = {}
        self.smolt_deployed_values = {}
        self.salmon_extracted_values = {}
        self.salmon_transferred_values = {}

        for m in self.environment.modules:
            mod_columns = self.columns[m.index]
            nmb_cols = len(mod_columns)
            lambda_values = []
            for idx in range(nmb_cols):
                var = self.column_variables[(m.index, idx)]
                lambda_values.append(var.X)

            # Continous variable: Extracted salmon from deploy period from tank at period
            for dep_p in self.environment.release_periods:
                for p in dep_p.extract_periods:
                    for t in m.tanks:
                        key = (dep_p.index, t.index, p.index)
                        value = sum(lambda_values[idx] * mod_columns[idx].extract_weight_values[key] for idx in range(nmb_cols))
                        self.extract_weight_values[key] = value

            # Continous variable: Population weight from deploy period in tank at period
            for dep_p in self.environment.release_periods:
                for p in dep_p.periods_after_deploy:
                    for t in m.tanks:
                        key = (dep_p.index, t.index, p.index)
                        value = sum(lambda_values[idx] * mod_columns[idx].population_weight_values[key] for idx in range(nmb_cols))
                        self.population_weight_values[key] = value

            # Continous variable: Transferred salmon from deploy period from tank to tank in period
            if self.allow_transfer:
                for dep_p in self.environment.release_periods:
                    for p in dep_p.transfer_periods:
                        for from_t in m.tanks:
                            for to_t in from_t.transferable_to:
                                key = (dep_p.index, from_t.index, to_t.index, p.index)
                                value = sum(lambda_values[idx] * mod_columns[idx].transfer_weight_values[key] for idx in range(nmb_cols))
                                self.transfer_weight_values[key] = value

            # Binary variable: Tank contains salmon in period
            # Binary variable: Salmon is extracted from tank in period
            for t in m.tanks:
                for p in self.environment.periods:
                    key = (t.index, p.index)
                    value = sum(lambda_values[idx] * mod_columns[idx].contains_salmon_values[key] for idx in range(nmb_cols))
                    self.contains_salmon_values[key] = value
                    value = sum(lambda_values[idx] * mod_columns[idx].salmon_extracted_values[key] for idx in range(nmb_cols))
                    self.salmon_extracted_values[key] = value

            # Binary variable: Smolt is deployed in module in period
            for dep_p in self.environment.plan_release_periods:
                key = (m.index, dep_p.index)
                value = sum(lambda_values[idx] * mod_columns[idx].smolt_deployed_values[key] for idx in range(nmb_cols))
                self.smolt_deployed_values[key] = value

            # Binary variable: Salmon is transferred to tank in period
            # self.salmon_transferred_values = {}
            self.salmon_transferred_variables = {}
            if self.allow_transfer:
                for t in m.tanks:
                    if len(t.transferable_from) > 0:
                        for p in self.environment.periods:
                            key = (t.index, p.index)
                            value = sum(lambda_values[idx] * mod_columns[idx].salmon_transferred_values[key] for idx in range(nmb_cols))
                            self.salmon_transferred_values[key] = value

    def extract_weight_value(self, depl_period: Period, tank: Tank, period: Period) -> float:
        """Returns the value of the continous MIP variable for weight of extracted salmon

        args:
            - depl_period: 'Period' The period when the extracted salmon was deployed
            - tank: 'Tank' The tank the salmon was extracted from
            - period: 'Period' The period when the salmon was extracted
        """

        return self.extract_weight_values[(depl_period.index, tank.index, period.index)]

    def population_weight_value(self, depl_period: Period, tank: Tank, period: Period) -> float:
        """Returns the value of the continous MIP variable for salmon weight in a tank

        args:
            - depl_period: 'Period' The period when the salmon was deployed
            - tank: 'Tank' The tank
            - period: 'Period' The period to get the salmon weight for
        """

        return self.population_weight_values[(depl_period.index, tank.index, period.index)]
    
    def transfer_weight_value(self, depl_period: Period, from_tank: Tank, to_tank: Tank, period: Period) -> float:
        """Returns the value of the continous MIP variable for weight of transferred salmon

        args:
            - depl_period: 'Period' The period when the transferred salmon was deployed
            - from_tank: 'Tank' The tank the salmon was transferred from
            - to_tank: 'Tank' The tank the salmon was transferred to
            - period: 'Period' The period when the salmon was transferred
        """

        return self.transfer_weight_values[(depl_period.index, from_tank.index, to_tank.index, period.index)]

    def contains_salmon_value(self, tank: Tank, period: Period) -> float:
        """Returns the value of the binary MIP variable for whether a tanks holds salmon at a given period

        args:
            - tank: 'Tank' The tank
            - period: 'Period' The period
        """

        return self.contains_salmon_values[(tank.index, period.index)]

    def smolt_deployed_value(self, module: Module, depl_period: Period) -> float:
        """Returns the value of the binary MIP variable for whether salmon has been deployed in a module at a given period

        args:
            - module: 'Module' The module
            - depl_period: 'Period' The deploy period
        """

        return self.smolt_deployed_values[(module.index, depl_period.index)]

    def salmon_extracted_value(self, tank: Tank, period: Period) -> float:
        """Returns the value of the binary MIP variable for whether salmon was extracted from a tank at the end of a given period

        args:
            - tank: 'Tank' The tank
            - period: 'Period' The period
        """

        return self.salmon_extracted_values[(tank.index, period.index)]

    def salmon_transferred_value(self, tank: Tank, period: Period) -> float:
        """Returns the value of the binary MIP variable for whether salmon was transferred to a tank at a given period

        args:
            - tank: 'Tank' The tank
            - period: 'Period' The period
        """

        return self.salmon_transferred_values[(tank.index, period.index)]

    def drop_positive_solution(self, model: gp.Model) -> None:
        """Prints the objective and the variable values for the last solution. Only positive values are printed.

        args:
            - model: 'gp.Model' The MIP model holding the MIP problem
        """
    
        for m in self.environment.modules:
            for idx in range(len(self.columns[m.index])):
                key = (m.index, idx)
                var = self.column_variables[key]
                print("lambda(%s) = %s"%(key, var.X))
        for key, value in self.extract_weight_values.items():
            if value > 0.5:
                print("extract_weight(%s) = %s"%(key, value))
        for key, value in self.population_weight_values.items():
            if value > 0.5:
                print("population_weight(%s) = %s"%(key, value))
        for key, value in self.transfer_weight_values.items():
            if value > 0.5:
                print("transfer_weight(%s) = %s"%(key, value))
        for key, value in self.contains_salmon_values.items():
            if value > 0.001:
                print("contains_salmon(%s) = %s"%(key, value))
        for key, value in self.smolt_deployed_values.items():
            if value > 0.001:
                print("smolt_deployed(%s) = %s"%(key, value))
        for key, value in self.salmon_extracted_values.items():
            if value > 0.001:
                print("salmon_extracted(%s) = %s"%(key, value))
        for key, value in self.salmon_transferred_values.items():
            if value > 0.001:
                print("salmon_transferred(%s) = %s"%(key, value))
        print("objective = %s"%model.ObjVal)

    def drop_dual_values(self, model: gp.Model) -> None:
        """Prints the objective and the dual values of the constraints for the last solution.

        args:
            - model: 'gp.Model' The MIP model holding the MIP problem
        """

        for m in self.environment.modules:
            for idx in range(len(self.columns[m.index])):
                key = (m.index, idx)
                var = self.column_variables[key]
                print("lambda(%s) = %s"%(key, var.X))
        for key, constr in self.period_biomass_constraints.items():
            print("dual of period_biomass_constraint(%s) = %s"%(key, constr.Pi))
        for key, constr in self.yearly_production_constraints.items():
            print("dual of yearly_production_constraint(%s) = %s"%(key, constr.Pi))
        for key, constr in self.module_convexity_constraints.items():
            print("dual of module_convexity_constraint(%s) = %s"%(key, constr.Pi))
        print("objective = %s"%model.ObjVal)
