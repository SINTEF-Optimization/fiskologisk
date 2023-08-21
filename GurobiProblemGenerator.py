import gurobipy as gp
from gurobipy import GRB
from enum import Enum
from Environment import Environment
from Period import Period
from Module import Module
from Tank import Tank

class ObjectiveProfile(Enum):
    Profit: 1
    Biomass: 2

class GurobiProblemGenerator:
    environment: Environment
    objective_profile: ObjectiveProfile
    allow_transfer: bool

    def __init__(self, environment: Environment, objective_profile: ObjectiveProfile = ObjectiveProfile.Profit, allow_transfer: bool = True) -> None:
        self.environment = environment
        self.objective_profile = objective_profile
        self.allow_transfer = allow_transfer

    def build_model(self) -> gp.Model:

        model = gp.Model()
        self.add_variables(model)
        self.add_constraints(model)
        self.add_objective(model)
        return model
    
    def add_variables(self, model: gp.Model) -> None:
        
        # Continous variable: Extracted salmon from deploy period from tank at period
        for dep_p in self.environment.release_periods:
            for t in self.environment.tanks:
                for p in dep_p.extract_periods:
                    model.addVar(name = extract_weight_variable(dep_p, t, p))
        
        # Continous variable: Population weight from deploy period in tank at period
        for dep_p in self.environment.release_periods:
            for t in self.environment.tanks:
                for p in dep_p.periods_after_deploy:
                    if p != self.environment.periods[0] or dep_p == p:
                        model.addVar(name = population_weight_variable(dep_p, t, p))

        # Continous variable: Transferred salmon from deploy period from tank to tank in period
        if self.allow_transfer:
            for dep_p in self.environment.release_periods:
                for from_t in self.environment.tanks:
                    for to_t in from_t.transferable_to:
                        for p in dep_p.transfer_periods:
                            model.addVar(name = transfer_weight_variable(dep_p, from_t, to_t, p))

        # Binary variable: Tank contains salmon in period
        for t in self.environment.tanks:
            for p in self.environment.periods:
                model.addVar(name = contains_salmon_variable(t, p), vtype = GRB.BINARY)
    
        # Binary variable: Smolt is deployed in module in period
        for m in self.environment.modules:
            for dep_p in self.environment.plan_release_periods:
                model.addVar(name = smolt_deployed_variable(m, dep_p), vtype = GRB.BINARY)

        # Binary variable: Salmon is extracted from tank in period
        for t in self.environment.tanks:
            for p in self.environment.periods:
                model.addVar(name = salmon_extracted_variable(t, p), vtype = GRB.BINARY)

        # Binary variable: Salmon is transferred to tank in period
        if self.allow_transfer:
            for t in self.environment.tanks:
                for p in self.environment.periods:
                    model.addVar(name = salmon_extracted_variable(t, p), vtype = GRB.BINARY)

    def add_constraints(self, model: gp.Model) -> None:
        pass

    def add_objective(self, model: gp.Model) -> None:
        match self.objective_profile:
            case ObjectiveProfile.Profit:
                self.add_profit_objective(model)
            case ObjectiveProfile.Biomass:
                self.add_biomass_objective(model)
            case _:
                raise ValueError("Unknown objective profile")

    def add_profit_objective(self, model: gp.Model) -> None:

        obj_expr = gp.LinExpr()
        parameters = self.environment.parameters

        # Revenue for selling post-smolt and harvestable salmon
        weight_classes = self.environment.weight_classes
        for dep_p in self.environment.release_periods:
            for t in self.environment.tanks:

                # Post-smolt
                for p in dep_p.postsmolt_extract_periods:
                    weight_distribution = dep_p.periods_after_deploy_data[p.index].weight_distribution
                    coef = 0
                    for idx in range(len(weight_classes)):
                        coef += weight_classes[idx].post_smolt_revenue * weight_distribution[idx]
                    obj_expr.addTerms(coef, model.getVarByName(extract_weight_variable(dep_p, t, p)))

                # Harvest
                for p in dep_p.harvest_periods:
                    weight_distribution = dep_p.periods_after_deploy_data[p.index].weight_distribution
                    coef = 0
                    for idx in range(len(weight_classes)):
                        coef += weight_classes[idx].harvest_revenue * weight_distribution[idx]
                    coef *= parameters.harvest_yield
                    obj_expr.addTerms(coef, model.getVarByName(extract_weight_variable(dep_p, t, p)))

        # Smolt cost
        for dep_p in self.environment.plan_release_periods:
            for t in self.environment.tanks:
                obj_expr.addTerms(-parameters.smolt_price, model.getVarByName(extract_weight_variable(dep_p, t, dep_p)))

        # Feeding cost

        # Oxygen cost

        # Tank operation

        model.setObjective(obj_expr, GRB.MAXIMIZE)

    def add_biomass_objective(self, model: gp.Model) -> None:
        pass


def extract_weight_variable(depl_period: Period, tank: Tank, period: Period) -> str:
    return "e_%s,%s,%s"%(depl_period.index, tank.index, period.index)

def population_weight_variable(depl_period: Period, tank: Tank, period: Period) -> str:
    return "x_%s,%s,%s"%(depl_period.index, tank.index, period.index)
    
def transfer_weight_variable(depl_period: Period, from_tank: Tank, to_tank: Tank, period: Period) -> str:
    return "y_%s,%s,%s,%s"%(depl_period.index, from_tank.index, to_tank.index, period.index)

def contains_salmon_variable(tank: Tank, period: Period) -> str:
    return "alpha_%s,%s"%(tank.index, period.index)

def smolt_deployed_variable(module: Module, depl_period: Period) -> str:
    return "delta_%s,%s"%(module.index, depl_period.index)

def salmon_extracted_variable(tank: Tank, period: Period) -> str:
    return "epsilon_%s,%s"%(tank.index, period.index)

def salmon_transferred_variable(tank: Tank, period: Period) -> str:
    return "sigma_%s,%s"%(tank.index, period.index)
