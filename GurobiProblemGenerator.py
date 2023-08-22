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
        self.add_objective(model)
        self.add_constraints(model)
        return model
    
    def add_variables(self, model: gp.Model) -> None:
        
        # Continous variable: Extracted salmon from deploy period from tank at period
        for dep_p in self.environment.release_periods:
            for p in dep_p.extract_periods:
                for t in self.environment.tanks:
                    model.addVar(name = extract_weight_variable(dep_p, t, p))
        
        # Continous variable: Population weight from deploy period in tank at period
        for dep_p in self.environment.release_periods:
            for p in dep_p.periods_after_deploy:
                set_init = p == self.environment.periods[0] and p != dep_p
                for t in self.environment.tanks:
                    v = model.addVar(name = population_weight_variable(dep_p, t, p))
                    if set_init:
                        v.Start = t.initial_weight if t.initial_weight > 0 and t.initial_deploy_period == dep_p.index else 0.0

        # Continous variable: Transferred salmon from deploy period from tank to tank in period
        if self.allow_transfer:
            for dep_p in self.environment.release_periods:
                for p in dep_p.transfer_periods:
                    for from_t in self.environment.tanks:
                        for to_t in from_t.transferable_to:
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
        for dep_p in self.environment.release_periods:
            for t in self.environment.tanks:

                # Population weight variable for all periods
                for p in dep_p.periods_after_deploy:
                    cost = dep_p.periods_after_deploy_data[p.index].feed_cost
                    obj_expr.addTerms(-cost, model.getVarByName(population_weight_variable(dep_p, t, p)))

                # Extract weight variable for extract periods
                for p in dep_p.extract_periods:
                    cost = dep_p.periods_after_deploy_data[p.index].feed_cost
                    obj_expr.addTerms(-cost, model.getVarByName(population_weight_variable(dep_p, t, p)))
                    obj_expr.addTerms(cost, model.getVarByName(extract_weight_variable(dep_p, t, p)))

        # Oxygen cost
        for dep_p in self.environment.release_periods:
            for p in dep_p.periods_after_deploy:
                for t in self.environment.tanks:
                    cost = dep_p.periods_after_deploy_data[p.index].oxygen_cost
                    obj_expr.addTerms(-cost, model.getVarByName(population_weight_variable(dep_p, t, p)))

        # Tank operation
        for p in self.environment.periods:
            for t in self.environment.tanks:

                # Minimum cost for operating a tank
                obj_expr.addTerms(-parameters.min_tank_cost, model.getVarByName(contains_salmon_variable(t, p)))

                for dep_p in p.deploy_periods:
                    obj_expr.addTerms(-parameters.marginal_increase_pumping_cost, model.getVarByName(population_weight_variable(dep_p, t, p)))

        model.setObjective(obj_expr, GRB.MAXIMIZE)

    def add_biomass_objective(self, model: gp.Model) -> None:

        obj_expr = gp.LinExpr()

        for dep_p in self.environment.release_periods:
            for p in dep_p.extract_periods:
                for t in self.environment.tanks:
                    obj_expr.addTerms(1.0, model.getVarByName(extract_weight_variable(dep_p, t, p)))

        model.setObjective(obj_expr, GRB.MAXIMIZE)

    def add_constraints(self, model: gp.Model) -> None:

        self.add_smolt_deployment_constraints(model)
        self.add_extraction_constraints(model)
        self.add_salmon_transfer_constraints(model)
        self.add_salmon_density_constraints(model)
        self.add_regulatory_constraints(model)
        self.add_biomass_development_constraints(model)
        self.add_improving_constraints(model)

    def add_smolt_deployment_constraints(self, model: gp.Model) -> None:

        max_w = self.environment.parameters.max_deploy_smolt
        min_w = self.environment.parameters.min_deploy_smolt
        for dep_p in self.environment.plan_release_periods:
            first = dep_p == self.environment.periods[0]
            prev_p = None if first else next(pp for pp in self.environment.periods if pp.index == dep_p.index - 1)
            for m in self.environment.modules:
                delta = model.getVarByName(smolt_deployed_variable(m, dep_p))
                deployed_smolt_expr = gp.LinExpr()

                # Only deploy smolt in module if all tanks were empty previous period (5.4)
                for t in m.tanks:
                    deployed_smolt_expr.addTerms(1.0, model.getVarByName(population_weight_variable(dep_p, t, dep_p)))
                    if not first:
                        model.addConstr(delta + model.getVarByName(contains_salmon_variable(t, prev_p)) <= 1, name = "empty_deploy_tank_%s,%s"%(m.index, t.index))
                    elif t.initial_weight > 0:
                        model.addConstr(delta == 0, name = "empty_deploy_tank_%s,%s"%(m.index, t.index))

                # Set weight range for deployed smolt (5.3)
                model.addConstr(min_w * delta <= deployed_smolt_expr, name = "min_smolt_dep_%s,%s"%(m.index, dep_p.index))
                model.addConstr(deployed_smolt_expr <= max_w * delta, name = "max_smolt_dep_%s,%s"%(m.index, dep_p.index))

    def add_extraction_constraints(self, model: gp.Model) -> None:

        max_w = self.environment.parameters.max_extract_weight
        for t in self.environment.tanks:
            for p in self.environment.periods:
                epsilon = model.getVarByName(salmon_extracted_variable(t, p))
                extr_w_expr = gp.LinExpr()

                # Set max limit for extracted salmon (5.5)
                for dep_p in p.deploy_periods_for_extract:
                    extr_w_expr.addTerms(1.0, model.getVarByName(extract_weight_variable(dep_p, t, p)))
                model.addConstr(extr_w_expr <= max_w * epsilon, name = "max_extract_%s,%s"(t.index, p.index))

                # Empty tank after extraction (5.6)
                if p != self.environment.periods[-1]:
                    next_p = next(np for np in self.environment.periods if np.index == p.index + 1)
                    model.addConstr(model.getVarByName(contains_salmon_variable(t, next_p)) + epsilon <= 1, name = "empty_extract_tank_%s,%s"%(t.index, p.index))

    def add_salmon_transfer_constraints(self, model: gp.Model) -> None:

        max_w = self.environment.parameters.max_transfer_weight
        min_w = self.environment.parameters.min_transfer_weight
        for p in self.environment.periods:
            first = p == self.environment.periods[0]
            prev_p = None if first else next(pp for pp in self.environment.periods if pp.index == p.index - 1)
            last = p == self.environment.periods[-1]
            next_p = None if last else next(np for np in self.environment.periods if np.index == p.index + 1)
            for t in self.environment.tanks:
                sigma = model.getVarByName(salmon_transferred_variable(t, p))
                min_w_expr = gp.LinExpr(min_w, sigma)
                max_w_expr = gp.LinExpr(max_w, sigma)

                # Only transfer to empty tanks (5.7)
                if not first:
                    model.addConstr(sigma + model.getVarByName(contains_salmon_variable(t, prev_p)) <= 1, name = "transfer_to_empty_%s,%s"%(t.index, p.index))
                elif t.initial_weight > 0:
                    model.addConstr(sigma == 0, name = "transfer_to_empty_%s,%s"%(t.index, p.index))

                # Tanks can not receive and extract same period (5.8)
                model.addConstr(sigma + model.getVarByName(salmon_extracted_variable(t, prev_p)) <= 1, name = "no_extract_if_transfer_%s,%s"%(t.index, p.index))

                for from_t in t.transferable_from:

                    # Do not empty tank transferred from (5.9)
                    if not last:
                        model.addConstr(sigma - model.getVarByName(contains_salmon_variable(from_t, next_p)) <= 0, name = "transfer_from_not_empty_%s,%s,%s"%(t.index, from_t.index, p.index))

                    # Set weight range for transferred salmon (5.10)
                    transf_w_expr = gp.LinExpr()
                    for dep_p in p.deploy_periods_for_transfer:
                        transf_w_expr.addTerms(1.0, model.getVarByName(transfer_weight_variable(dep_p, from_t, t, p)))
                    model.addConstr(min_w_expr <= transf_w_expr, name = "min_transfer_%s,%s,%s"%(t.index, from_t.index, p.index))
                    model.addConstr(transf_w_expr <= max_w_expr, name = "max_transfer_%s,%s,%s"%(t.index, from_t.index, p.index))

    def add_salmon_density_constraints(self, model: gp.Model) -> None:

        max_den = self.environment.parameters.max_tank_density
        for t in self.environment.tanks:
            inv_vol = t.inverse_volume
            for p in self.environment.periods:

                # Set limit on tank density (5.11)
                weight_expr = gp.LinExpr()
                for dep_p in p.deploy_periods:
                    weight_expr.addTerms(inv_vol, model.getVarByName(population_weight_variable(dep_p, t, p)))
                for dep_p in p.deploy_periods_for_transfer:
                    for from_t in t.transferable_from:
                        weight_expr.addTerms(inv_vol, model.getVarByName(transfer_weight_variable(dep_p, from_t, t, p)))
                model.addConstr(weight_expr <= max_den * model.getVarByName(contains_salmon_variable(t, p)), name = "max_density_%s,%s"%(t.index, p.index))

    def add_regulatory_constraints(self, model: gp.Model) -> None:

        # Set limit on total biomass each period (5.12)
        max_mass = self.environment.parameters.max_total_biomass
        for p in self.environment.periods:
            weight_expr = gp.LinExpr()
            for dep_p in p.deploy_periods:
                for t in self.environment.tanks:
                    weight_expr.addTerms(1.0, model.getVarByName(population_weight_variable(dep_p, t, p)))
            model.addConstr(weight_expr <= max_mass, name = "max_biomass_%s"%p.index)

        # Set limit on yearly production (5.13)
        max_prod = self.environment.parameters.max_yearly_production
        for y in self.environment.years:
            extr_w_expr = gp.LinExpr()
            for p in y.periods:
                for dep_p in p.deploy_periods_for_extract:
                    for t in self.environment.tanks:
                        extr_w_expr.addTerms(1.0, model.getVarByName(extract_weight_variable(dep_p, t, p)))
            model.addConstr(extr_w_expr <= max_prod, name = "max_year_prod_%s"%y.year)

    def add_biomass_development_constraints(self, model: gp.Model) -> None:

        loss = self.environment.parameters.monthly_loss
        for dep_p in self.environment.release_periods:
            if len(dep_p.periods_after_deploy) > 0:
                for p in dep_p.periods_after_deploy[:-1]:
                    p_data = dep_p.periods_after_deploy_data[p.index]
                    growth_factor = (1.0 - loss) * p_data.growth_factor
                    transf_growth_factor = (1.0 - loss) * p_data.transferred_growth_factor

                    is_transfer = p in dep_p.transfer_periods
                    is_extract = p in dep_p.extract_periods

                    for t in self.environment.tanks:
                        next_w_expr = gp.LinExpr()

                        # Add updated weight from salmon at start of period (5.14, 5.15, 5.16, 5.17)
                        next_w_expr.addTerms(growth_factor, model.getVarByName(population_weight_variable(dep_p, t, p)))
                        
                        if is_transfer:
                            # Subtract weight from salmon transfered out during period (5.15, 5.16)
                            for to_t in t.transferable_to:
                                next_w_expr.addTerms(-growth_factor, model.getVarByName(transfer_weight_variable(dep_p, t, to_t, p)))

                            # Add weight from salmon transfered in during period (5.15, 5.16)
                            for from_t in t.transferable_from:
                                next_w_expr.addTerms(transf_growth_factor, model.getVarByName(transfer_weight_variable(dep_p, from_t, t, p)))
                        
                        if is_extract:
                            # Subtract weight from salmon extracted during period (5.16, 5.17)
                            next_w_expr.addTerms(-growth_factor, model.getVarByName(extract_weight_variable(dep_p, t, p)))
                        next_p = next(np for np in dep_p.periods_after_deploy if np.index == p.index + 1)
                        model.addConstr(model.getVarByName(population_weight_variable(dep_p, t, next_p)) == next_w_expr, name = "next_period_mass_%s,%s,%s"%(dep_p.index, t.index, p.index))

                # All salmon in last possible period must be harvested (5.18)
                last_p = dep_p.periods_after_deploy[-1]
                model.addConstr(model.getVarByName(population_weight_variable(dep_p, t, last_p)) == model.getVarByName(extract_weight_variable(dep_p, t, last_p)), name = "extract_last_period_%s,%s"%(dep_p.index, t.index))

    def add_improving_constraints(self, model: gp.Model) -> None:

        prev_p = self.environment.periods[0]
        for p in self.environment.periods[1:]:
            is_deploy = p in self.environment.plan_release_periods
            for m in self.environment.modules:
                for t in m.tanks:

                    # Add improving constraint: Tank is empty if empty previos period and no transfer or deploy this period (5.20, 5.21)
                    expr = gp.LinExpr()
                    if is_deploy:
                        expr.addTerms(1, model.getVarByName(smolt_deployed_variable(m, p)))
                    expr.addTerms(1, model.getVarByName(salmon_transferred_variable(t, p)))
                    expr.addTerms(1, model.getVarByName(contains_salmon_variable(t, prev_p)))
                    expr.addTerms(-1, model.getVarByName(salmon_extracted_variable(t, prev_p)))
                    model.addConstr(expr >= model.getVarByName(contains_salmon_variable(t, prev_p)), name = "no_transfer_deploy_%s,%s"%(dep_p.index, t.index))
            prev_p = p

        for dep_p in self.environment.plan_release_periods:
            first_extract_idx = dep_p.extract_periods[0].index
            up_to_extract = [p for p in dep_p.periods_after_deploy if p.index <= first_extract_idx ]
            for m in self.environment.modules:
                delta = model.getVarByName(smolt_deployed_variable(m, p))
                for p in up_to_extract:

                    # Add improving constraint: At least one tank in module has salmon from deploy up to first possible extract period (5.22)
                    sum_alpha = gp.LinExpr()
                    for t in m.tanks:
                        sum_alpha.addTerms(1.0, model.getVarByName(contains_salmon_variable(t, p)))
                    model.addConstr(delta <= sum_alpha, name = "before_first_extract__%s,%s,%s"%(m.index, dep_p.index, p.index))
                
                # Remaining to add: (5.23)

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
