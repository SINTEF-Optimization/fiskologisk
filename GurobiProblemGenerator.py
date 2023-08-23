import gurobipy as gp
from gurobipy import GRB
from enum import Enum
from Environment import Environment
from Period import Period
from Module import Module
from Tank import Tank

class ObjectiveProfile(Enum):
    PROFIT = 1
    BIOMASS = 2

class GurobiProblemGenerator:

    environment: Environment
    objective_profile: ObjectiveProfile
    allow_transfer: bool

    extract_weight_variables: dict[(int, int, int), gp.Var]
    population_weight_variables: dict[(int, int, int), gp.Var]
    transfer_weight_variables: dict[(int, int, int, int), gp.Var]
    contains_salmon_variables: dict[(int, int), gp.Var]
    smolt_deployed_variables: dict[(int, int), gp.Var]
    salmon_extracted_variables: dict[(int, int), gp.Var]
    salmon_transferred_variables: dict[(int, int), gp.Var]

    def __init__(self, environment: Environment, objective_profile: ObjectiveProfile = ObjectiveProfile.PROFIT, allow_transfer: bool = True) -> None:
        self.environment = environment
        self.objective_profile = objective_profile
        self.allow_transfer = allow_transfer

        self.extract_weight_variables = {}
        self.population_weight_variables = {}
        self.transfer_weight_variables = {}
        self.contains_salmon_variables = {}
        self.smolt_deployed_variables = {}
        self.salmon_extracted_variables = {}
        self.salmon_transferred_variables = {}

    def build_model(self) -> gp.Model:

        model = gp.Model()
        self.add_variables(model)
        self.add_objective(model)
        self.add_constraints(model)
        return model

    def add_variables(self, model: gp.Model) -> None:

        # Continous variable: Extracted salmon from deploy period from tank at period
        self.extract_weight_variables = {}
        for dep_p in self.environment.release_periods:
            for p in dep_p.extract_periods:
                for t in self.environment.tanks:
                    key = (dep_p.index, t.index, p.index)
                    var = model.addVar(name = "e_%s,%s,%s"%key)
                    self.extract_weight_variables[key] = var

        # Continous variable: Population weight from deploy period in tank at period
        self.population_weight_variables = {}
        for dep_p in self.environment.release_periods:
            for p in dep_p.periods_after_deploy:
                set_init = p == self.environment.periods[0] and p != dep_p
                for t in self.environment.tanks:
                    key = (dep_p.index, t.index, p.index)
                    var = model.addVar(name = "x_%s,%s,%s"%key)
                    self.population_weight_variables[key] = var
                    if set_init:
                        var.Start = t.initial_weight if t.initial_weight > 0 and t.initial_deploy_period == dep_p.index else 0.0

        # Continous variable: Transferred salmon from deploy period from tank to tank in period
        self.transfer_weight_variables = {}
        if self.allow_transfer:
            for dep_p in self.environment.release_periods:
                for p in dep_p.transfer_periods:
                    for from_t in self.environment.tanks:
                        for to_t in from_t.transferable_to:
                            key = (dep_p.index, from_t.index, to_t.index, p.index)
                            var = model.addVar(name = "y_%s,%s,%s,%s"%key)
                            self.transfer_weight_variables[key] = var

        # Binary variable: Tank contains salmon in period
        self.contains_salmon_variables = {}
        for t in self.environment.tanks:
            for p in self.environment.periods:
                key = (t.index, p.index)
                var = model.addVar(name = "alpha_%s,%s"%key, vtype = GRB.BINARY)
                self.contains_salmon_variables[key] = var

        # Binary variable: Smolt is deployed in module in period
        self.smolt_deployed_variables = {}
        for m in self.environment.modules:
            for dep_p in self.environment.plan_release_periods:
                key = (m.index, dep_p.index)
                var = model.addVar(name = "delta_%s,%s"%key, vtype = GRB.BINARY)
                self.smolt_deployed_variables[key] = var

        # Binary variable: Salmon is extracted from tank in period
        self.salmon_extracted_variables = {}
        for t in self.environment.tanks:
            for p in self.environment.periods:
                key = (t.index, p.index)
                var = model.addVar(name = "epsilon_%s,%s"%key, vtype = GRB.BINARY)
                self.salmon_extracted_variables[key] = var

        # Binary variable: Salmon is transferred to tank in period
        self.salmon_transferred_variables = {}
        if self.allow_transfer:
            for t in self.environment.tanks:
                for p in self.environment.periods:
                    key = (t.index, p.index)
                    var = model.addVar(name = "sigma_%s,%s"%key, vtype = GRB.BINARY)
                    self.salmon_transferred_variables[key] = var

    def add_objective(self, model: gp.Model) -> None:
        match self.objective_profile:
            case ObjectiveProfile.PROFIT:
                self.add_profit_objective(model)
            case ObjectiveProfile.BIOMASS:
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
            for t in self.environment.tanks:
                obj_expr.addTerms(-parameters.smolt_price, self.population_weight_variable(dep_p, t, dep_p))

        # Feeding cost
        for dep_p in self.environment.release_periods:
            for t in self.environment.tanks:

                # Population weight variable for all periods
                for p in dep_p.periods_after_deploy:
                    cost = dep_p.periods_after_deploy_data[p.index].feed_cost
                    obj_expr.addTerms(-cost, self.population_weight_variable(dep_p, t, p))

                # Extract weight variable for extract periods
                for p in dep_p.extract_periods:
                    cost = dep_p.periods_after_deploy_data[p.index].feed_cost
                    obj_expr.addTerms(-cost, self.population_weight_variable(dep_p, t, p))
                    obj_expr.addTerms(cost, self.extract_weight_variable(dep_p, t, p))

        # Oxygen cost
        for dep_p in self.environment.release_periods:
            for p in dep_p.periods_after_deploy:
                for t in self.environment.tanks:
                    cost = dep_p.periods_after_deploy_data[p.index].oxygen_cost
                    obj_expr.addTerms(-cost, self.population_weight_variable(dep_p, t, p))

        # Tank operation
        for p in self.environment.periods:
            for t in self.environment.tanks:

                # Minimum cost for operating a tank
                obj_expr.addTerms(-parameters.min_tank_cost, self.contains_salmon_variable(t, p))

                for dep_p in p.deploy_periods:
                    obj_expr.addTerms(-parameters.marginal_increase_pumping_cost, self.population_weight_variable(dep_p, t, p))

        model.setObjective(obj_expr, GRB.MAXIMIZE)

    def add_biomass_objective(self, model: gp.Model) -> None:

        obj_expr = gp.LinExpr()

        for dep_p in self.environment.release_periods:
            for p in dep_p.extract_periods:
                for t in self.environment.tanks:
                    obj_expr.addTerms(1.0, self.extract_weight_variable(dep_p, t, p))

        model.setObjective(obj_expr, GRB.MAXIMIZE)

    def add_constraints(self, model: gp.Model) -> None:

        self.add_smolt_deployment_constraints(model)
        self.add_extraction_constraints(model)
        if self.allow_transfer:
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
                delta = self.smolt_deployed_variable(m, dep_p)
                deployed_smolt_expr = gp.LinExpr()

                # Only deploy smolt in module if all tanks were empty previous period (5.4)
                for t in m.tanks:
                    deployed_smolt_expr.addTerms(1.0, self.population_weight_variable(dep_p, t, dep_p))
                    if not first:
                        model.addConstr(delta + self.contains_salmon_variable(t, prev_p) <= 1, name = "empty_deploy_tank_%s,%s"%(m.index, t.index))
                    elif t.initial_weight > 0:
                        model.addConstr(delta == 0, name = "empty_deploy_tank_%s,%s"%(m.index, t.index))

                # Set weight range for deployed smolt (5.3)
                model.addConstr(min_w * delta <= deployed_smolt_expr, name = "min_smolt_dep_%s,%s"%(m.index, dep_p.index))
                model.addConstr(deployed_smolt_expr <= max_w * delta, name = "max_smolt_dep_%s,%s"%(m.index, dep_p.index))

    def add_extraction_constraints(self, model: gp.Model) -> None:

        max_w = self.environment.parameters.max_extract_weight
        for t in self.environment.tanks:
            for p in self.environment.periods:
                epsilon = self.salmon_extracted_variable(t, p)
                extr_w_expr = gp.LinExpr()

                # Set max limit for extracted salmon (5.5)
                for dep_p in p.deploy_periods_for_extract:
                    extr_w_expr.addTerms(1.0, self.extract_weight_variable(dep_p, t, p))
                model.addConstr(extr_w_expr <= max_w * epsilon, name = "max_extract_%s,%s"%(t.index, p.index))

                # Empty tank after extraction (5.6)
                if p != self.environment.periods[-1]:
                    next_p = next(np for np in self.environment.periods if np.index == p.index + 1)
                    model.addConstr(self.contains_salmon_variable(t, next_p) + epsilon <= 1, name = "empty_extract_tank_%s,%s"%(t.index, p.index))

    def add_salmon_transfer_constraints(self, model: gp.Model) -> None:

        max_w = self.environment.parameters.max_transfer_weight
        min_w = self.environment.parameters.min_transfer_weight
        for p in self.environment.periods:
            first = p == self.environment.periods[0]
            prev_p = None if first else next(pp for pp in self.environment.periods if pp.index == p.index - 1)
            last = p == self.environment.periods[-1]
            next_p = None if last else next(np for np in self.environment.periods if np.index == p.index + 1)
            for t in self.environment.tanks:
                sigma = self.salmon_transferred_variable(t, p)
                min_w_expr = gp.LinExpr(min_w, sigma)
                max_w_expr = gp.LinExpr(max_w, sigma)

                # Only transfer to empty tanks (5.7)
                if not first:
                    model.addConstr(sigma + self.contains_salmon_variable(t, prev_p) <= 1, name = "transfer_to_empty_%s,%s"%(t.index, p.index))
                elif t.initial_weight > 0:
                    model.addConstr(sigma == 0, name = "transfer_to_empty_%s,%s"%(t.index, p.index))

                # Tanks can not receive and extract same period (5.8)
                model.addConstr(sigma + self.salmon_extracted_variable(t, p) <= 1, name = "no_extract_if_transfer_%s,%s"%(t.index, p.index))

                for from_t in t.transferable_from:

                    # Do not empty tank transferred from (5.9)
                    if not last:
                        model.addConstr(sigma - self.contains_salmon_variable(from_t, next_p) <= 0, name = "transfer_from_not_empty_%s,%s,%s"%(t.index, from_t.index, p.index))

                    # Set weight range for transferred salmon (5.10)
                    transf_w_expr = gp.LinExpr()
                    for dep_p in p.deploy_periods_for_transfer:
                        transf_w_expr.addTerms(1.0, self.transfer_weight_variable(dep_p, from_t, t, p))
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
                    weight_expr.addTerms(inv_vol, self.population_weight_variable(dep_p, t, p))
                if self.allow_transfer:
                    for dep_p in p.deploy_periods_for_transfer:
                        for from_t in t.transferable_from:
                            weight_expr.addTerms(inv_vol, self.transfer_weight_variable(dep_p, from_t, t, p))
                model.addConstr(weight_expr <= max_den * self.contains_salmon_variable(t, p), name = "max_density_%s,%s"%(t.index, p.index))

    def add_regulatory_constraints(self, model: gp.Model) -> None:

        # Set limit on total biomass each period (5.12)
        max_mass = self.environment.parameters.max_total_biomass
        for p in self.environment.periods:
            weight_expr = gp.LinExpr()
            for dep_p in p.deploy_periods:
                for t in self.environment.tanks:
                    weight_expr.addTerms(1.0, self.population_weight_variable(dep_p, t, p))
            model.addConstr(weight_expr <= max_mass, name = "max_biomass_%s"%p.index)

        # Set limit on yearly production (5.13)
        max_prod = self.environment.parameters.max_yearly_production
        for y in self.environment.years:
            extr_w_expr = gp.LinExpr()
            for p in y.periods:
                for dep_p in p.deploy_periods_for_extract:
                    for t in self.environment.tanks:
                        extr_w_expr.addTerms(1.0, self.extract_weight_variable(dep_p, t, p))
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
                        next_w_expr.addTerms(growth_factor, self.population_weight_variable(dep_p, t, p))
                        
                        if is_transfer and self.allow_transfer:
                            # Subtract weight from salmon transfered out during period (5.15, 5.16)
                            for to_t in t.transferable_to:
                                next_w_expr.addTerms(-growth_factor, self.transfer_weight_variable(dep_p, t, to_t, p))

                            # Add weight from salmon transfered in during period (5.15, 5.16)
                            for from_t in t.transferable_from:
                                next_w_expr.addTerms(transf_growth_factor, self.transfer_weight_variable(dep_p, from_t, t, p))
                        
                        if is_extract:
                            # Subtract weight from salmon extracted during period (5.16, 5.17)
                            next_w_expr.addTerms(-growth_factor, self.extract_weight_variable(dep_p, t, p))
                        next_p = next(np for np in dep_p.periods_after_deploy if np.index == p.index + 1)
                        model.addConstr(self.population_weight_variable(dep_p, t, next_p) == next_w_expr, name = "next_period_mass_%s,%s,%s"%(dep_p.index, t.index, p.index))

                # All salmon in last possible period must be extracted (5.18)
                last_p = dep_p.periods_after_deploy[-1]
                for t in self.environment.tanks:
                    model.addConstr(self.population_weight_variable(dep_p, t, last_p) == self.extract_weight_variable(dep_p, t, last_p), name = "extract_last_period_%s,%s"%(dep_p.index, t.index))

    def add_improving_constraints(self, model: gp.Model) -> None:

        prev_p = self.environment.periods[0]
        for p in self.environment.periods[1:]:
            is_deploy = p in self.environment.plan_release_periods
            for m in self.environment.modules:
                for t in m.tanks:

                    # Add improving constraint: Tank is empty if empty previous period and no transfer or deploy this period (5.20, 5.21)
                    expr = gp.LinExpr()
                    if is_deploy:
                        expr.addTerms(1, self.smolt_deployed_variable(m, p))
                    if self.allow_transfer:
                        expr.addTerms(1, self.salmon_transferred_variable(t, p))
                    expr.addTerms(1, self.contains_salmon_variable(t, prev_p))
                    expr.addTerms(-1, self.salmon_extracted_variable(t, prev_p))
                    model.addConstr(expr >= self.contains_salmon_variable(t, prev_p), name = "no_transfer_deploy_%s,%s"%(p.index, t.index))
            prev_p = p

        for dep_p in self.environment.plan_release_periods:
            first_extract_idx = dep_p.extract_periods[0].index
            up_to_extract = [p for p in dep_p.periods_after_deploy if p.index <= first_extract_idx ]
            for m in self.environment.modules:
                delta = self.smolt_deployed_variable(m, dep_p)
                for p in up_to_extract:

                    # Add improving constraint: At least one tank in module has salmon from deploy up to first possible extract period (5.22)
                    sum_alpha = gp.LinExpr()
                    for t in m.tanks:
                        sum_alpha.addTerms(1.0, self.contains_salmon_variable(t, p))
                    model.addConstr(gp.LinExpr(1.0, delta) <= sum_alpha, name = "before_first_extract__%s,%s,%s"%(m.index, dep_p.index, p.index))
                
                for t in m.tanks:

                    # Add improving constraint: Force alpha to be one from deploy to extraction: (5.23)
                    expr_lhs_1 = gp.LinExpr()
                    expr_lhs_1.addTerms(2, delta)
                    expr_lhs_1.addConstant(-2)
                    expr_lhs_1.addTerms(1, self.contains_salmon_variable(t, dep_p))
                    for p_idx in range(1, len(dep_p.periods_after_deploy)):
                        p = dep_p.periods_after_deploy[p_idx]
                        expr_lhs_2 = gp.LinExpr()
                        for pp_idx in range(1, p_idx + 1):
                            pp = dep_p.periods_after_deploy[pp_idx]
                            if self.allow_transfer:
                                expr_lhs_2.addTerms(1, self.salmon_transferred_variable(t, pp))
                            if pp_idx != p_idx:
                                expr_lhs_2.addTerms(-1, self.salmon_extracted_variable(t, pp))
                    model.addConstr(expr_lhs_1 + expr_lhs_2 <= self.contains_salmon_variable(t, p), name = "force_alpha_%s,%s,%s"%(t.index, dep_p.index, p.index))

    def extract_weight_variable(self, depl_period: Period, tank: Tank, period: Period) -> gp.Var:
        return self.extract_weight_variables[(depl_period.index, tank.index, period.index)]

    def population_weight_variable(self, depl_period: Period, tank: Tank, period: Period) -> gp.Var:
        return self.population_weight_variables[(depl_period.index, tank.index, period.index)]
    
    def transfer_weight_variable(self, depl_period: Period, from_tank: Tank, to_tank: Tank, period: Period) -> gp.Var:
        return self.transfer_weight_variables[(depl_period.index, from_tank.index, to_tank.index, period.index)]

    def contains_salmon_variable(self, tank: Tank, period: Period) -> gp.Var:
        return self.contains_salmon_variables[(tank.index, period.index)]

    def smolt_deployed_variable(self, module: Module, depl_period: Period) -> gp.Var:
        return self.smolt_deployed_variables[(module.index, depl_period.index)]

    def salmon_extracted_variable(self, tank: Tank, period: Period) -> gp.Var:
        return self.salmon_extracted_variables[(tank.index, period.index)]

    def salmon_transferred_variable(self, tank: Tank, period: Period) -> gp.Var:
        return self.salmon_transferred_variables[(tank.index, period.index)]
