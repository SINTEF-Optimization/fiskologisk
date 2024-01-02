from collections import defaultdict
from dataclasses import dataclass, field
import time
import sys
from typing import Dict, List, Tuple
import gurobipy as gp
from Environment import Environment
from GurobiMasterProblemGenerator import GurobiMasterProblemGenerator
from GurobiProblemGenerator import GurobiProblemGenerator, ObjectiveProfile
from Module import Module
from Period import Period
from SolutionProvider import SolutionProvider
from Tank import Tank
from collections import Counter


@dataclass
class ColumnPeriodInfo:
    period: Period
    biomass: float
    extracted: float


@dataclass
class SolutionDict:
    extract: Dict[Tuple[Period, Tank, Period], float] = field(default_factory=dict)
    biomass: Dict[Tuple[Period, Tank, Period], float] = field(default_factory=dict)
    transfer: Dict[Tuple[Period, Tank, Tank, Period], float] = field(default_factory=dict)
    contains_salmon: Dict[Tuple[Tank, Period], float] = field(default_factory=dict)
    did_deploy: Dict[Tuple[Module, Period], float] = field(default_factory=dict)
    did_extract: Dict[Tuple[Tank, Period], float] = field(default_factory=dict)
    did_transfer: Dict[Tuple[Tank, Period], float] = field(default_factory=dict)

    def add(self, other):
        def add_dicts(a, b):
            return dict(Counter(a) + Counter(b))

        self.extract = add_dicts(self.extract, other.extract)
        self.biomass = add_dicts(self.biomass, other.biomass)
        self.transfer = add_dicts(self.transfer, other.transfer)
        self.contains_salmon = add_dicts(self.contains_salmon, other.contains_salmon)
        self.did_deploy = add_dicts(self.did_deploy, other.did_deploy)
        self.did_extract = add_dicts(self.did_extract, other.did_extract)
        self.did_transfer = add_dicts(self.did_transfer, other.did_transfer)


def module_solution(
    prob: GurobiProblemGenerator, solution_module: Module, actual_module: Module, prod_cycle: [Period]
) -> SolutionDict:
    env = prob.environment
    solution_dict = SolutionDict()
    prod_cycle_idxs = set(p.index for p in prod_cycle)
    # print("PROD CYCLE ", prod_cycle_idxs)
    tank_map = {t1: t2 for t1, t2 in zip(solution_module.tanks, actual_module.tanks)}

    # Continous variable: Extracted salmon from deploy period from tank at period
    for dep_p in env.release_periods:
        for p in dep_p.extract_periods:
            for t in solution_module.tanks:
                if p.index in prod_cycle_idxs:
                    sol_key = (dep_p.index, t.index, p.index)
                    act_key = (dep_p.index, tank_map[t].index, p.index)
                    solution_dict.extract[act_key] = prob.extract_weight_variables[sol_key].X

    # Continous variable: Population weight from deploy period in tank at period
    for dep_p in env.release_periods:
        for p in dep_p.periods_after_deploy:
            for t in solution_module.tanks:
                if p.index in prod_cycle_idxs:
                    sol_key = (dep_p.index, t.index, p.index)
                    act_key = (dep_p.index, tank_map[t].index, p.index)
                    solution_dict.biomass[act_key] = prob.population_weight_variables[sol_key].X

    # Continous variable: Transferred salmon from deploy period from tank to tank in period
    if prob.allow_transfer:
        for dep_p in env.release_periods:
            for p in dep_p.transfer_periods:
                for from_t in solution_module.tanks:
                    for to_t in from_t.transferable_to:
                        if p.index in prod_cycle_idxs:
                            sol_key = (dep_p.index, from_t.index, to_t.index, p.index)
                            act_key = (dep_p.index, tank_map[from_t].index, tank_map[to_t].index, p.index)
                            solution_dict.transfer[act_key] = prob.transfer_weight_variables[sol_key].X

    # Binary variable: Tank contains salmon in period
    for t in solution_module.tanks:
        for p in env.periods:
            if p.index in prod_cycle_idxs:
                sol_key = (t.index, p.index)
                act_key = (tank_map[t].index, p.index)
                solution_dict.contains_salmon[act_key] = round(prob.contains_salmon_variables[sol_key].X)
                # if p.index == max(prod_cycle_idxs):
                #     assert abs(prob.contains_salmon_variables[key].X) > 1e-4

    # Binary variable: Smolt is deployed in module in period
    for dep_p in env.plan_release_periods:
        if dep_p.index in prod_cycle_idxs:
            sol_key = (solution_module.index, dep_p.index)
            act_key = (actual_module.index, dep_p.index)
            solution_dict.did_deploy[act_key] = round(prob.smolt_deployed_variables[sol_key].X)

    # Binary variable: Salmon is extracted from tank in period
    for t in solution_module.tanks:
        for p in env.periods:
            if p.index in prod_cycle_idxs:
                sol_key = (t.index, p.index)
                act_key = (tank_map[t].index, p.index)
                solution_dict.did_extract[act_key] = round(prob.salmon_extracted_variables[sol_key].X)

    # Binary variable: Salmon is transferred to tank in period
    if prob.allow_transfer:
        for t in solution_module.tanks:
            if len(t.transferable_from) > 0:
                for p in env.periods:
                    if p.index in prod_cycle_idxs:
                        sol_key = (t.index, p.index)
                        act_key = (tank_map[t].index, p.index)
                        solution_dict.did_transfer[act_key] = round(prob.salmon_transferred_variables[sol_key].X)

    return solution_dict


def set_solution(model: gp.Model, prob: GurobiProblemGenerator, sol: SolutionDict):
    var_dicts = [
        (prob.extract_weight_variables, sol.extract),
        (prob.population_weight_variables, sol.biomass),
        (prob.transfer_weight_variables, sol.transfer),
        (prob.contains_salmon_variables, sol.contains_salmon),
        (prob.smolt_deployed_variables, sol.did_deploy),
        (prob.salmon_extracted_variables, sol.did_extract),
        (prob.salmon_transferred_variables, sol.did_transfer),
    ]

    for vars, vals in var_dicts:
        for key, var in vars.items():
            model.addConstr(var == vals.get(key, 0.0))


@dataclass
class DecompCyclesColumn:
    obj_value: float
    is_initial: bool
    module: Module
    periods: List[ColumnPeriodInfo]
    solution_dict: SolutionDict


@dataclass
class ShadowPrices:
    period_biomass: Dict[int, float]
    yearly_production: Dict[int, float]
    module_period: Dict[Module, Dict[int, float]]
    module_initial: Dict[Module, float]


@dataclass
class DecompCyclesSolution(SolutionProvider):
    solution_dict: SolutionDict

    def extract_weight_value(self, depl_period: Period, tank: Tank, period: Period) -> float:
        return self.solution_dict.extract.get((depl_period, tank, period), 0.0)

    def population_weight_value(self, depl_period: Period, tank: Tank, period: Period) -> float:
        return self.solution_dict.biomass.get((depl_period, tank, period), 0.0)

    def transfer_weight_value(self, depl_period: Period, from_tank: Tank, to_tank: Tank, period: Period) -> float:
        return self.solution_dict.transfer.get((depl_period, from_tank, to_tank, period), 0.0)

    def contains_salmon_value(self, tank: Tank, period: Period) -> float:
        return self.solution_dict.contains_salmon.get((tank, period), 0.0)

    def smolt_deployed_value(self, module: Module, depl_period: Period) -> float:
        return self.solution_dict.did_deploy.get((module, depl_period), 0.0)

    def salmon_extracted_value(self, tank: Tank, period: Period) -> float:
        return self.solution_dict.did_extract.get((tank, period), 0.0)

    def salmon_transferred_value(self, tank: Tank, period: Period) -> float:
        return self.solution_dict.did_transfer.get((tank, period), 0.0)


def decomp_cycles_solve(
    env: Environment,
    objective_profile: ObjectiveProfile = ObjectiveProfile.PROFIT,
    allow_transfer: bool = True,
    add_symmetry_breaks: bool = False,
) -> DecompCyclesSolution:
    
    # Column generation MIP model of the full production planning problem.
    # Columns represent a single production cycle for a single module. The first period
    # of a production cycle is the only period where biomass is deployed into one or more tanks.
    # The last period of a production cycle is when all tanks are empty.
    # Constraints on the master problem are:
    #  - At most one production cycle can be active in a module in each time period.
    #  - For modules that are in the middle of a production cycle in the initial period,
    #    exactly one initial production cycle is selected.
    #  - The maximum biomass constraint needs to be fulfilled for all selected production cycles 
    #    within a single time period.
    #  - The maximum yearly production constraint needs to be fulfilled for all selected production
    #    cycles within a year.

    # (relaxed) Restricted master problem
    rmp = gp.Model()
    rmp.Params.Threads = 2
    rmp.ModelSense = gp.GRB.MAXIMIZE

    planning_start_period = min(env.periods, key=lambda p: p.index)
    initial_deploy_periods = get_initial_deploy_periods(env)
    period_years = {p: y for y in env.years for p in y.periods}

    # Set limit on total biomass each period (6.22)
    regulation_rescale = len(env.tanks) / env.parameters.tanks_in_regulations
    max_mass = env.parameters.max_total_biomass * regulation_rescale
    bio_cstr = {p: rmp.addConstr(gp.LinExpr() <= max_mass, name="max_biomass_%s" % p.index) for p in env.periods}

    # Set limit on yearly production (6.23)
    max_prod = env.parameters.max_yearly_production * regulation_rescale
    prod_constr = {y.year: rmp.addConstr(gp.LinExpr() <= max_prod, name="max_year_prod_%s" % y.year) for y in env.years}

    # Set limit on each module for selecting only one prod-cycle column in each time period
    pack_cstr = {
        m: {p: rmp.addConstr(gp.LinExpr() <= 1, name=f"pack_m{m.index}_p{p.index}") for p in env.periods}
        for m in env.modules
    }

    # Initial production cycles
    init_cstr = {
        m: rmp.addConstr(gp.LinExpr() == 1, name=f"init_m{m.index}")
        for m, p in initial_deploy_periods.items()
        if p is not None
    }

    # The list of associated data with each column added to the master problem.
    # This list is extended by the `add_column` function below.
    master_columns: List[Tuple[DecompCyclesColumn, gp.Var]] = []

    def add_column(col: DecompCyclesColumn):
        var = rmp.addVar(name=f"lambda_{len(master_columns)}", obj=col.obj_value)
        master_columns.append((col, var))
        yearly_extraction = defaultdict(float)

        for period_info in col.periods:
            rmp.chgCoeff(bio_cstr[period_info.period], var, period_info.biomass)
            rmp.chgCoeff(pack_cstr[col.module][period_info.period], var, 1.0)
            if col.is_initial:
                rmp.chgCoeff(init_cstr[col.module], var, 1.0)

            yearly_extraction[period_years[period_info.period].year] += period_info.extracted

        for year, extracted in yearly_extraction.items():
            rmp.chgCoeff(prod_constr[year], var, extracted)

    # The pricing subproblem MIP is shared between all modules. Initial conditions
    # are added and removed to this MIP as needed.
    subproblem_generator, pricing_mip = build_subproblem(env, objective_profile, allow_transfer, add_symmetry_breaks)

    #
    # ADD INITIAL COLUMNS
    #
    for col in initial_columns(env, objective_profile, allow_transfer, add_symmetry_breaks, initial_deploy_periods):
        add_column(col)

    iter = 0
    n_initial_subproblems = 0
    n_subproblems = 0
    t0 = time.time()

    # Column generation loop.
    while True:
        iter += 1
        # Optimized relaxed restricted master problem
        print(f"Iter {iter}: solving relaxed restricted master problem")
        rmp.optimize()

        assert rmp.Status == gp.GRB.OPTIMAL

        # Extract the shadow prices for the current relaxed master problem.
        shadow_prices = ShadowPrices(
            period_biomass={p.index: c.Pi for p, c in bio_cstr.items()},
            yearly_production={y: c.Pi for y, c in prod_constr.items()},
            module_period={m: {p.index: c.pi for p, c in ps.items()} for m, ps in pack_cstr.items()},
            module_initial={m: c.Pi for m, c in init_cstr.items()},
        )

        # Set up solving of the pricing problem
        n_columns_before = len(master_columns)
        subproblem_generator.set_subproblem_objective(
            pricing_mip, shadow_prices.period_biomass, shadow_prices.yearly_production
        )

        # The following function solves the pricing problem and adds columns to the master problem
        # for a single relevant period interval.
        def do_price(p1: int, p2: int, init_m: Module):
            modules = [init_m] if init_m is not None else env.modules
            constant_price = 0
            is_initial = init_m is not None
            constraints = []

            if is_initial:
                constant_price += shadow_prices.module_initial[init_m]
                constant_price += sum(shadow_prices.module_period[init_m][p] for p in range(p1, p2 + 1))
                subproblem_generator.add_initial_value_constraints(pricing_mip, init_m.index, 0)
            else:
                constant_price += min(
                    (sum(shadow_prices.module_period[m][p] for p in range(p1, p2 + 1)) for m in env.modules)
                )

                for dep_p in env.release_periods:
                    if dep_p.index < p1:
                        for p in dep_p.periods_after_deploy:
                            for t in env.modules[0].tanks:
                                constraints.append(
                                    pricing_mip.addConstr(
                                        subproblem_generator.population_weight_variable(dep_p, t, p) == 0,
                                        name=f"no pre-planning biomass",
                                    )
                                )

            for period in env.periods:
                if period.index < p1 or period.index >= p2:
                    for t in env.modules[0].tanks:
                        constraints.append(
                            pricing_mip.addConstr(
                                subproblem_generator.contains_salmon_variable(t, period) == 0,
                                name=f"lock_noproduction_t{t.index}_p{period.index}",
                            )
                        )

            pricing_mip.optimize()
            if pricing_mip.status != gp.GRB.OPTIMAL:
                pricing_mip.computeIIS()
                pricing_mip.write("iis.ilp")
                raise Exception()

            value = pricing_mip.ObjVal
            margin = 1e-4 * max(abs(value), abs(constant_price), 1)
            if value - constant_price > margin:
                obj_value = subproblem_generator.calculate_core_objective(0)
                for actual_module in modules:
                    column = extract_column(
                        obj_value, subproblem_generator, is_initial, env.modules[0], actual_module, p1, p2
                    )
                    add_column(column)

            subproblem_generator.remove_constraints(pricing_mip, constraints)
            if is_initial:
                subproblem_generator.remove_initial_value_constraints(pricing_mip)

        # There are two types of relevant periods: initial and non-initial.
        # First, we solve for module-specific initial production cycle columns.
        print(f"Iter {iter}: solving initial production cycles")
        for module in env.modules:
            if initial_deploy_periods[module] is not None:
                for p2 in initial_deploy_periods[module].extract_periods:
                    n_initial_subproblems += 1
                    do_price(planning_start_period.index, p2.index + 1, module)

        # Then, we solve all non-initial periods, where the modules are assumed to be interchangable.
        print(f"Iter {iter}: solving all production cycle ranges")
        for p1 in env.periods:
            for p2 in p1.extract_periods:
                n_subproblems += 1
                do_price(p1.index, p2.index + 1, None)

        # Add new columns
        # If we added at least one column, try again.
        if len(master_columns) == n_columns_before:
            print(
                f"No positive reduced cost columns found. Solved {n_initial_subproblems} initial and {n_subproblems} other production plans in {time.time()-t0:.2f} sec."
            )
            break

    # Optimize non-relaxed (binary) problem
    relaxation_obj = rmp.ObjVal

    for _, v in master_columns:
        v.VType = gp.GRB.BINARY

    rmp.optimize()
    priceandbranch_obj = rmp.ObjVal
    print(
        f"GAP {(relaxation_obj/priceandbranch_obj-1.0)*100.0:.3f}%  ({relaxation_obj/1e6:.2f}M  -- {priceandbranch_obj/1e6:.2f}M)"
    )

    solution_dict = SolutionDict()
    final_obj = 0.0
    for col, var in master_columns:
        if var.X > 0.5:
            solution_dict.add(col.solution_dict)
            final_obj += col.obj_value

    # Verify the whole solution in the original non-decomposed problem
    full_gpg = GurobiProblemGenerator(env, objective_profile, allow_transfer, add_symmetry_breaks)
    full_model = full_gpg.build_model()

    set_solution(full_model, full_gpg, solution_dict)
    full_model.optimize()
    if full_model.status != gp.GRB.OPTIMAL:
        full_model.computeIIS()
        full_model.write("iis.ilp")
        raise Exception()

    # For simplicity, we just return the full MIP model here, since we have already solved it with a fixed
    # assignment to verify the correctness of our solution.
    return full_gpg, full_model


def initial_columns(
    environment: Environment,
    objective_profile: ObjectiveProfile,
    allow_transfer: bool,
    add_symmetry_breaks: bool,
    initial_deploy_periods: Dict[Module, Period | None],
):
    
    # Solve the full MIP formulation until we find the first
    # feasible solution. Use this to extract a set of initial
    # columns, since these are required for the master problem to 
    # be feasible.

    old_gmpg = GurobiMasterProblemGenerator(
        environment,
        objective_profile=objective_profile,
        allow_transfer=allow_transfer,
        add_symmetry_breaks=add_symmetry_breaks,
    )

    subproblem_generator = old_gmpg.sub_problem_generator()
    model = subproblem_generator.build_model()
    model.Params.SolutionLimit = 1
    model.optimize()

    for module, init_p in initial_deploy_periods.items():
        if init_p is None:
            continue

        p1 = min(environment.periods, key=lambda p: p.index).index
        p2 = p1
        assert get_module_biomass(subproblem_generator, module, p1) > 1
        while get_module_biomass(subproblem_generator, module, p2) > 1:
            p2 += 1

        col = extract_column(0.0, subproblem_generator, True, module, module, p1, p2)
        yield col


def extract_column(
    obj_value: float,
    subproblem_generator: GurobiProblemGenerator,
    is_initial: bool,
    solution_module: Module,
    actual_module: Module,
    p1: int,
    p2: int,
) -> List[DecompCyclesColumn]:
    env = subproblem_generator.environment
    periods = env.period_indices

    periodinfo: List[ColumnPeriodInfo] = []
    for p in range(p1, p2 + 1):
        period = periods[p]
        biomass = get_module_biomass(subproblem_generator, solution_module, p)
        extracted = get_module_extracted(subproblem_generator, solution_module, p)

        periodinfo.append(ColumnPeriodInfo(period, biomass, extracted))

    assert periodinfo[-1].biomass < 1e-5

    solution_dict = module_solution(
        subproblem_generator, solution_module, actual_module, [periods[p] for p in range(p1, p2 + 1)]
    )
    return DecompCyclesColumn(obj_value, is_initial, actual_module, periodinfo, solution_dict)


def get_module_biomass(subproblem_generator: GurobiProblemGenerator, module: Module, period_idx: int) -> float:
    period = subproblem_generator.environment.period_indices[period_idx]
    return sum(
        (
            subproblem_generator.population_weight_variable(dep_p, tank, period).X
            for dep_p in period.deploy_periods
            for tank in module.tanks
        )
    )


def get_module_extracted(subproblem_generator: GurobiProblemGenerator, module: Module, period_idx: int) -> float:
    period = subproblem_generator.environment.period_indices[period_idx]

    return sum(
        (
            subproblem_generator.extract_weight_variable(dep_p, tank, period).X
            for dep_p in period.deploy_periods
            for tank in module.tanks
            if period in dep_p.extract_periods
        )
    )


def get_initial_deploy_periods(environment: Environment) -> Dict[Module, Period | None]:
    initial_deploy_periods = {}
    for module in environment.modules:
        ps = set(tank.initial_deploy_period for tank in module.tanks if tank.initial_weight > 1e-5)
        assert len(ps) == 0 or len(ps) == 1
        if len(ps) == 0:
            initial_deploy_periods[module] = None
        else:
            initial_deploy_periods[module] = environment.period_indices[next(iter(ps))]
    return initial_deploy_periods


def build_subproblem(
    environment: Environment,
    objective_profile: ObjectiveProfile,
    allow_transfer: bool,
    add_symmetry_breaks: bool,
) -> Tuple[GurobiProblemGenerator, gp.Model]:
    # This is a bit of a hack: we re-use the first ("old") decomposition solver to produce
    # a subproblem for a single module, but we remove the constraints setting the initial
    # biomass, so the model becomes the same for all modules. (initial biomasses are enabled
    # and disabled as needed in the `do_price` function)

    old_gmpg = GurobiMasterProblemGenerator(
        environment,
        objective_profile=objective_profile,
        allow_transfer=allow_transfer,
        add_symmetry_breaks=add_symmetry_breaks,
    )

    sub_problem_generator = old_gmpg.sub_problem_generator()
    model = sub_problem_generator.build_module_subproblemn(0)
    sub_problem_generator.remove_initial_value_constraints(model)
    return (sub_problem_generator, model)
