from collections import defaultdict
from dataclasses import dataclass
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


class DecompCyclesSolution(SolutionProvider):
    pass


@dataclass
class ColumnPeriodInfo:
    period: Period
    biomass: float
    extracted: float


@dataclass
class DecompCyclesColumn:
    obj_value: float
    periods: List[ColumnPeriodInfo]


@dataclass
class ShadowPrices:
    period_biomass: Dict[int, float]
    yearly_production: Dict[int, float]
    module_period: Dict[Module, Dict[int, float]]
    module_initial: Dict[Module, float]


def decomp_cycles_solve(
    env: Environment,
    objective_profile: ObjectiveProfile = ObjectiveProfile.PROFIT,
    allow_transfer: bool = True,
    add_symmetry_breaks: bool = False,
) -> DecompCyclesSolution:
    # (relaxed) Restricted master problem
    rmp = gp.Model()
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
    init_cstr = {m: rmp.addConstr(gp.LinExpr() == 1, name=f"init_m{m.index}") for m in env.modules}

    master_columns: List[Tuple[Module, DecompCyclesColumn, gp.Var]] = []

    def add_column(col: DecompCyclesColumn, modules: List[Module], is_initial: bool):
        for module in modules:
            var = rmp.addVar(name=f"lambda_{len(master_columns)}", obj=col.obj_value)
            master_columns.append((module, col, var))
            yearly_extraction = defaultdict(float)

            for period_info in col.periods:
                rmp.chgCoeff(bio_cstr[period_info.period], var, period_info.biomass)
                rmp.chgCoeff(pack_cstr[module][period_info.period], var, 1.0)
                if is_initial:
                    rmp.chgCoeff(init_cstr[module], var, 1.0)

                yearly_extraction[period_years[period_info.period].year] += period_info.extracted

            for year, extracted in yearly_extraction.items():
                rmp.chgCoeff(prod_constr[year], var, extracted)

    subproblem_generator, pricing_mip = build_subproblem(env, objective_profile, allow_transfer, add_symmetry_breaks)

    #
    # ADD INITIAL COLUMNS
    #
    for module, col in initial_columns(
        env, objective_profile, allow_transfer, add_symmetry_breaks, initial_deploy_periods
    ):
        add_column(col, [module], True)

    iter = 0
    while True:
        iter += 1
        # Optimized relaxed restricted master problem
        print(f"Iter {iter}: solving relaxed restricted master problem")
        rmp.optimize()

        assert rmp.Status == gp.GRB.OPTIMAL

        # Get shadow prices
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

        def do_price(p1: int, p2: int, init_m: Module):
            constant_price = 0
            is_initial = init_m is not None

            if is_initial:
                constant_price += shadow_prices.module_initial[init_m]
                constant_price += sum(shadow_prices.module_period[init_m][p] for p in range(p1, p2 + 1))
                subproblem_generator.add_initial_value_constraints(pricing_mip, init_m.index, 0)
            else:
                constant_price += min(
                    (sum(shadow_prices.module_period[m][p] for p in range(p1, p2 + 1)) for m in env.modules)
                )

            constraints = []
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
            margin = 1e-4 * max(abs(value), abs(constant_price))
            if value - constant_price > margin:
                obj_value = subproblem_generator.calculate_core_objective(0)
                column = extract_column(obj_value, subproblem_generator, pricing_mip, env.modules[0], p1, p2)

                col_modules = [init_m] if init_m is not None else env.modules
                add_column(column, col_modules, is_initial)

            subproblem_generator.remove_constraints(pricing_mip, constraints)
            if is_initial:
                subproblem_generator.remove_initial_value_constraints(pricing_mip)

        # First, we solve for module-specific initial production cycle columns.
        print(f"Iter {iter}: solving initial production cycles")
        for module in env.modules:
            if initial_deploy_periods[module] is not None:
                for p2 in initial_deploy_periods[module].extract_periods:
                    do_price(planning_start_period.index, p2.index + 1, module)

        # Then, we solve all non-initial periods, where the modules are assumed to be interchangable.
        print(f"Iter {iter}: solving all production cycle ranges")
        for p1 in env.periods:
            for p2 in p1.extract_periods:
                do_price(p1.index, p2.index + 1, None)

        # Add new columns
        # If we added at least one column, try again.
        if len(master_columns) == n_columns_before:
            break

    # Optimize non-relaxed (binary) problem
    for _, _, v in master_columns:
        v.VType = gp.GRB.BINARY

    rmp.optimize()

    return DecompCyclesSolution()


def initial_columns(
    environment: Environment,
    objective_profile: ObjectiveProfile,
    allow_transfer: bool,
    add_symmetry_breaks: bool,
    initial_deploy_periods: Dict[Module, Period | None],
):
    old_gmpg = GurobiMasterProblemGenerator(
        environment,
        objective_profile=objective_profile,
        allow_transfer=allow_transfer,
        add_symmetry_breaks=add_symmetry_breaks,
    )

    subproblem_generator = old_gmpg.sub_problem_generator()
    model = subproblem_generator.build_model()
    model.params.SolutionLimit = 1
    model.optimize()

    for module, init_p in initial_deploy_periods.items():
        if init_p is None:
            continue

        p1 = min(environment.periods, key=lambda p: p.index).index
        p2 = p1
        assert get_module_biomass(subproblem_generator, module, p1) > 1e-5
        while get_module_biomass(subproblem_generator, module, p2) > 1e-5:
            p2 += 1

        col = extract_column(0.0, subproblem_generator, model, module, p1, p2)
        yield (module, col)


def extract_column(
    obj_value: float, subproblem_generator: GurobiProblemGenerator, model: gp.Model, module: Module, p1: int, p2: int
) -> List[DecompCyclesColumn]:
    env = subproblem_generator.environment
    periods = env.period_indices
    # print("EXCTRACT COL ", p1, p2, periods[p1], periods[p2])
    p1_is_deploy = subproblem_generator.smolt_deployed_variable(module, periods[p1]).X > 0.5

    # P1 should either be the first period, or it should be a deploy period
    # assert p1_is_deploy or p1 == subproblem_generator.environment.periods[0]

    # Extract solution
    periodinfo: List[ColumnPeriodInfo] = []
    for p in range(p1, p2 + 1):
        period = periods[p]
        biomass = get_module_biomass(subproblem_generator, module, p)
        extracted = get_module_extracted(subproblem_generator, module, p)

        periodinfo.append(ColumnPeriodInfo(period, biomass, extracted))

    # The last period should be empty
    assert periodinfo[-1].biomass < 1e-5

    return DecompCyclesColumn(obj_value, periodinfo)


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
    if period_idx not in period.extract_periods:
        return 0.0

    return sum(
        (
            subproblem_generator.extract_weight_variable(dep_p, tank, period).X
            for dep_p in period.deploy_periods
            for tank in module.tanks
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
    # and disabled below)

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
