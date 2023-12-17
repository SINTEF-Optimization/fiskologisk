from dataclasses import dataclass
import time
import sys
from typing import Dict, Tuple
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
class DecompCyclesColumn:
    pass


@dataclass
class ShadowPrices:
    period_biomass: Dict[int, float]
    yearly_production: Dict[int, float]
    module_period: Dict[Module, Dict[int, float]]
    module_initial: Dict[Module, float]


def build_subproblem(
    environment: Environment,
    objective_profile: ObjectiveProfile = ObjectiveProfile.PROFIT,
    allow_transfer: bool = True,
    add_symmetry_breaks: bool = False,
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
    return (sub_problem_generator, model)


def decomp_cycles_solve(
    environment: Environment,
    objective_profile: ObjectiveProfile = ObjectiveProfile.PROFIT,
    allow_transfer: bool = True,
    add_symmetry_breaks: bool = False,
) -> DecompCyclesSolution:
    # (relaxed) Restricted master problem
    rmp = gp.Model()

    # Set limit on total biomass each period (6.22)
    regulation_rescale = len(environment.tanks) / environment.parameters.tanks_in_regulations
    max_mass = environment.parameters.max_total_biomass * regulation_rescale
    bio_cstr = {
        p: rmp.addConstr(gp.LinExpr() <= max_mass, name="max_biomass_%s" % p.index) for p in environment.periods
    }

    # Set limit on yearly production (6.23)
    max_prod = environment.parameters.max_yearly_production * regulation_rescale
    prod_constr = {
        y.year: rmp.addConstr(gp.LinExpr() <= max_prod, name="max_year_prod_%s" % y.year) for y in environment.years
    }

    # Set limit on each module for selecting only one prod-cycle column in each time period
    pack_cstr = {
        m: {p: rmp.addConstr(gp.LinExpr() <= 1, name=f"pack_m{m.index}_p{p.index}") for p in environment.periods}
        for m in environment.modules
    }

    # Initial production cycles
    init_cstr = {m: rmp.addConstr(gp.LinExpr() == 1, name=f"init_m{m.index}") for m in environment.modules}

    master_columns: Dict[gp.Var, DecompCyclesColumn] = {}

    def add_column(col: DecompCyclesColumn):
        pass

    subproblem_generator, pricing_mip = build_subproblem(
        environment, objective_profile, allow_transfer, add_symmetry_breaks
    )

    # TODO add initial columsn

    while True:
        # Optimized relaxed restricted master problem
        rmp.optimize()

        # Get shadow prices
        shadow_prices = ShadowPrices(
            period_biomass={p.index: c.Pi for p, c in bio_cstr.items()},
            yearly_production={y: c.Pi for y, c in prod_constr.items()},
            module_period={m: {p: c.pi for p, c in ps.items()} for m, ps in pack_cstr.items()},
            module_initial={m: c.Pi for m, c in init_cstr.items()},
        )

        # Set up solving of the pricing problem
        n_columns_before = len(master_columns)
        subproblem_generator.set_subproblem_objective(
            pricing_mip, shadow_prices.period_biomass, shadow_prices.yearly_production
        )

        def do_price(p1, p2, initial_for_module):
            # master_columns[x] = y
            raise Exception()

        # First, we solve for module-specific initial production cycle columns.
        for module in environment.modules:
            if any(t.initial_weight > 1e-5 for t in module.tanks):
                initial_period = next(p for p in environment.periods if p.index == 16)  # TODO
                for p2 in initial_period.extract_periods:
                    do_price(initial_period, p2, module)

        # Then, we solve all non-initial period, where the modules are assumed to be interchangable.
        for p1 in environment.periods:
            for p2 in p1.extract_periods:
                do_price(p1, p2, None)

        # Add new columns
        # If we added at least one column, try again.
        if len(master_columns) == n_columns_before:
            break

    # Optimize non-relaxed (binary) problem
    for v in master_columns.keys():
        v.VType = gp.GRB.BINARY

    rmp.optimize()

    return DecompCyclesSolution()  # TODO
