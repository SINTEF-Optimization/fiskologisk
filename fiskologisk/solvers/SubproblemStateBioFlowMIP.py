from collections import defaultdict
from dataclasses import dataclass
from math import prod
from typing import Dict, List, Optional, Set, Tuple
import gurobipy as grb

from fiskologisk.domain.Environment import Environment

#
# STATE+BIOMASS FLOW MIP IN STATE-EXPANDED NETWORK
#
# This is an experiment from 2023-11-27 that was not particularly successful, 
# and is not currently used anywhere.
# 
# We solve a single-module full-horizon planning problem by creating a state 
# space graph. As in the DP approaches, the node in state space graph determines 
# the time and the age of the fish and the number of tanks in use, but does not 
# discretize the biomass. We solve the planning problem as a two flow problem over the 
# state space graph where the binary *state flow* determines a path through one state
# for each time step, and the *biomass flow* from one node to a successor is deactivated
# when the state flow does not go over this edge.
#
#


@dataclass(frozen=True)
class StateKey:
    time: int
    age: int
    tanks_in_use: int
    tanks_cleaning: int


@dataclass
class DPSolution:
    objective_value: float
    period_tanks: List[Tuple[int, int]]


def solve_state_bio_flow_mip(
    environment: Environment,
    module_idx: int,
    period_biomass_duals: dict[int, float],
    yearly_production_duals: dict[int, float],
) -> Optional[DPSolution]:
    #
    # Check some problem definition limits for the algorithm.
    #

    module_tanks = environment.modules[module_idx].tanks
    module_tanks = module_tanks[:1]
    if len(module_tanks) >= 6:
        print("Warning: DP heuristic solver maybe unsuitable with a large number of tanks in a module")

    planning_start_time = min(p.index for p in environment.periods)
    planning_end_time = max(p.index for p in environment.periods)
    assert len(set(p.index for p in environment.periods)) == planning_end_time - planning_start_time + 1

    for tank in environment.modules[module_idx].tanks:
        print(
            "TANK initial deploy ",
            tank.initial_deploy_period,
            "initial use",
            tank.initial_use,
            "initial weight",
            tank.initial_weight,
        )

    initial_tanks_in_use = [tank for tank in module_tanks if tank.initial_use and tank.initial_weight >= 0.01]
    print("Initially used tanks", [t.index for t in initial_tanks_in_use])

    initial_biomass = sum(tank.initial_weight for tank in initial_tanks_in_use)
    initial_tanks_cleaning = [tank for tank in module_tanks if tank.initial_use and tank.initial_weight < 0.01]

    print([f"Initial deploy period {tank.index} {tank.initial_deploy_period}" for tank in initial_tanks_in_use])

    min_initial_age = min(planning_start_time - tank.initial_deploy_period for tank in initial_tanks_in_use)

    max_initial_age = max(planning_start_time - tank.initial_deploy_period for tank in initial_tanks_in_use)

    assert all(planning_start_time - tank.initial_deploy_period > 0 for tank in initial_tanks_in_use)
    if min_initial_age != max_initial_age:
        print("Warning: initial tank deploy period", [tank.initial_deploy_period for tank in initial_tanks_in_use])
    assert min_initial_age == max_initial_age

    initial_age = min_initial_age

    min_tank_volume = min(1.0 / t.inverse_volume for t in module_tanks)
    max_tank_volume = max(1.0 / t.inverse_volume for t in module_tanks)

    if max_tank_volume / min_tank_volume >= 1.2:
        raise Exception("DP heuristic solver does not support uneven tank sizes")

    #
    # Compute problem parameters in algorithm-specific format for DP heuristic.
    #

    avg_tank_volume = sum(1.0 / t.inverse_volume for t in module_tanks) / len(module_tanks)

    period_production_duals = {}
    for year, price in yearly_production_duals.items():
        for env_year in environment.years:
            if env_year.year == year:
                for period in env_year.periods:
                    assert period.index not in period_production_duals
                    period_production_duals[period.index] = price

    # assert sum(period_production_duals.values()) < 1e-5
    # assert sum(yearly_production_duals.values()) < 1e-5
    # assert sum(period_biomass_duals.values()) < 1e-5

    # A map from deploy period and age to price,
    # where the price is the integral of the probability distribution of
    # weight classes multiplied by the post-smolt price for that weight class.
    # NOTE: post-smolt prices for a weight class are not time-variant,
    #       (though the problem definition here supports it)
    post_smolt_sell_price = {
        dep_p.index: {
            p.index: 1.0
            * (
                -period_production_duals[p.index]
                + sum(
                    frac * cls.post_smolt_revenue
                    for cls, frac in zip(
                        environment.weight_classes,
                        dep_p.periods_after_deploy_data[p.index].weight_distribution,
                    )
                )
            )
            for p in dep_p.postsmolt_extract_periods
        }
        for dep_p in environment.release_periods
    }

    # Same, but the harvest yield (being <100%) is included in the price.
    print("## harvest yield", environment.parameters.harvest_yield)
    harvest_sell_price = {
        dep_p.index: {
            p.index: -period_production_duals[p.index]
            + environment.parameters.harvest_yield
            * sum(
                frac * cls.harvest_revenue
                for cls, frac in zip(
                    environment.weight_classes,
                    dep_p.periods_after_deploy_data[p.index].weight_distribution,
                )
            )
            for p in dep_p.harvest_periods
        }
        for dep_p in environment.release_periods
    }

    biomass_costs = {
        dep_p.index: {
            p.index: dep_p.periods_after_deploy_data[p.index].oxygen_cost
            + environment.parameters.marginal_increase_pumping_cost
            + period_biomass_duals[p.index]
            for p in dep_p.periods_after_deploy
        }
        for dep_p in environment.release_periods
    }

    biomass_costs_feed = {
        dep_p.index: {p.index: dep_p.periods_after_deploy_data[p.index].feed_cost for p in dep_p.periods_after_deploy}
        for dep_p in environment.release_periods
    }

    transfer_periods = {
        dep_p.index: {p.index: True for p in dep_p.transfer_periods} for dep_p in environment.release_periods
    }

    loss_factor = 1.0 - environment.parameters.monthly_loss

    growth_factors = {
        dep_p.index: {
            p.index: loss_factor * dep_p.periods_after_deploy_data[p.index].growth_factor
            for p in dep_p.periods_after_deploy
        }
        for dep_p in environment.release_periods
    }

    transfer_growth_factors = {
        dep_p.index: {
            p.index: loss_factor * dep_p.periods_after_deploy_data[p.index].transferred_growth_factor
            for p in dep_p.periods_after_deploy
        }
        for dep_p in environment.release_periods
    }

    accumulated_growth_factors = {
        dep_p.index: {
            p.index: prod(
                (
                    loss_factor * dep_p.periods_after_deploy_data[prev_p.index].growth_factor
                    for prev_p in dep_p.periods_after_deploy
                    if prev_p.index < p.index
                )
            )
            for p in dep_p.periods_after_deploy
        }
        for dep_p in environment.release_periods
    }

    accumulated_minimum_growth_adjustment_factors = {
        dep_p.index: {
            p.index: min(
                (
                    dep_p.periods_after_deploy_data[prev_p.index].transferred_growth_factor
                    / dep_p.periods_after_deploy_data[prev_p.index].growth_factor
                    for prev_p in dep_p.periods_after_deploy
                    if prev_p.index < p.index
                    if p.index >= min((x.index for x in dep_p.transfer_periods), default=-1)
                ),
                default=1.0,
            )
            for p in dep_p.periods_after_deploy
        }
        for dep_p in environment.release_periods
    }

    print(f"# Montlyloss = {loss_factor}")
    print("# Valid periods")
    for dep_p in environment.release_periods:
        print(" - Release at ", dep_p.index)
        for p in dep_p.periods_after_deploy:
            data = dep_p.periods_after_deploy_data[p.index]
            print(
                f"   - p={p.index} G={growth_factors[dep_p.index][p.index]} G^T={data.transferred_growth_factor}",
                f"accum_G={accumulated_growth_factors[dep_p.index][p.index]}",
                f"accum_G_min={accumulated_minimum_growth_adjustment_factors[dep_p.index][p.index]}",
                f"costs={biomass_costs[dep_p.index][p.index]}",
                "DEPLOY" if p.index == dep_p.index else "",
                f"POSTSMOLT p={post_smolt_sell_price[dep_p.index][p.index]}"
                if p in dep_p.postsmolt_extract_periods
                else "",
                f"HARVEST  p={harvest_sell_price[dep_p.index][p.index]}" if p in dep_p.harvest_periods else "",
                "TRANSFER" if p in dep_p.transfer_periods else "",
            )

    max_biomass_per_tank = environment.parameters.max_tank_density * avg_tank_volume
    print("## max biomass per tank", max_biomass_per_tank)
    print(
        "## deploy limits",
        environment.parameters.min_deploy_smolt,
        environment.parameters.max_deploy_smolt,
    )

    max_module_use_length = 25

    problem_json = {
        # PARAMETERS
        "volume_bins": 500,
        "max_module_use_length": max_module_use_length,
        "num_tanks": len(module_tanks),
        "planning_start_time": planning_start_time,
        "planning_end_time": planning_end_time,
        "initial_biomass": initial_biomass,
        "initial_tanks_in_use": len(initial_tanks_in_use),
        "initial_tanks_cleaning": len(initial_tanks_cleaning),
        "initial_age": initial_age,
        "smolt_deploy_price": environment.parameters.smolt_price,
        "max_deploy": environment.parameters.max_deploy_smolt,
        "min_deploy": environment.parameters.min_deploy_smolt,
        "tank_const_cost": environment.parameters.min_tank_cost,
        "max_biomass_per_tank": max_biomass_per_tank,
        # TABLES INDEXED ON BY TUPLE (DEPLOY_TIME,AGE)
        # TODO: clean this up a bit by putting all the tables into `deploy_period_data`.
        "post_smolt_sell_price": post_smolt_sell_price,
        "harvest_sell_price": harvest_sell_price,
        "biomass_costs": biomass_costs,
        "biomass_costs_feed": biomass_costs_feed,
        "transfer_periods": transfer_periods,
        "monthly_growth_factors": growth_factors,
        "monthly_growth_factors_transfer": transfer_growth_factors,
        "accumulated_growth_factors": accumulated_growth_factors,
        "accumulated_minimum_growth_adjustment_factors": accumulated_minimum_growth_adjustment_factors,
        "deploy_period_data": {},
        "logarithmic_bins": False,
    }

    m = grb.Model()

    @dataclass
    class StateEdge:
        source: StateKey
        target: StateKey
        cost: float

    @dataclass
    class BiomassEdge:
        source: StateKey | None
        target: StateKey | None
        cost: float
        growth_factor: float

    # Initial state
    initial_state: StateKey
    if len(initial_tanks_in_use) > 0:
        initial_state = StateKey(
            planning_start_time, initial_age, len(initial_tanks_in_use), len(initial_tanks_cleaning)
        )
    else:
        initial_state = StateKey(planning_start_time - 1, 0, 0, 0)

    states: Set[StateKey] = set()
    state_edges: List[StateEdge] = []
    biomass_edges: List[BiomassEdge] = []
    queue: List[StateKey] = [initial_state]

    while len(queue) > 0:
        s1 = queue.pop()
        if s1 in states:
            continue
        states.add(s1)

        def succ(edge: StateEdge):
            state_edges.append(edge)
            queue.append(edge.target)

        # Successors
        if s1.time == planning_end_time:
            continue

        deploy_time = s1.time - s1.age
        if s1.tanks_in_use == 0:
            #
            # Empty module
            #
            # Stay empty
            succ(StateEdge(s1, StateKey(s1.time + 1, 0, 0, 0), 0.0))

            # Deploy
            if (s1.time + 1) in growth_factors:
                assert (s1.time + 1) in growth_factors[s1.time + 1]

                for deploy_tanks in range(1, len(module_tanks) + 1):
                    s2 = StateKey(s1.time + 1, 0, deploy_tanks, 0)
                    succ(StateEdge(s1, s2, 0.0))
                    biomass_edges.append(BiomassEdge(None, s2, environment.parameters.smolt_price, 1.0))

        else:
            #
            # Non-empty module
            #

            unfed_cost = biomass_costs[deploy_time][s1.time]
            fed_cost = unfed_cost + biomass_costs_feed[deploy_time][s1.time]
            tank_cost = environment.parameters.min_tank_cost * s1.tanks_in_use

            growth_factor = growth_factors[deploy_time][s1.time]
            can_grow_next = (s1.time + 1) in biomass_costs[deploy_time]

            # Grow undisturbed
            if can_grow_next:
                s2 = StateKey(s1.time + 1, s1.age + 1, s1.tanks_in_use, 0)
                succ(StateEdge(s1, s2, tank_cost))
                biomass_edges.append(BiomassEdge(s1, s2, fed_cost, growth_factor))

            # Transfer
            can_transfer = can_grow_next and (
                deploy_time in transfer_periods and s1.time in transfer_periods[deploy_time]
            )

            if can_transfer:
                max_total_tanks_after_transfer = min(s1.tanks_in_use + 2, len(module_tanks) - s1.tanks_cleaning)
                for next_tanks in range(s1.tanks_in_use + 1, max_total_tanks_after_transfer + 1):
                    # This is an approximation of the effective growth made to
                    # fit into the state/edge formalism. In practice, it is only
                    # slightly suboptimal to use this effective growth, so
                    # we accept if for performance reasons.

                    transferred_fraction = (next_tanks - s1.tanks_in_use) / next_tanks
                    effective_growth_factor = (
                        transferred_fraction * transfer_growth_factors[deploy_time][s1.time]
                        + (1.0 - transferred_fraction) * growth_factor
                    )

                    s2 = StateKey(s1.time + 1, s1.age + 1, next_tanks, 0)
                    succ(StateEdge(s1, s2, tank_cost))
                    biomass_edges.append(BiomassEdge(s1, s2, fed_cost, effective_growth_factor))

            # Harvest / post-smolt-harvest
            for is_postsmolt, pricetable in [(True, post_smolt_sell_price), (False, harvest_sell_price)]:
                if deploy_time not in pricetable or s1.time not in pricetable[deploy_time]:
                    continue

                price = pricetable[deploy_time][s1.time]
                biomass_edges.append(BiomassEdge(s1, None, unfed_cost - price, 1.0))
                for harvest_tanks in range(1, s1.tanks_in_use + 1):
                    if s1.tanks_in_use - harvest_tanks > 0:
                        if can_grow_next:
                            s2 = StateKey(s1.time + 1, s1.age + 1, s1.tanks_in_use - harvest_tanks, harvest_tanks)
                            succ(StateEdge(s1, s2, tank_cost))
                            biomass_edges.append(BiomassEdge(s1, s2, fed_cost, 1.0))
                    else:
                        succ(StateEdge(s1, StateKey(s1.time + 1, 0, 0, 0), tank_cost))

    print(f"Created {len(states)} nodes, {len(state_edges)} state edges and {len(biomass_edges)} biomass edges.")

    # Formulation: two sets of edges on the 'state' nodes: (1) state flow (2) biomass flow

    state_flow = {
        (e.source, e.target): m.addVar(vtype=grb.GRB.BINARY, obj=0.0, name=f"x_{e.source},{e.target}")
        for e in state_edges
    }
    biomass_flow = {
        (e.source, e.target): m.addVar(lb=0.0, obj=e.cost, name=f"b_{e.source},{e.target}") for e in biomass_edges
    }
    biomass_edge_idx = {(e.source, e.target): e for e in biomass_edges}

    # Preserve state flow

    state_incoming = defaultdict(list)
    state_outgoing = defaultdict(list)

    for a, b in state_flow.keys():
        state_incoming[b].append(a)
        state_outgoing[a].append(b)

    biomass_incoming = defaultdict(list)
    biomass_outgoing = defaultdict(list)

    for a, b in biomass_flow.keys():
        biomass_incoming[b].append(a)
        biomass_outgoing[a].append(b)

    for s1 in states:
        # Preserve state flow
        is_init = s1 == initial_state
        is_final = s1.time == planning_end_time
        assert not (is_init and is_final)

        incoming_state_flow = grb.LinExpr(1 if is_init else 0) + sum(
            (state_flow[(s2, s1)] for s2 in state_incoming[s1])
        )

        outgoing_state_flow = grb.LinExpr(1 if is_final else 0) + sum(
            (state_flow[(s1, s2)] for s2 in state_outgoing[s1])
        )

        m.addConstr(incoming_state_flow == outgoing_state_flow, name=f"st_fl_{s1}")

        # Preserve biomass flow
        incoming_biomass_flow = (initial_biomass if is_init else 0.0) + sum(
            biomass_edge_idx[(s2, s1)].growth_factor * biomass_flow[(s2, s1)] for s2 in biomass_incoming[s1]
        )
        outgoing_biomass_flow = sum(biomass_flow[(s1, s2)] for s2 in biomass_outgoing[s1])
        m.addConstr(incoming_biomass_flow == outgoing_biomass_flow, name=f"bio_fl_{s1}")

    for e in biomass_edges:
        is_init = s1 == initial_state
        incoming_state_flow = grb.LinExpr(1 if is_init else 0) + sum(
            (state_flow[(s2, s1)] for s2 in state_incoming[s1])
        )
        s1, s2 = e.source, e.target
        if s1 is None:
            assert s2 is not None
            m.addConstr(
                biomass_flow[(s1, s2)] <= s2.tanks_in_use * max_biomass_per_tank * incoming_state_flow,
                name=f"c_{e.source},{e.target}",
            )

        elif s2 is None:
            assert s1 is not None
            m.addConstr(
                biomass_flow[(s1, s2)] <= s1.tanks_in_use * max_biomass_per_tank * incoming_state_flow,
                name=f"c_{e.source},{e.target}",
            )

        else:
            assert (s1, s2) in state_flow
            m.addConstr(
                biomass_flow[(s1, s2)] <= s1.tanks_in_use * max_biomass_per_tank * state_flow[(s1, s2)],
                name=f"c_{e.source},{e.target}",
            )

    # m.setParam("Cuts",0)
    # m.setParam("Heuristics",0)
    m.write("smallmip.lp")
    m.optimize()
    # m.computeIIS()
    # m.write("smallmip.ilp")
    objective_value = m.ObjVal
    out_states = []
    # for (s1,s2),v in sorted(state_flow.items(), key=lambda x: x[0][0].time):
    #     print(s1,s2,v.X)
    for s1 in states:
        is_init = s1 == initial_state
        use_state = (1 if is_init else 0) + sum((state_flow[(s2, s1)].X for s2 in state_incoming[s1]))
        assert use_state >= -1e-5 and use_state <= 1.0 + 1e-5

        if use_state > 0.5:
            out_states.append({"period": s1.time, "num_tanks": s1.tanks_in_use})

    out_states.sort(key=lambda x: x["period"])
    print("OUT STATES")
    for out in out_states:
        print(f" out state {out}")

    retval = DPSolution(
        objective_value,
        [(state["period"], state["num_tanks"]) for state in out_states],
    )
    # print("RETVAL", retval)
    raise Exception()

    return retval
