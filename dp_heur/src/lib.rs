use log::{debug, trace, warn};
use ordered_float::OrderedFloat;
use pyo3::{exceptions::PyAssertionError, prelude::*};
use std::collections::HashMap;

/// A Python module implemented in Rust.
#[pymodule]
fn dp_heur(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(solve_module_json, m)?)?;
    Ok(())
}

#[derive(Clone, Copy, Debug)]
struct ModuleState {
    deploy_age: usize,
    biomass: usize,
    tanks: usize,
}

#[derive(serde::Deserialize)]
struct Problem {
    num_tanks: usize,
    planning_start_time: usize,
    planning_end_time: usize,
    smolt_deploy_price: f32,
    tank_const_cost: f32,
    max_biomass_per_tank: f32,

    // The meaning of map<(a,b),...> is that the first index
    // specifies the deploy period and the second is the age.
    // If a period is missing for the first index, that period is not
    // a deploy period. If an age is missing as the second index,
    // that age is not a post-smolt harvest age.
    post_smolt_sell_price: HashMap<usize, HashMap<usize, f32>>,

    // This is the same but the harvest yield rate is multiplied in.
    harvest_sell_price: HashMap<usize, HashMap<usize, f32>>,

    // Costs per mass
    biomass_costs: HashMap<usize, HashMap<usize, f32>>, // Different cost for every month

    // Periods in which tranfer is possible
    transfer_periods: HashMap<usize, HashMap<usize, bool>>,

    // How much the salmon has grown (accumulated factor)
    // since deployment.
    monthly_growth_factors: HashMap<usize, HashMap<usize, f32>>,
    monthly_growth_factors_transfer: HashMap<usize, HashMap<usize, f32>>,
    accumulated_growth_factors: HashMap<usize, HashMap<usize, f32>>,
    accumulated_minimum_growth_adjustment_factors: HashMap<usize, HashMap<usize, f32>>,

    deploy_period_data: HashMap<usize, HashMap<usize, PeriodSpec>>,

    max_deploy_per_tank: f32,
    min_deploy_per_tank: f32,

    volume_bins: usize,
    max_module_use_length: usize,
}

#[derive(serde::Deserialize)]
struct PeriodSpec {
    harvest_sell_price: f32,
    post_smolt_sell_price: f32,
    biomass_cost: f32,
    can_transfer: bool,
    growth_factor: f32,
    growth_factor_transfer: f32,
    accumulated_growth_factor: f32,
    accumulated_minimum_growth_factor: f32,
}

#[derive(serde::Serialize)]
pub struct Solution {
    actions: Vec<Action>,
}

fn state_biomass_limits(problem: &Problem, time: usize, age: usize, tanks: usize) -> (f32, f32) {
    trace!(
        "Computing biomass limits for t={} age={} p0={}",
        time,
        age,
        time - age
    );
    let deploy_time = time - age;

    assert!(
        problem.accumulated_minimum_growth_adjustment_factors[&deploy_time]
            .contains_key(&(time - 1))
            == problem.accumulated_growth_factors[&deploy_time].contains_key(&(time - 1))
    );

    let minimum_growth_adjustment = if age == 0 {
        1.0
    } else {
        problem.accumulated_minimum_growth_adjustment_factors[&deploy_time][&(time - 1)]
    };

    let maximum_growth = if age == 0 {
        1.0
    } else {
        problem.accumulated_growth_factors[&deploy_time][&(time - 1)]
    };

    let minimum_growth = maximum_growth * minimum_growth_adjustment;

    trace!(
        "  min growth {} max growth {}",
        minimum_growth,
        maximum_growth
    );
    assert!(minimum_growth <= maximum_growth);
    assert!(minimum_growth >= 0.75 * maximum_growth);

    // Asssume that the biomass stays within the limits of smolt deployment and the growth range.
    let minimum_deployed_grown = problem.min_deploy_per_tank * minimum_growth;
    let maximum_deployed_grown = problem.max_deploy_per_tank * maximum_growth * tanks as f32;

    // All the states should satisfy the `max_biomass_per_tank` constraint, so
    // we can also assume the biomass does not exceed this number.
    let maximum_density = problem.max_biomass_per_tank * tanks as f32;

    let minimum_total_biomass = minimum_deployed_grown;
    let maximum_total_biomass = maximum_deployed_grown.min(maximum_density);

    trace!(
        "Biomass limits {} {}",
        minimum_total_biomass,
        maximum_total_biomass
    );

    (minimum_total_biomass, maximum_total_biomass)
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + t * (b - a)
}

fn calc_biomass(problem: &Problem, time: usize, state: &ModuleState) -> f32 {
    let (lb, ub) = state_biomass_limits(problem, time, state.deploy_age, state.tanks);
    let relative_biomass = (state.biomass as f32) / (problem.volume_bins - 1) as f32;
    lerp(lb, ub, relative_biomass)
}

fn round_biomass_level(
    problem: &Problem,
    time: usize,
    age: usize,
    tanks: usize,
    biomass: f32,
) -> usize {
    let (lb, ub) = state_biomass_limits(problem, time, age, tanks);
    let relative_biomass = (biomass - lb) / (ub - lb);
    if relative_biomass < -0.1 || relative_biomass >= 1.1 {
        warn!("Biomass is outside state biomass limits");
    }

    let relative_biomass_clamped = relative_biomass.clamp(0.0, 1.0);
    (relative_biomass_clamped * (problem.volume_bins - 1) as f32).round() as usize
}

fn foreach_successor_state(
    problem: &Problem,
    time: usize,
    state: &ModuleState,
    mut f: impl FnMut(f32, ModuleState),
) {
    let deploy_time = time - state.deploy_age;
    let next_time = time + 1;
    if state.tanks == 0 {
        // We can stay empty ...
        f(0.0, *state);

        // ... or we can start a new production cycle
        let can_deploy = problem.monthly_growth_factors.contains_key(&next_time);

        if can_deploy {
            assert!(problem.monthly_growth_factors[&next_time].contains_key(&next_time));

            for deploy_tanks in 1..=problem.num_tanks {
                for deploy_biomass in 0..problem.volume_bins {
                    let new_state = ModuleState {
                        deploy_age: 0,
                        tanks: deploy_tanks,
                        biomass: deploy_biomass,
                    };

                    let next_biomass = calc_biomass(problem, next_time, &new_state);

                    // NOTE: we don't put the costs for the tank and biomass
                    //       on this edge, we put it on the outgoing edge for the
                    //       period in question.
                    //
                    // let biomass_cost =
                    //     problem.smolt_deploy_price + problem.biomass_costs[&next_time][&next_time];
                    // let tank_cost = problem.tank_const_cost * deploy_tanks as f32;
                    // let cost = next_biomass * biomass_cost + tank_cost;

                    // It shouldn't be possible to violate per-tank biomass constraints when deploying.

                    trace!(
                        " deploying {} kg  max_per_tank={}, max={}",
                        next_biomass,
                        problem.max_biomass_per_tank,
                        deploy_tanks as f32 * problem.max_biomass_per_tank
                    );

                    assert!(
                        next_biomass <= deploy_tanks as f32 * problem.max_biomass_per_tank * 1.001
                    );

                    let cost = next_biomass * problem.smolt_deploy_price;
                    f(cost, new_state);
                }
            }
        }
    } else {
        // Determine costs for the source state.
        trace!(
            "Determine costs for the source state.  {}/{}",
            deploy_time,
            time
        );
        let prev_biomass = calc_biomass(problem, time, state);
        let biomass_cost = problem.biomass_costs[&deploy_time][&time];
        let tank_cost = problem.tank_const_cost * state.tanks as f32;
        let prev_cost = prev_biomass * biomass_cost + tank_cost;

        let growth_factor = problem.monthly_growth_factors[&deploy_time][&time];
        let next_biomass_undisturbed = growth_factor * prev_biomass;

        let can_grow_next = problem.monthly_growth_factors[&deploy_time].contains_key(&next_time);

        let can_grow_undisturbed = can_grow_next
            && next_biomass_undisturbed
                <= state.tanks as f32 * problem.max_biomass_per_tank * 1.0001;

        if can_grow_undisturbed {
            // We can stay where we are, while the fish are growing

            f(
                prev_cost,
                ModuleState {
                    deploy_age: state.deploy_age + 1,
                    biomass: round_biomass_level(
                        problem,
                        time + 1,
                        state.deploy_age + 1,
                        state.tanks,
                        next_biomass_undisturbed,
                    ),
                    tanks: state.tanks,
                },
            );
        }

        // Transfer during this period
        let can_transfer = can_grow_next
            && problem
                .transfer_periods
                .get(&deploy_time)
                .and_then(|t| t.get(&time))
                .copied()
                .unwrap_or(false);

        // let can_transfer = false;

        // println!(
        //     " succ grow_next={},{} transf={} tanks={} deploy={} time={} biomass={} ({}) --> {} ({})",
        //     can_grow_next,
        //     can_grow_undisturbed,
        //     can_transfer,
        //     state.tanks,
        //     deploy_time,
        //     time,
        //     prev_biomass,
        //     state.biomass,
        //     next_biomass_undisturbed,
        //     round_biomass_level(
        //         problem,
        //         time + 1,
        //         state.deploy_age + 1,
        //         state.tanks,
        //         next_biomass_undisturbed
        //     )
        // );

        if can_transfer {
            let transfer_growth_factor =
                problem.monthly_growth_factors_transfer[&deploy_time][&time];

            for next_tanks in (state.tanks + 1)..=(problem.num_tanks) {
                let additional_new_tanks = next_tanks - state.tanks;
                assert!(additional_new_tanks >= 1 && additional_new_tanks <= problem.num_tanks - 1);
                let new_biomass_when_transferring = state.tanks as f32 / next_tanks as f32
                    * next_biomass_undisturbed
                    + additional_new_tanks as f32 / next_tanks as f32
                        * transfer_growth_factor
                        * prev_biomass;

                let can_grow_transfer = can_grow_next
                    && new_biomass_when_transferring
                        <= next_tanks as f32 * problem.max_biomass_per_tank * 1.0001;

                if !can_grow_transfer {
                    continue;
                }

                f(
                    prev_cost,
                    ModuleState {
                        deploy_age: state.deploy_age + 1,
                        biomass: round_biomass_level(
                            problem,
                            time + 1,
                            state.deploy_age + 1,
                            next_tanks,
                            new_biomass_when_transferring,
                        ),
                        tanks: next_tanks,
                    },
                );
            }
        }

        // ... or we can harvest post-smolt if the weight range is correct
        // ... or we can harvest mature fish if the weight range is correct

        for (_is_postsmolt, sell_price_table) in [
            (true, &problem.post_smolt_sell_price),
            (false, &problem.harvest_sell_price),
        ] {
            let revenue_per_weight = sell_price_table
                .get(&deploy_time)
                .and_then(|t| t.get(&time))
                .copied();

            if let Some(revenue_per_weight) = revenue_per_weight {
                for harvest_tanks in 1..=(state.tanks) {
                    let remaining_tanks = state.tanks - harvest_tanks;

                    if remaining_tanks > 0 && !can_grow_next {
                        // There is no growth factor, so the fish should live longer from
                        // this state. We can only harvest everything.

                        continue;
                    }

                    let tank_fraction = harvest_tanks as f32 / state.tanks as f32;

                    let harvested_weight = tank_fraction * next_biomass_undisturbed;
                    let remaining_weight = next_biomass_undisturbed - harvested_weight;

                    let total_revenue = revenue_per_weight * harvested_weight;

                    // TODO oxygen costs also apply?
                    let this_cost = harvest_tanks as f32 * problem.tank_const_cost;
                    let cost = prev_cost + this_cost - total_revenue;

                    if remaining_tanks == 0 {
                        f(
                            cost,
                            ModuleState {
                                deploy_age: 0,
                                biomass: 0,
                                tanks: 0,
                            },
                        );
                    } else {
                        f(
                            cost,
                            ModuleState {
                                deploy_age: state.deploy_age + 1,
                                biomass: round_biomass_level(
                                    problem,
                                    time + 1,
                                    state.deploy_age + 1,
                                    remaining_tanks,
                                    remaining_weight,
                                ),
                                tanks: remaining_tanks,
                            },
                        )
                    }
                }
            }
        }
    }
}

//
// A-star vs. DP
// With DP, the current time is global, but yuo need to expand all reachable states
// With A-star, the current time is part of the state, you need a heurstic and a
// priority queue.

// TODO: Loop through only reachable states: for example, apply a
//       permutation to the state-index-mapping so that unreachable
//       states are grouped together at the end?

#[derive(serde::Serialize)]
struct Action {}

#[pyfunction]
pub fn solve_module_json(problem: &str) -> PyResult<String> {
    let problem: Problem = serde_json::from_str(problem)
        .map_err(|e| PyAssertionError::new_err(format!("Problem parse error: {}", e)))?;

    let solution = solve_module(&problem);
    let solution_json = serde_json::to_string(&solution)
        .map_err(|e| PyAssertionError::new_err(format!("{}", e)))?;

    Ok(solution_json)
}

fn solve_module(problem: &Problem) -> Solution {
    let n_states_per_time =
        problem.volume_bins * problem.num_tanks as usize * problem.max_module_use_length + 1;
    let n_time_steps = problem.planning_end_time - problem.planning_start_time + 1;
    debug!(
        "solving dp_heuristic with {} states in {} time periods",
        n_states_per_time, n_time_steps,
    );

    // let n_states = n_states_per_time * problem.time_steps;

    const UNREACHABLE_NODE: i32 = -1;
    const ROOT_NODE: i32 = -2;

    let mut state_costs = vec![f32::INFINITY; n_states_per_time];
    let mut state_nodes: Vec<i32> = vec![UNREACHABLE_NODE; n_states_per_time];

    let mut next_state_costs = vec![f32::INFINITY; n_states_per_time];
    let mut next_state_nodes: Vec<i32> = vec![UNREACHABLE_NODE; n_states_per_time];

    struct Node {
        prev_state: u32,
        prev_node: i32,
    }

    let mut nodes: Vec<Node> = Vec::new();
    nodes.reserve(n_states_per_time * n_time_steps / 4);

    // TODO: what about non-empty initial state?
    let initial_state_idx = 0;
    state_costs[initial_state_idx] = 0.0;
    state_nodes[initial_state_idx] = ROOT_NODE;

    // State idxs are stored as follows:
    //  * state 0 = tanks empty
    //  * state i>0 = i-1 is converted from (age,biomass,tanks) as follows:
    //    biomass + (biomass_levels)*age + (biomass_levels*max_use_length)*tanks

    for current_time in (problem.planning_start_time - 1)..=problem.planning_end_time {
        for state_idx in 0..n_states_per_time {
            let state = idx_to_state(state_idx, problem);

            assert!(state_idx == state_to_idx(&state, problem));

            if state_costs[state_idx].is_infinite() {
                trace!("State unreachable t={} {:?}", current_time, state);
                // Unreachable state.
                continue;
            }
            // println!("FINDING SUCCS OF State t={} {:?}", current_time, state);

            foreach_successor_state(problem, current_time, &state, |cost, next: ModuleState| {
                let next_state_idx = state_to_idx(&next, problem);
                let total_cost = state_costs[state_idx] + cost;
                // println!("  next({}) {:?}", next_state_idx, next);

                if next_state_costs[next_state_idx] > total_cost {
                    let new_node_idx = nodes.len() as i32;
                    let prev_node = state_nodes[state_idx];
                    assert!(prev_node >= 0 || prev_node == ROOT_NODE);
                    nodes.push(Node {
                        prev_state: state_to_idx(&state, problem) as u32,
                        prev_node: prev_node as i32,
                    });
                    next_state_costs[next_state_idx] = total_cost;
                    next_state_nodes[next_state_idx] = new_node_idx;
                }
            });
        }

        std::mem::swap(&mut state_costs, &mut next_state_costs);
        std::mem::swap(&mut state_nodes, &mut next_state_nodes);
        next_state_costs.fill(f32::INFINITY);
        next_state_nodes.fill(UNREACHABLE_NODE);
    }

    // Find the best final state and trace back to create plan

    fn mk_action(s1: &ModuleState, s2: &ModuleState) -> Action {
        Action {}
    }

    let best_state_idx = state_costs
        .iter()
        .enumerate()
        .min_by_key(|(_i, x)| OrderedFloat(**x))
        .unwrap()
        .0;

    let objective = state_costs[best_state_idx];
    let mut node = &nodes[state_nodes[best_state_idx] as usize];
    let mut states: Vec<(usize, usize)> = vec![(problem.planning_end_time + 1, best_state_idx)];
    let mut t = problem.planning_end_time;

    while node.prev_node >= 0 {
        t -= 1;
        node = &nodes[node.prev_node as usize];
        states.push((t, node.prev_state as usize));
    }

    assert!(t == problem.planning_start_time - 1);
    states.reverse();

    println!(
        "total number of nodes: {}  (initial allocation {})",
        nodes.len(),
        n_states_per_time * n_time_steps / 4
    );
    println!(
        "states (horison: {}-{}): {:?}",
        problem.planning_start_time, problem.planning_end_time, states
    );

    for (t, state_idx) in states {
        let state = idx_to_state(state_idx, problem);
        let biomass = if state.tanks == 0 {
            0.0
        } else {
            calc_biomass(problem, t, &state)
        };

        let (lb, ub) = if state.tanks == 0 {
            (0., 0.)
        } else {
            state_biomass_limits(problem, t, state.deploy_age, state.tanks)
        };
        println!(
            "t {} tanks={} age={} biomass={}-- {}  [{},{}]",
            t, state.tanks, state.deploy_age, state.biomass, biomass, lb, ub
        );
    }

    println!("Best solution has cost {}", objective);
    Solution { actions: vec![] }
}

fn state_to_idx(next: &ModuleState, problem: &Problem) -> usize {
    assert!(next.biomass < problem.volume_bins);
    assert!(next.deploy_age < problem.max_module_use_length);
    assert!(next.tanks <= problem.num_tanks);

    let next_state_idx = if next.tanks == 0 {
        assert!(next.biomass == 0);
        assert!(next.deploy_age == 0);
        0
    } else {
        1 + next.biomass
            + problem.volume_bins * next.deploy_age
            + (problem.volume_bins * problem.max_module_use_length) * (next.tanks - 1)
    };
    next_state_idx
}

fn idx_to_state(idx: usize, problem: &Problem) -> ModuleState {
    if idx == 0 {
        ModuleState {
            biomass: 0,
            deploy_age: 0,
            tanks: 0,
        }
    } else {
        let tank_stride = problem.volume_bins * problem.max_module_use_length;
        let age_stride = problem.volume_bins;

        let tank_idx = idx - 1;
        let tanks = tank_idx / tank_stride + 1;

        let age_idx = tank_idx % tank_stride;
        let deploy_age = age_idx / age_stride;

        let biomass = age_idx % age_stride;
        ModuleState {
            tanks,
            biomass,
            deploy_age,
        }
    }
}
