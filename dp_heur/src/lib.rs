use log::{debug, trace};
use ordered_float::OrderedFloat;
use pyo3::{exceptions::PyAssertionError, prelude::*};
use std::collections::HashMap;

#[pymodule]
fn dp_heur(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(solve_module_json, m)?)?;
    Ok(())
}

#[pyfunction]
pub fn solve_module_json(problem: &str) -> PyResult<String> {
    let problem: Problem = serde_json::from_str(problem)
        .map_err(|e| PyAssertionError::new_err(format!("Problem parse error: {}", e)))?;

    let solution = solve_module(&problem);
    let solution_json = serde_json::to_string(&solution)
        .map_err(|e| PyAssertionError::new_err(format!("{}", e)))?;

    Ok(solution_json)
}

#[derive(serde::Deserialize, Debug)]
struct Problem {
    num_tanks: usize,
    planning_start_time: usize,
    planning_end_time: usize,
    smolt_deploy_price: f32,
    tank_const_cost: f32,
    max_biomass_per_tank: f32,

    initial_age: usize,
    initial_tanks_in_use: usize,
    initial_tanks_cleaning: usize,
    initial_biomass: f32,

    // The meaning of map<(a,b),...> is that the first index
    // specifies the deploy period and the second is the age.
    // If a period is missing for the first index, that period is not
    // a deploy period. If an age is missing as the second index,
    // that age is not a post-smolt harvest age.

    // TODO the distinction between post_smolt and harvest might not be necessary?
    post_smolt_sell_price: HashMap<usize, HashMap<usize, f32>>,

    // This is the same but the harvest yield rate is multiplied in.
    harvest_sell_price: HashMap<usize, HashMap<usize, f32>>,

    // Costs per mass
    biomass_costs: HashMap<usize, HashMap<usize, f32>>, // Different cost for every month

    // TODO split the feeding costs into a separate cost table
    // so that feeding doesn't apply to the harvest month.
    biomass_costs_feed: HashMap<usize, HashMap<usize, f32>>,

    // Periods in which tranfer is possible
    transfer_periods: HashMap<usize, HashMap<usize, bool>>,

    // How much the salmon has grown (accumulated factor)
    // since deployment.
    monthly_growth_factors: HashMap<usize, HashMap<usize, f32>>,
    monthly_growth_factors_transfer: HashMap<usize, HashMap<usize, f32>>,

    max_deploy: f32,
    min_deploy: f32,

    volume_bins: usize,
    max_module_use_length: usize,

    logarithmic_bins: bool,
}

#[derive(serde::Serialize)]
pub struct Solution {
    objective: f32,
    states: Vec<SolutionState>,
}

#[derive(serde::Serialize)]
struct SolutionState {
    deploy_period: i32,
    period: u32,
    biomass: f32,
    num_tanks: u32,
    num_tanks_cleaning: u32,
}

#[derive(Clone, Copy, Debug)]
struct ModuleState {
    deploy_age: usize,
    biomass: usize,
    tanks_in_use: usize,
    tanks_being_cleaned: usize,
}

fn state_to_idx(next: &ModuleState, problem: &Problem) -> usize {
    // State idxs are stored as follows:
    //  * state 0 = tanks empty
    //  * state i>0 = i-1 is converted from (age,biomass,tanks,cleaning) as follows:
    //    biomass + (biomass_levels)*age
    //            + (biomass_levels*max_use_length)*tanks
    //            + (biomass_levels*max_use_length*num_tanks)*cleaning

    assert!(next.biomass < problem.volume_bins);
    assert!(next.deploy_age < problem.max_module_use_length);
    assert!(next.tanks_in_use <= problem.num_tanks);

    let next_state_idx = if next.tanks_in_use == 0 {
        assert!(next.biomass == 0);
        assert!(next.deploy_age == 0);
        assert!(next.tanks_being_cleaned == 0);
        0
    } else {
        // If no tanks are in use, we should be in the 0 state.
        assert!(next.tanks_in_use >= 1);

        // If all tanks are being cleaned, we should be in the 0 state.
        assert!(next.tanks_being_cleaned < problem.num_tanks);

        let cleaning_stride =
            problem.volume_bins * problem.max_module_use_length * problem.num_tanks;
        let tank_stride = problem.volume_bins * problem.max_module_use_length;
        let age_stride = problem.volume_bins;

        1 + next.biomass
            + age_stride * next.deploy_age
            + tank_stride * (next.tanks_in_use - 1)
            + cleaning_stride * next.tanks_being_cleaned
    };
    next_state_idx
}

fn idx_to_state(idx: usize, problem: &Problem) -> ModuleState {
    if idx == 0 {
        ModuleState {
            biomass: 0,
            deploy_age: 0,
            tanks_in_use: 0,
            tanks_being_cleaned: 0,
        }
    } else {
        let cleaning_stride =
            problem.volume_bins * problem.max_module_use_length * problem.num_tanks;
        let tank_stride = problem.volume_bins * problem.max_module_use_length;
        let age_stride = problem.volume_bins;

        let cleaning_idx = idx - 1;
        let tanks_being_cleaned = cleaning_idx / cleaning_stride;

        let tank_idx = cleaning_idx % cleaning_stride;
        let tanks_in_use = tank_idx / tank_stride + 1;

        let age_idx = tank_idx % tank_stride;
        let deploy_age = age_idx / age_stride;

        let biomass = age_idx % age_stride;
        ModuleState {
            tanks_in_use,
            biomass,
            deploy_age,
            tanks_being_cleaned,
        }
    }
}

fn state_biomass_limits(problem: &Problem, time: usize, age: usize, tanks: usize) -> (f32, f32) {
    let maximum_total_biomass = problem.max_biomass_per_tank * tanks as f32;
    let minimum_total_biomass = problem.min_deploy.min(maximum_total_biomass);
    (minimum_total_biomass, maximum_total_biomass)
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + t * (b - a)
}

fn calc_biomass(problem: &Problem, time: usize, state: &ModuleState) -> f32 {
    let (lb, ub) = state_biomass_limits(problem, time, state.deploy_age, state.tanks_in_use);
    let relative_biomass = (state.biomass as f32) / (problem.volume_bins - 1) as f32;
    if problem.logarithmic_bins {
        10.0f32.powf(lerp(lb.log10(), ub.log10(), relative_biomass))
    } else {
        lerp(lb, ub, relative_biomass)
    }
}

fn round_biomass_level(
    problem: &Problem,
    time: usize,
    age: usize,
    tanks: usize,
    biomass: f32,
) -> usize {
    let (lb, ub) = state_biomass_limits(problem, time, age, tanks);

    let relative_biomass = if problem.logarithmic_bins {
        (biomass.log10() - lb.log10()) / (ub.log10() - lb.log10())
    } else {
        (biomass - lb) / (ub - lb)
    };

    if relative_biomass < -0.1 {
        println!(
            "time {} age {} tanks {} bio {} LB {} UB {}",
            time, age, tanks, biomass, lb, ub
        );
        panic!("Biomass is outside state biomass limits");
    }

    let relative_biomass_clamped = relative_biomass.clamp(0.0, 1.0);
    (relative_biomass_clamped * (problem.volume_bins - 1) as f32).round() as usize
}

#[derive(Clone, Copy, Debug)]
struct Action {
    harvest: f32,
    transfer: f32,
}

fn foreach_successor_state(
    problem: &Problem,
    prev_time: usize,
    prev_state: &ModuleState,
    mut f: impl FnMut(f32, ModuleState),
) {
    let deploy_time = prev_time - prev_state.deploy_age;
    let next_time = prev_time + 1;
    if prev_state.tanks_in_use == 0 {
        // We can stay empty ...
        f(
            0.0,
            // Action {
            //     harvest: 0.0,
            //     transfer: 0.0,
            // },
            *prev_state,
        );

        // ... or we can start a new production cycle
        let can_deploy = problem.monthly_growth_factors.contains_key(&next_time);

        if can_deploy {
            assert!(problem.monthly_growth_factors[&next_time].contains_key(&next_time));

            for deploy_tanks in 1..=problem.num_tanks {
                for deploy_biomass in 0..problem.volume_bins {
                    let new_state = ModuleState {
                        deploy_age: 0,
                        tanks_in_use: deploy_tanks,
                        biomass: deploy_biomass,
                        tanks_being_cleaned: 0,
                    };

                    let next_biomass = calc_biomass(problem, next_time, &new_state);

                    // NOTE: we don't put the costs for the tank and biomass
                    //       on this edge, we put it on the outgoing edge for the
                    //       period in question.
                    //
                    // It shouldn't be possible to violate per-tank biomass constraints when deploying.

                    assert!(
                        next_biomass <= deploy_tanks as f32 * problem.max_biomass_per_tank * 1.001
                    );

                    let cost = next_biomass * problem.smolt_deploy_price;
                    f(
                        cost,
                        // Action {
                        //     harvest: 0.0,
                        //     transfer: 0.0,
                        // },
                        new_state,
                    );
                }
            }
        }
    } else {
        // Determine costs for the source state.
        trace!(
            "Determine costs for the source state.  {}/{}",
            deploy_time,
            prev_time
        );

        let prev_biomass = calc_biomass(problem, prev_time, prev_state);
        let unfed_cost = problem.biomass_costs[&deploy_time][&prev_time];
        let fed_cost = unfed_cost + problem.biomass_costs_feed[&deploy_time][&prev_time];

        let prev_tank_cost = problem.tank_const_cost * prev_state.tanks_in_use as f32;
        let prev_cost = prev_biomass * fed_cost + prev_tank_cost;

        let growth_factor = problem.monthly_growth_factors[&deploy_time][&prev_time];
        let next_biomass_undisturbed = growth_factor * prev_biomass;

        let can_grow_next = problem.biomass_costs[&deploy_time].contains_key(&next_time);

        let can_grow_undisturbed = can_grow_next
            && next_biomass_undisturbed
                <= prev_state.tanks_in_use as f32 * problem.max_biomass_per_tank * 1.0001;

        if can_grow_undisturbed {
            // We can stay where we are, while the fish are growing

            f(
                prev_cost,
                // Action {
                //     harvest: 0.0,
                //     transfer: 0.0,
                // },
                ModuleState {
                    deploy_age: prev_state.deploy_age + 1,
                    biomass: round_biomass_level(
                        problem,
                        prev_time + 1,
                        prev_state.deploy_age + 1,
                        prev_state.tanks_in_use,
                        next_biomass_undisturbed,
                    ),
                    tanks_in_use: prev_state.tanks_in_use,
                    tanks_being_cleaned: 0,
                },
            );
        }

        // Transfer during this period
        let can_transfer = can_grow_next
            && problem
                .transfer_periods
                .get(&deploy_time)
                .and_then(|t| t.get(&prev_time))
                .copied()
                .unwrap_or(false);

        if can_transfer {
            let transfer_growth_factor =
                problem.monthly_growth_factors_transfer[&deploy_time][&prev_time];

            let max_total_tanks_after_transfer = (problem.num_tanks
                - prev_state.tanks_being_cleaned)
                .min(prev_state.tanks_in_use + 2);

            for next_tanks in (prev_state.tanks_in_use + 1)..=max_total_tanks_after_transfer {
                let additional_tanks = next_tanks - prev_state.tanks_in_use;
                assert!(additional_tanks >= 1 && additional_tanks <= problem.num_tanks - 1);

                let untransferred_fraction = prev_state.tanks_in_use as f32 / next_tanks as f32;
                let untransferred_weight = growth_factor * prev_biomass * untransferred_fraction;

                let transferred_fraction = additional_tanks as f32 / next_tanks as f32;
                let transferred_weight =
                    transfer_growth_factor * prev_biomass * transferred_fraction;

                let new_biomass_when_transferring = untransferred_weight + transferred_weight;

                let can_grow_transfer = can_grow_next
                    && new_biomass_when_transferring
                        <= next_tanks as f32 * problem.max_biomass_per_tank * 1.0001;

                if !can_grow_transfer {
                    continue;
                }

                f(
                    prev_cost,
                    // Action {
                    //     harvest: 0.0,
                    //     transfer: transferred_weight,
                    // },
                    ModuleState {
                        deploy_age: prev_state.deploy_age + 1,
                        biomass: round_biomass_level(
                            problem,
                            next_time,
                            prev_state.deploy_age + 1,
                            next_tanks,
                            new_biomass_when_transferring,
                        ),
                        tanks_in_use: next_tanks,
                        tanks_being_cleaned: 0,
                    },
                );
            }
        }

        // ... or we can harvest post-smolt if the weight range is correct
        // ... or we can harvest mature fish if the weight range is correct

        for (is_postsmolt, sell_price_table) in [
            (true, &problem.post_smolt_sell_price),
            (false, &problem.harvest_sell_price),
        ] {
            let revenue_per_weight = sell_price_table
                .get(&deploy_time)
                .and_then(|t| t.get(&prev_time))
                .copied();

            if let Some(revenue_per_weight) = revenue_per_weight {
                for harvest_tanks in 1..=(prev_state.tanks_in_use) {
                    let remaining_tanks = prev_state.tanks_in_use - harvest_tanks;

                    if remaining_tanks > 0 && !can_grow_next {
                        // There is no growth factor, so the fish shouldn't live longer from
                        // this state. We can only harvest everything.

                        continue;
                    }

                    let max_remaining_biomass_level = if remaining_tanks == 0 {
                        0
                    } else {
                        problem.volume_bins - 1
                    };

                    for next_biomass_level in 0..=max_remaining_biomass_level {
                        let next_state = ModuleState {
                            deploy_age: if remaining_tanks == 0 {
                                0
                            } else {
                                prev_state.deploy_age + 1
                            },
                            biomass: next_biomass_level,
                            tanks_in_use: remaining_tanks,
                            tanks_being_cleaned: if remaining_tanks == 0 {
                                0
                            } else {
                                harvest_tanks
                            },
                        };

                        let next_weight = calc_biomass(problem, next_time, &next_state);

                        if next_weight > 0.95 * prev_biomass {
                            // println!("  harvest < 5%");
                            continue;
                        }

                        let prev_remaining_weight = next_weight / growth_factor;
                        let harvested_weight = prev_biomass - prev_remaining_weight;

                        if harvested_weight
                            > 1.0001 * problem.max_biomass_per_tank * harvest_tanks as f32
                        {
                            continue;
                        }

                        // println!("harvest ok");

                        let revenue = revenue_per_weight * harvested_weight;

                        let unfed_weight = harvested_weight;
                        let fed_weight = prev_remaining_weight;

                        let cost =
                            unfed_weight * unfed_cost + fed_weight * fed_cost + prev_tank_cost
                                - revenue;

                        f(
                            cost,
                            // Action {
                            //     harvest: harvested_weight,
                            //     transfer: 0.0,
                            // },
                            next_state,
                        );
                    }
                }
            }
        }
    }
}

fn solve_module(problem: &Problem) -> Solution {
    let n_states_per_time = problem.volume_bins
        * problem.num_tanks as usize
        * problem.max_module_use_length
        * problem.num_tanks
        + 1;

    let (initial_state, first_time) = if problem.initial_tanks_in_use > 0 {
        let state = ModuleState {
            deploy_age: problem.initial_age,
            biomass: round_biomass_level(
                problem,
                problem.planning_start_time,
                problem.initial_age,
                problem.initial_tanks_in_use,
                problem.initial_biomass,
            ),
            tanks_in_use: problem.initial_tanks_in_use,
            tanks_being_cleaned: problem.initial_tanks_cleaning,
        };

        (state, problem.planning_start_time)
    } else {
        let state = ModuleState {
            tanks_in_use: 0,
            deploy_age: 0,
            biomass: 0,
            tanks_being_cleaned: 0,
        };
        (state, problem.planning_start_time - 1)
    };

    trace!("Initial state {:?}", initial_state);

    let n_time_steps = problem.planning_end_time - first_time + 1;
    debug!(
        "solving dp_heuristic with {} states in {} time periods",
        n_states_per_time, n_time_steps,
    );

    const UNREACHABLE_NODE: i32 = -1;
    const ROOT_NODE: i32 = -2;

    #[derive(Clone, Copy)]
    struct Node {
        cost: f32,
        prev_state: i32,
    }

    let mut all_nodes = vec![
        Node {
            cost: f32::INFINITY,
            prev_state: UNREACHABLE_NODE
        };
        n_states_per_time * n_time_steps
    ];

    let mut reachable_states: Vec<u32> = Vec::with_capacity(n_states_per_time);
    let mut next_reachable_states: Vec<u32> = Vec::with_capacity(n_states_per_time);

    let initial_state_idx = state_to_idx(&initial_state, problem);
    all_nodes[initial_state_idx] = Node {
        cost: 0.0,
        prev_state: ROOT_NODE,
    };

    reachable_states.push(initial_state_idx as u32);

    let mut n_states_total = 0;
    let mut n_states_processed = 0;
    let mut n_edges = 0;

    for prev_time in (first_time)..problem.planning_end_time {
        let (before, after) =
            all_nodes.split_at_mut((prev_time + 1 - first_time) * n_states_per_time);
        let prev_time_nodes = &before[((prev_time - first_time) * n_states_per_time)
            ..((prev_time + 1 - first_time) * n_states_per_time)];
        let next_time_nodes = &mut after[0..n_states_per_time];

        for state_idx in reachable_states.iter() {
            // for state_idx in 0..n_states_per_time {
            let prev_state_idx = *state_idx as usize;
            let prev_state = idx_to_state(prev_state_idx, problem);

            assert!(prev_state_idx == state_to_idx(&prev_state, problem));

            n_states_total += 1;
            assert!(!prev_time_nodes[prev_state_idx].cost.is_infinite());
            n_states_processed += 1;

            foreach_successor_state(
                problem,
                prev_time,
                &prev_state,
                |cost, next: ModuleState| {
                    n_edges += 1;
                    let next_state_idx = state_to_idx(&next, problem);
                    let total_cost = prev_time_nodes[prev_state_idx].cost + cost;

                    if next_time_nodes[next_state_idx].cost > total_cost {
                        if next_time_nodes[next_state_idx].cost.is_infinite() {
                            next_reachable_states.push(next_state_idx as u32);
                        }

                        next_time_nodes[next_state_idx].cost = total_cost;
                        next_time_nodes[next_state_idx].prev_state =
                            state_to_idx(&prev_state, problem) as i32;
                    }
                },
            );
        }

        std::mem::swap(&mut reachable_states, &mut next_reachable_states);
        next_reachable_states.clear();
    }

    println!(
        "STATS states {} processed {} edges {} avg.edge {}",
        n_states_total,
        n_states_processed,
        n_edges,
        n_edges as f32 / n_states_processed as f32
    );

    // Find the best final state and trace back to create plan

    let last_time_nodes = &all_nodes[(problem.planning_end_time - first_time) * n_states_per_time
        ..(problem.planning_end_time + 1 - first_time) * n_states_per_time];

    let best_state_idx = last_time_nodes
        .iter()
        .enumerate()
        .min_by_key(|(_i, x)| OrderedFloat(x.cost))
        .unwrap()
        .0;

    let mut current_node = &last_time_nodes[best_state_idx];
    let objective = current_node.cost;
    let mut states: Vec<(usize, usize)> = vec![(problem.planning_end_time, best_state_idx)];
    let mut t = problem.planning_end_time;

    while current_node.prev_state >= 0 {
        t -= 1;
        states.push((t, current_node.prev_state as usize));

        let time_nodes = &all_nodes
            [(t - first_time) * n_states_per_time..(t + 1 - first_time) * n_states_per_time];

        current_node = &time_nodes[current_node.prev_state as usize];
    }

    assert!(t == first_time);
    states.reverse();

    println!(
        "states (horison: {}-{}): {:?}",
        problem.planning_start_time, problem.planning_end_time, states
    );

    let mut output = Vec::new();
    for (t, state_idx) in states {
        let state = idx_to_state(state_idx, problem);
        let biomass = if state.tanks_in_use == 0 {
            0.0
        } else {
            calc_biomass(problem, t, &state)
        };

        let (lb, ub) = if state.tanks_in_use == 0 {
            (0., 0.)
        } else {
            state_biomass_limits(problem, t, state.deploy_age, state.tanks_in_use)
        };
        println!(
            "t {} tanks={} cleaning={} age={} biomass={}-- {}  [{},{}]",
            t,
            state.tanks_in_use,
            state.tanks_being_cleaned,
            state.deploy_age,
            state.biomass,
            biomass,
            lb,
            ub
        );
        // println!("    costs: {:?}", cost_descr);

        output.push(SolutionState {
            deploy_period: if state.tanks_in_use > 0 {
                (t - state.deploy_age) as i32
            } else {
                -1
            },
            period: t as u32,
            biomass,
            num_tanks: state.tanks_in_use as u32,
            num_tanks_cleaning: state.tanks_being_cleaned as u32,
        });
    }

    println!("Best solution has cost {}", objective);
    Solution {
        states: output,
        objective,
    }
}
