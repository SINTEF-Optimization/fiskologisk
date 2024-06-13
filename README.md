# Fiskologisk

This repository contains code to solve a production scheduling optimization
problem. The problem is to determine long-term production plans for land-based
salmon farms. In land-based salmon farming, salmon can either be grown to
harvest size, or sold as post-smolt to be deployed to conventional sea farms.
Land-based farms are organized in modules, each containing a number of tanks.
Salmon can be transferred between tanks in the same module, but are not
transferred between modules. With multiple modules and tanks, and and
time-dependent growth and prices, the planning problem becomes very complex to
solve optimally.

The optimization problem and the mixed-integer programming (MIP) base model is
described in more detail in Johan Moldeklev Føsund and Erlend Hjelle
Strandkleiv's Master's thesis
[*Using Branch and Price to Optimize Land-Based Salmon Production*](https://ntnuopen.ntnu.no/ntnu-xmlui/handle/11250/3108221). 
This repository contains an implementation of their model and several
decomposition schemes -- the most effective of them being the production cycle
decomposition. See the solver descriptions below.

## Requirements and installation

Requirements: 
 
  * Python >=3.10
  * Gurobipy (`pip install gurobipy`)
  * Optional: Rust tooling and maturin (`pip install maturin`) for the DP subproblem solver

## Iterations

In this context, we are solving a long-term planning problem (typically 2-4
years). However, the initial state of the facility, meaning the age and weight
of the fish currently in the tanks, can still significantly influence the
decisions made in the production plan.

Since we are solving hypothetical case studies, we do not have data for the
current state of the tanks. We use a planning horizon that starts in the month
of March, and it is not reasonable to assume that the tanks are empty at that
time. Instead, we solve the planning problem in multiple iterations, starting
with iteration 0 with all empty tanks. Then, in each successive iteration we use
the ending state of the previous iteration's month of March in the final year to
use as the next iteration's state in the month of March in the initial year.
Three such initial iterations seemed to produce a reasonable starting state.

A complete run of the planning iterations would look like:

```
python run_iteration.py 
python run_iteration.py 
python run_iteration.py 
python run_iteration.py 
```

In performance evaluations, we consider only the last command invocation to
represent the solving of the actual planning problem that would be used in a
productive environment, and the three initial runs are only there to provide
a reasonable initial state for the facility.

## Algorithms

The program can be configured to use different solver implementations by giving
the command line parameters `--Decomposition 0/1/2` and 
`--Use_heuristic true/false`. Each configuration is described briefly below.

### Full MIP

The complete MIP model for all decisions, based on the model described in 
[*Using Branch and Price to Optimize Land-Based Salmon Production*](https://ntnuopen.ntnu.no/ntnu-xmlui/handle/11250/3108221).

As it is the default setting, it is executed by running: 

```
python run_iteration.py 
```

... or, explicitly:

```
python run_iteration.py --Decomposition 0
```

### Single-module full-horizon MIP decomposition

Also described in the same [Master's thesis](https://ntnuopen.ntnu.no/ntnu-xmlui/handle/11250/3108221),
this algorithm decomposes decisions about each module into separate subproblems.
The subproblems are solved using the same MIP formulation as in the full MIP,
but the solutions of the different modules are combined in a master MIP problem 
using column generation. Executed by running:

```
python run_iteration.py --Decomposition 1
```

### Single-module full-horizon decomposition with custom DP subproblem solver

This algorithm uses the same decomposition as the previous one, but instead of
using the MIP formulation for the single-module production planning problem, we
use a dynamic programming algorithm over a state space that is discretized to a
fixed set of biomass levels.

This subproblem solver is implemented in Rust for performance reasons, so to
use this algorithm you will need to install the 
[Rust toolchain](https://rustup.rs/), maturin (`pip install maturin`), and then
install the DP algorithm as a module into your Python environment:

```
cd fiskologisk/solvers/dp_heur
maturin develop --release
```

Then, the algorithm is executed by running:

```
python run_iteration.py --Decomposition 1 --Use_heuristic true
```

### Production-cycle decomposition

This algorithm uses a decomposition where decisions pertaning to a single
production cycle is treated as a subproblem, and production-cycle solutions
are packed into the overall production plan by a set packing MIP formulation.

Executed by running:

```
python run_iteration.py --Decomposition 2
```


# Contributors

The code in this repository was mainly written by Kjell Fredrik Pettersen,
Giorgio Sartor, and Bjørnar Luteberget in 2023. The development was funded by
*SINTEF's 2023 strategisk egenfinansiert prosjekt Fiskologisk*. This work is
also heavily based on the modeling and case study performed by Johan Moldeklev
Føsund and Erlend Hjelle Strandkleiv in their 
[Master's thesis](https://ntnuopen.ntnu.no/ntnu-xmlui/handle/11250/3108221).