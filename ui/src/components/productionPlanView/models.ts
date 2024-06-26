export type SalmonPlanSolution = {
    modules: Module[],
    pre_planning_horizon: PrePlanningHorizon,
    planning_horizon: PlanningHorizon,
    production_cycles: ProductionCycle[],
}

export type Module = {
    module_index: number,
    tank_indices: number[],
    tank_transfers: { from: number, to: number }[]
}

export type PrePlanningHorizon = {
    first_year: number,
    first_period: number,
    last_period: number
    deploy_periods: number[],
}

export type PlanningHorizon = {
    first_year: number,
    first_period: number,
    last_ordinary_horizon_period: number,
    last_period: number
    deploy_periods: number[],
}

export type ProductionCycle = {
    deploy_period: number,
    module: number,
    tank_cycles: TankCycle[],
}

export type TankCycle = {
    tank: number,
    start_period: number,
    start_cause: StartCause,
    transfer?: { period: number, from_tank: number, biomass: number },
    end_period: number,
    end_cause: EndCause,
    period_biomasses: { period: number, biomass: number }[],

}

export enum StartCause {
    PrePlanningDeploy = "pre_planning_deploy",
    Deploy = "deploy",
    Transfer = "transfer",
}
export enum EndCause {
    PostSmolt = "post_smolt",
    Harvest = "harvest",
    PlanningHorizonExtension = "planning_horizon_extension",
}