export type SalmonPlan = {
    global_params: GlobalParams
    weight_classes: WeightClass[]
    periods: Period[]
    modules: Module[]
}

export type WeightClassRef = number;
export type TankRef = { module_idx: number, tank_idx: number };
export type PeriodRef = number;

export type GlobalParams = {
    smolt_price_nok_per_kg: number

    minimum_smolt_deploy: number
    maximum_smolt_deploy: number
    maximum_extraction: number
    minimum_tank_transfer: number
    maximum_tank_transfer: number

    maximum_total_biomass: number
    maximum_yearly_production: number

}

export type Module = {
    tanks: Tank[]
}

export type Tank = {
    volume: number
    maximum_density: number
    periods: TankPeriod[]
}

export type TankPeriod = {
    period: PeriodRef
    deployed_period: PeriodRef

    salmon_deployed: number
    salmon_weight: number
    salmon_extracted: number
    salmon_harvest_yield: number
    salmon_transfer_out: number

    salmon_transferred_to_this_tank: { from_tank: TankRef, weight: number }[]
    salmon_classes: { class: WeightClassRef, weight: number }[]

    feed_cost: number
    oxygen_cost: number
    constant_operation_cost: number
}

export type WeightClass = {
    individual_weight_lb: number
    individual_weight_ub: number
    revenue_postsmolt_per_kg: number
    revenue_harvest_per_kg: number
};

export type Period = {
    start_date :Date
    end_date :Date
};