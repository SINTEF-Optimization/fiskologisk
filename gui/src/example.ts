import { SalmonPlan } from "./model"

export const example1: SalmonPlan = {
    global_params: {
        maximum_total_biomass: 100,
        maximum_yearly_production: 100,
        smolt_price_nok_per_kg: 1,
        maximum_extraction: 100,

        minimum_smolt_deploy: 0,
        maximum_smolt_deploy: 100,
        minimum_tank_transfer: 0,
        maximum_tank_transfer: 100,
    },
    weight_classes: [
        {
            individual_weight_lb: 0.0,
            individual_weight_ub: 1.0,
            revenue_harvest_per_kg: 1.0,
            revenue_postsmolt_per_kg: 1.0,
        },
        {
            individual_weight_lb: 1.0,
            individual_weight_ub: 2.0,
            revenue_harvest_per_kg: 1.0,
            revenue_postsmolt_per_kg: 1.0,
        },
        {
            individual_weight_lb: 2.0,
            individual_weight_ub: 3.0,
            revenue_harvest_per_kg: 1.0,
            revenue_postsmolt_per_kg: 1.0,
        },
    ],
    periods: [
        {
            start_date: new Date("2021-03-01"),
            end_date: new Date("2021-04-01"),
        },
        {
            start_date: new Date("2021-04-01"),
            end_date: new Date("2021-05-01"),
        },
        {
            start_date: new Date("2021-05-01"),
            end_date: new Date("2021-06-01"),
        },
        {
            start_date: new Date("2021-06-01"),
            end_date: new Date("2021-07-01"),
        },
        {
            start_date: new Date("2021-07-01"),
            end_date: new Date("2021-08-01"),
        },
        {
            start_date: new Date("2021-08-01"),
            end_date: new Date("2021-09-01"),
        },
        {
            start_date: new Date("2021-09-01"),
            end_date: new Date("2021-10-01"),
        },
        {
            start_date: new Date("2021-10-01"),
            end_date: new Date("2021-11-01"),
        }
    ],
    modules: [
        {
            tanks: [
                {
                    maximum_density: 1.0,
                    volume: 1.0,
                    periods: [
                        {
                            period: 0,
                            deployed_period: 0,

                            salmon_deployed: 1.0,
                            salmon_weight: 1.0,
                            salmon_extracted: 0.0,
                            salmon_harvest_yield: 0.0,

                            salmon_transfer_out: 0.0,
                            salmon_transferred_to_this_tank: [],
                            salmon_classes: [{ class: 0, weight: 1.0 }],

                            feed_cost: 1.0,
                            oxygen_cost: 1.0,
                            constant_operation_cost: 1.0,
                        },
                        {
                            period: 1,
                            deployed_period: 0,

                            salmon_deployed: 0.0,
                            salmon_weight: 2.0,
                            salmon_extracted: 0.0,
                            salmon_harvest_yield: 0.0,
                            salmon_transfer_out: 0.0,

                            salmon_transferred_to_this_tank: [],
                            salmon_classes: [{ class: 1, weight: 2.0 }],

                            feed_cost: 1.0,
                            oxygen_cost: 1.0,
                            constant_operation_cost: 1.0,
                        },
                        {
                            period: 2,
                            deployed_period: 0,

                            salmon_deployed: 0.0,
                            salmon_weight: 3.0,
                            salmon_extracted: 3.0,
                            salmon_harvest_yield: 0.0,
                            salmon_transfer_out: 0.0,

                            salmon_transferred_to_this_tank: [],
                            salmon_classes: [{ class: 1, weight: 3.0 }],

                            feed_cost: 1.0,
                            oxygen_cost: 1.0,
                            constant_operation_cost: 1.0,
                        },
                        {
                            period: 3,
                            deployed_period: 0,

                            salmon_deployed: 0.0,
                            salmon_weight: 0.0,
                            salmon_extracted: 0.0,
                            salmon_harvest_yield: 0.0,
                            salmon_transfer_out: 0.0,

                            salmon_transferred_to_this_tank: [],
                            salmon_classes: [],

                            feed_cost: 0.0,
                            oxygen_cost: 0.0,
                            constant_operation_cost: 0.0,
                        },
                        {
                            period: 4,
                            deployed_period: 0,

                            salmon_deployed: 0.0,
                            salmon_weight: 5.0,
                            salmon_extracted: 0.0,
                            salmon_harvest_yield: 0.0,
                            salmon_transfer_out: 0.0,

                            salmon_transferred_to_this_tank: [{ from_tank: { module_idx: 0, tank_idx: 3 }, weight: 5 }],
                            salmon_classes: [{ class: 2, weight: 5 }],

                            feed_cost: 1.0,
                            oxygen_cost: 1.0,
                            constant_operation_cost: 1.0,
                        },
                        {
                            period: 5,
                            deployed_period: 0,

                            salmon_deployed: 0.0,
                            salmon_weight: 6.0,
                            salmon_extracted: 0.0,
                            salmon_harvest_yield: 0.0,
                            salmon_transfer_out: 0.0,

                            salmon_transferred_to_this_tank: [],
                            salmon_classes: [{ class: 2, weight: 6 }],

                            feed_cost: 1.0,
                            oxygen_cost: 1.0,
                            constant_operation_cost: 1.0,
                        },
                        {
                            period: 6,
                            deployed_period: 0,

                            salmon_deployed: 0.0,
                            salmon_weight: 7.0,
                            salmon_extracted: 0.0,
                            salmon_harvest_yield: 0.0,
                            salmon_transfer_out: 0.0,

                            salmon_transferred_to_this_tank: [],
                            salmon_classes: [{ class: 2, weight: 7 }],

                            feed_cost: 1.0,
                            oxygen_cost: 1.0,
                            constant_operation_cost: 1.0,
                        },
                        {
                            period: 7,
                            deployed_period: 0,

                            salmon_deployed: 0.0,
                            salmon_weight: 8.0,
                            salmon_extracted: 8.0,
                            salmon_harvest_yield: 0.75 * 8.0,
                            salmon_transfer_out: 0.0,

                            salmon_transferred_to_this_tank: [],
                            salmon_classes: [{ class: 2, weight: 8 }],

                            feed_cost: 1.0,
                            oxygen_cost: 1.0,
                            constant_operation_cost: 1.0,
                        },
                    ]
                },
                {
                    maximum_density: 1.0,
                    volume: 1.0,
                    periods: [
                        {
                            period: 0,
                            deployed_period: 0,

                            salmon_deployed: 1.0,
                            salmon_weight: 1.0,
                            salmon_extracted: 0.0,
                            salmon_harvest_yield: 0.0,
                            salmon_transfer_out: 0.0,

                            salmon_transferred_to_this_tank: [],
                            salmon_classes: [{ class: 0, weight: 1.0 }],

                            feed_cost: 1.0,
                            oxygen_cost: 1.0,
                            constant_operation_cost: 1.0,
                        },
                        {
                            period: 1,
                            deployed_period: 0,

                            salmon_deployed: 0.0,
                            salmon_weight: 2.0,
                            salmon_extracted: 0.0,
                            salmon_harvest_yield: 0.0,
                            salmon_transfer_out: 0.0,

                            salmon_transferred_to_this_tank: [],
                            salmon_classes: [{ class: 1, weight: 2.0 }],

                            feed_cost: 1.0,
                            oxygen_cost: 1.0,
                            constant_operation_cost: 1.0,
                        },
                        {
                            period: 2,
                            deployed_period: 0,

                            salmon_deployed: 0.0,
                            salmon_weight: 3.0,
                            salmon_extracted: 3.0,
                            salmon_harvest_yield: 0.0,
                            salmon_transfer_out: 0.0,

                            salmon_transferred_to_this_tank: [],
                            salmon_classes: [{ class: 1, weight: 3.0 }],

                            feed_cost: 1.0,
                            oxygen_cost: 1.0,
                            constant_operation_cost: 1.0,
                        },
                        {
                            period: 3,
                            deployed_period: 0,

                            salmon_deployed: 0.0,
                            salmon_weight: 0.0,
                            salmon_extracted: 0.0,
                            salmon_harvest_yield: 0.0,
                            salmon_transfer_out: 0.0,

                            salmon_transferred_to_this_tank: [],
                            salmon_classes: [],

                            feed_cost: 0.0,
                            oxygen_cost: 0.0,
                            constant_operation_cost: 0.0,
                        },
                        {
                            period: 4,
                            deployed_period: 0,

                            salmon_deployed: 0.0,
                            salmon_weight: 5.0,
                            salmon_extracted: 0.0,
                            salmon_harvest_yield: 0.0,
                            salmon_transfer_out: 0.0,

                            salmon_transferred_to_this_tank: [{ from_tank: { module_idx: 0, tank_idx: 3 }, weight: 5 }],
                            salmon_classes: [{ class: 2, weight: 5 }],

                            feed_cost: 1.0,
                            oxygen_cost: 1.0,
                            constant_operation_cost: 1.0,
                        },
                        {
                            period: 5,
                            deployed_period: 0,

                            salmon_deployed: 0.0,
                            salmon_weight: 6.0,
                            salmon_extracted: 0.0,
                            salmon_harvest_yield: 0.0,
                            salmon_transfer_out: 0.0,

                            salmon_transferred_to_this_tank: [],
                            salmon_classes: [{ class: 2, weight: 6 }],

                            feed_cost: 1.0,
                            oxygen_cost: 1.0,
                            constant_operation_cost: 1.0,
                        },
                        {
                            period: 6,
                            deployed_period: 0,

                            salmon_deployed: 0.0,
                            salmon_weight: 7.0,
                            salmon_extracted: 0.0,
                            salmon_harvest_yield: 0.0,
                            salmon_transfer_out: 0.0,

                            salmon_transferred_to_this_tank: [],
                            salmon_classes: [{ class: 2, weight: 7 }],

                            feed_cost: 1.0,
                            oxygen_cost: 1.0,
                            constant_operation_cost: 1.0,
                        },
                        {
                            period: 7,
                            deployed_period: 0,

                            salmon_deployed: 0.0,
                            salmon_weight: 8.0,
                            salmon_extracted: 8.0,
                            salmon_harvest_yield: 0.75 * 8.0,
                            salmon_transfer_out: 0.0,

                            salmon_transferred_to_this_tank: [],
                            salmon_classes: [{ class: 2, weight: 8 }],

                            feed_cost: 1.0,
                            oxygen_cost: 1.0,
                            constant_operation_cost: 1.0,
                        },
                    ]
                },
                {
                    maximum_density: 1.0,
                    volume: 1.0,
                    periods: [
                        {
                            period: 0,
                            deployed_period: 0,

                            salmon_deployed: 1.0,
                            salmon_weight: 1.0,
                            salmon_extracted: 0.0,
                            salmon_harvest_yield: 0.0,
                            salmon_transfer_out: 0.0,

                            salmon_transferred_to_this_tank: [],
                            salmon_classes: [{ class: 0, weight: 1.0 }],

                            feed_cost: 1.0,
                            oxygen_cost: 1.0,
                            constant_operation_cost: 1.0,
                        },
                        {
                            period: 1,
                            deployed_period: 0,

                            salmon_deployed: 0.0,
                            salmon_weight: 2.0,
                            salmon_extracted: 0.0,
                            salmon_harvest_yield: 0.0,
                            salmon_transfer_out: 0.0,

                            salmon_transferred_to_this_tank: [],
                            salmon_classes: [{ class: 1, weight: 2.0 }],

                            feed_cost: 1.0,
                            oxygen_cost: 1.0,
                            constant_operation_cost: 1.0,
                        },
                        {
                            period: 2,
                            deployed_period: 0,

                            salmon_deployed: 0.0,
                            salmon_weight: 3.0,
                            salmon_extracted: 0.0,
                            salmon_harvest_yield: 0.0,
                            salmon_transfer_out: 0.0,

                            salmon_transferred_to_this_tank: [],
                            salmon_classes: [{ class: 1, weight: 3.0 }],

                            feed_cost: 1.0,
                            oxygen_cost: 1.0,
                            constant_operation_cost: 1.0,
                        },
                        {
                            period: 3,
                            deployed_period: 0,

                            salmon_deployed: 0.0,
                            salmon_weight: 4.0,
                            salmon_extracted: 0.0,
                            salmon_harvest_yield: 0.0,
                            salmon_transfer_out: 0.0,

                            salmon_transferred_to_this_tank: [],
                            salmon_classes: [{ class: 1, weight: 4.0 }],

                            feed_cost: 1.0,
                            oxygen_cost: 1.0,
                            constant_operation_cost: 1.0,
                        },
                        {
                            period: 4,
                            deployed_period: 0,

                            salmon_deployed: 0.0,
                            salmon_weight: 5.0,
                            salmon_extracted: 0.0,
                            salmon_harvest_yield: 0.0,
                            salmon_transfer_out: 5.0,

                            salmon_transferred_to_this_tank: [],
                            salmon_classes: [{ class: 2, weight: 5.0 }],

                            feed_cost: 1.0,
                            oxygen_cost: 1.0,
                            constant_operation_cost: 1.0,
                        },
                        {
                            period: 5,
                            deployed_period: 0,

                            salmon_deployed: 0.0,
                            salmon_weight: 6.0,
                            salmon_extracted: 0.0,
                            salmon_harvest_yield: 0.0,
                            salmon_transfer_out: 0.0,

                            salmon_transferred_to_this_tank: [],
                            salmon_classes: [{ class: 2, weight: 6 }],

                            feed_cost: 1.0,
                            oxygen_cost: 1.0,
                            constant_operation_cost: 1.0,
                        },
                        {
                            period: 6,
                            deployed_period: 0,

                            salmon_deployed: 0.0,
                            salmon_weight: 7.0,
                            salmon_extracted: 0.0,
                            salmon_harvest_yield: 0.0,
                            salmon_transfer_out: 0.0,

                            salmon_transferred_to_this_tank: [],
                            salmon_classes: [{ class: 2, weight: 7 }],

                            feed_cost: 1.0,
                            oxygen_cost: 1.0,
                            constant_operation_cost: 1.0,
                        },
                        {
                            period: 7,
                            deployed_period: 0,

                            salmon_deployed: 0.0,
                            salmon_weight: 8.0,
                            salmon_extracted: 8.0,
                            salmon_harvest_yield: 0.75 * 8.0,
                            salmon_transfer_out: 0.0,

                            salmon_transferred_to_this_tank: [],
                            salmon_classes: [{ class: 2, weight: 8 }],

                            feed_cost: 1.0,
                            oxygen_cost: 1.0,
                            constant_operation_cost: 1.0,
                        },
                    ]
                },
                {
                    maximum_density: 1.0,
                    volume: 1.0,
                    periods: [
                        {
                            period: 0,
                            deployed_period: 0,

                            salmon_deployed: 1.0,
                            salmon_weight: 1.0,
                            salmon_extracted: 0.0,
                            salmon_harvest_yield: 0.0,
                            salmon_transfer_out: 0.0,

                            salmon_transferred_to_this_tank: [],
                            salmon_classes: [{ class: 0, weight: 1.0 }],

                            feed_cost: 1.0,
                            oxygen_cost: 1.0,
                            constant_operation_cost: 1.0,
                        },
                        {
                            period: 1,
                            deployed_period: 0,

                            salmon_deployed: 0.0,
                            salmon_weight: 2.0,
                            salmon_extracted: 0.0,
                            salmon_harvest_yield: 0.0,
                            salmon_transfer_out: 0.0,

                            salmon_transferred_to_this_tank: [],
                            salmon_classes: [{ class: 1, weight: 2.0 }],

                            feed_cost: 1.0,
                            oxygen_cost: 1.0,
                            constant_operation_cost: 1.0,
                        },
                        {
                            period: 2,
                            deployed_period: 0,

                            salmon_deployed: 0.0,
                            salmon_weight: 3.0,
                            salmon_extracted: 0.0,
                            salmon_harvest_yield: 0.0,
                            salmon_transfer_out: 0.0,

                            salmon_transferred_to_this_tank: [],
                            salmon_classes: [{ class: 1, weight: 3.0 }],

                            feed_cost: 1.0,
                            oxygen_cost: 1.0,
                            constant_operation_cost: 1.0,
                        },
                        {
                            period: 3,
                            deployed_period: 0,

                            salmon_deployed: 0.0,
                            salmon_weight: 4.0,
                            salmon_extracted: 0.0,
                            salmon_harvest_yield: 0.0,
                            salmon_transfer_out: 0.0,

                            salmon_transferred_to_this_tank: [],
                            salmon_classes: [{ class: 1, weight: 4.0 }],

                            feed_cost: 1.0,
                            oxygen_cost: 1.0,
                            constant_operation_cost: 1.0,
                        },
                        {
                            period: 4,
                            deployed_period: 0,

                            salmon_deployed: 0.0,
                            salmon_weight: 5.0,
                            salmon_extracted: 0.0,
                            salmon_harvest_yield: 0.0,
                            salmon_transfer_out: 5.0,

                            salmon_transferred_to_this_tank: [],
                            salmon_classes: [{ class: 2, weight: 5.0 }],

                            feed_cost: 1.0,
                            oxygen_cost: 1.0,
                            constant_operation_cost: 1.0,
                        },
                        {
                            period: 5,
                            deployed_period: 0,

                            salmon_deployed: 0.0,
                            salmon_weight: 6.0,
                            salmon_extracted: 0.0,
                            salmon_harvest_yield: 0.0,
                            salmon_transfer_out: 0.0,

                            salmon_transferred_to_this_tank: [],
                            salmon_classes: [{ class: 2, weight: 6 }],

                            feed_cost: 1.0,
                            oxygen_cost: 1.0,
                            constant_operation_cost: 1.0,
                        },
                        {
                            period: 6,
                            deployed_period: 0,

                            salmon_deployed: 0.0,
                            salmon_weight: 7.0,
                            salmon_extracted: 0.0,
                            salmon_harvest_yield: 0.0,
                            salmon_transfer_out: 0.0,

                            salmon_transferred_to_this_tank: [],
                            salmon_classes: [{ class: 2, weight: 7 }],

                            feed_cost: 1.0,
                            oxygen_cost: 1.0,
                            constant_operation_cost: 1.0,
                        },
                        {
                            period: 7,
                            deployed_period: 0,

                            salmon_deployed: 0.0,
                            salmon_weight: 8.0,
                            salmon_extracted: 8.0,
                            salmon_harvest_yield: 0.75 * 8.0,
                            salmon_transfer_out: 0.0,

                            salmon_transferred_to_this_tank: [],
                            salmon_classes: [{ class: 2, weight: 8 }],

                            feed_cost: 1.0,
                            oxygen_cost: 1.0,
                            constant_operation_cost: 1.0,
                        },
                    ]
                },
            ]
        }
    ]
};