class Parameters:
    """
    Set of global parameters for the MIP problem

    Attributes:
        smolt_price                      Purchase price of smolt (NOK/kg)
        min_tank_cost                    The minimum monthly cost for operating one tanke (NOK)
        marginal_increase_pumping_cost   Marginal increase in monthly pumping costs for one tank (NOK)
        max_deploy_smolt                 Maximum weight of smolt deployed in module (kg)
        min_deploy_smolt                 Minimum weight of smolt deployed in module (kg)
        max_extract_weight               Maximum weight of salmon extracted from a tank (kg)
        max_transfer_weight              Maximum weight of salmon transferred from a tank (kg)
        min_transfer_weight              Minimum weight of salmon transferred from a tank (kg)
        max_tank_density                 Max legal tank density (kg/m3)
        max_total_biomass                Max legal total biomass (kg)
        max_yearly_production            Max legal total production in a year (kg)
        monthly_loss                     Monthly production loss as portion of biomass
        harvest_yield                    Harvest yield as portion of harvested salmon
        post_smolt_revenue               Revenue for post-smolt in each weight class. Key is weight of weight class
        harvest_revenue                  Revenue for harvested salmon in each weight class. Key is weight of weight class
    """

    smolt_price : float
    min_tank_cost : float
    marginal_increase_pumping_cost : float
    max_deploy_smolt : float
    min_deploy_smolt : float
    max_extract_weight : float
    max_transfer_weight : float
    min_transfer_weight : float
    max_tank_density : float
    max_total_biomass : float
    max_yearly_production : float
    post_smolt_revenue : dict[float, float]
    harvest_revenue : dict[float, float]
    monthly_loss : float
    harvest_yield : float
