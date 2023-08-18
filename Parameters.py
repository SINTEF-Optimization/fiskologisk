class Parameters:
    """
    Set of global parameters for the MIP problem
    """

    smolt_price : float
    """Purchase price of smolt (NOK/kg)"""

    min_tank_cost : float
    """The minimum monthly cost for operating one tanke (NOK)"""

    marginal_increase_pumping_cost : float
    """Marginal increase in monthly pumping costs for one tank (NOK)"""

    max_deploy_smolt : float
    """Maximum weight of smolt deployed in module (kg)"""

    min_deploy_smolt : float
    """Minimum weight of smolt deployed in module (kg)"""

    max_extract_weight : float
    """Maximum weight of salmon extracted from a tank (kg)"""

    max_transfer_weight : float
    """Maximum weight of salmon transferred from a tank (kg)"""

    min_transfer_weight : float
    """Minimum weight of salmon transferred from a tank (kg)"""

    max_tank_density : float
    """Max legal tank density (kg/m3)"""

    max_total_biomass : float
    """Max legal total biomass (kg)"""

    max_yearly_production : float
    """Max legal total production in a year (kg)"""

    monthly_loss : float
    """Monthly production loss as portion of biomass"""

    harvest_yield : float
    """Harvest yield as portion of harvested salmon"""
