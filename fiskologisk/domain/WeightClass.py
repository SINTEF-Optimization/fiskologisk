class WeightClass:

    weight: float
    """Weight of the weight class (kg)"""

    post_smolt_revenue: float
    """Revenue for post-smolt in the weight class (NOK/kg)"""

    harvest_revenue: float
    """Revenue for harvested salmon in each weight class (NOK/kg)"""

    def __init__(self, weight: float, post_smolt_revenue: float, harvest_revenue: float) -> None:
        self.weight = weight
        self.post_smolt_revenue = post_smolt_revenue
        self.harvest_revenue = harvest_revenue