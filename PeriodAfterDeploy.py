class PeriodAfterDeploy:
    """
    Collection of data for a period P1 for salmon deployed in period P0

    Attributes:
        feed_cost                   Feed cost in P1 for salmon deployed in P0 (NOK/kg)
        oxygen_cost                 Oxygen cost in P1 for salmon deployed in P0 (NOK/kg)
        growth_factor               Growth factor for salmon in nP1 originally deployed in P0
        transferred_growth_factor   Growth factor for salmon transferred in P1 originally deployed in P0
        weight_distribution         Fraction of weigth falling into each weight class for salmon in P1 originally deplyed in P0. One entry for each weight class.
    """

    feed_cost : float
    oxygen_cost : float
    growth_factor : float
    transferred_growth_factor : float
    weight_distribution : list[float]

    def __init__(self, feed_cost: float, oxygen_cost: float, growth_factor: float, transferred_growth_factor: float, weight_distribution: list[float]) -> None:
        self.feed_cost = feed_cost
        self.oxygen_cost = oxygen_cost
        self.growth_factor = growth_factor
        self.transferred_growth_factor = transferred_growth_factor
        self.weight_distribution = weight_distribution