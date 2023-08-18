class PeriodAfterDeploy:
    """
    Collection of data for a period P1 for salmon deployed in period P0
    """

    feed_cost : float
    """Feed cost in P1 for salmon deployed in P0 (NOK/kg)"""

    oxygen_cost : float
    """Oxygen cost in P1 for salmon deployed in P0 (NOK/kg)"""

    expected_weight : float
    """Expected weight for salmon in P1 originally deployed in P0"""

    growth_factor : float
    """Growth factor for salmon in P1 originally deployed in P0"""

    transferred_growth_factor : float
    """Growth factor for salmon transferred in P1 originally deployed in P0"""

    weight_distribution : list[float]
    """Fraction of weigth falling into each weight class for salmon in P1 originally deplyed in P0. One entry for each weight class."""

    def __init__(self, feed_cost: float, oxygen_cost: float, growth_factor: float, expected_weight: float, transferred_growth_factor: float, weight_distribution: list[float]) -> None:
        self.feed_cost = feed_cost
        self.oxygen_cost = oxygen_cost
        self.expected_weight = expected_weight
        self.growth_factor = growth_factor
        self.transferred_growth_factor = transferred_growth_factor
        self.weight_distribution = weight_distribution