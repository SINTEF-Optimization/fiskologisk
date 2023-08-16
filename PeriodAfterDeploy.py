class PeriodAfterDeploy:
    """
    Collection of data for a period P1 for salmon deployed in period P0

    Attributes:
        feed_cost                   Feed cost in P1 for salmon deployed in P0 (NOK/kg)
        oxygen_cost                 Oxygen cost in P1 for salmon deployed in P0 (NOK/kg)
        growth_factor               Growth factor for salmon in nP1 originally deployed in P0
        transferred_growth_factor   Growth factor for salmon transferred in P1 originally deployed in P0
        weight_fraction             Fraction of weigth falling into each weight class for salmon in P1 originally deplyed in P0. Key is weight of weight class
    """
    feed_cost : float
    oxygen_cost : float
    growth_factor : float
    transferred_growth_factor : float
    weight_fraction : dict[float, float]
