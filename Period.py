from __future__ import annotations

class Period:
    """
    A period of one month either before or in the planning horizon.
    All months are expected to have 30 days for simplicity.
    """

    index : int
    """A unique index among all periods. The number of months between the start of two periods is the differences between their indices"""

    month : int
    """The month number, in range 0-11"""

    is_deploy : bool
    """Whether this period is a possible deploy period"""

    is_planning : bool
    """Whether this period is in (True) or prior to (False) the planning horizon"""

    deploy_periods : list[Period]
    """Possible deploy periods for salmon existing in this period"""

    deploy_periods_for_extract : list[Period]
    """Possible deploy periods for salmon extracted in this period, either for harvest or as post-smolt"""

    deploy_periods_for_transfer : list[Period]
    """Possible deploy periods for salmon transferred in this period"""

    postsmolt_extract_periods : list[Period]
    """Possible post-smolt extract periods for salmon deployed in this period"""

    harvest_periods : list[Period]
    """Possible harvest periods for salmon deployed in this period"""

    extract_periods : list[Period]
    """Possible extract periods for salmon deployed in this period, either for harvest or as post-smolt"""

    nonextract_periods : list[Period]
    """Periods when salmon deployed in this period might still exist but can not be extracted, neither for harvest nor as post-smolt"""

    transfer_periods : list[Period]
    """Possible transfer periods for salmon deployed in this period"""

    periods_after_deploy : list[PeriodAfterDeploy]
    """Information related to specific periods for salmon deployed this period"""

    initial_weights : dict[int, float]
    """Weight of salmon deployed in this period in each tank. Only used if this is a period prior to planning horizon. Key is tank index. Part of initial conditions to the MIP problem."""

    def __init__(self, index: int, month: int, is_deploy: bool, is_planning: bool) -> None:
        self.index = index
        self.month = month
        self.is_deploy = is_deploy
        self.is_planning = is_planning
        self.deploy_periods = []
        self.deploy_periods_for_extract = []
        self.deploy_periods_for_transfer = []
        self.postsmolt_extract_periods = []
        self.harvest_periods = []
        self.extract_periods = []
        self.nonextract_periods = []
        self.transfer_periods = []
        self.periods_after_deploy = []
        self.initial_weights = {}

    def add_after_deploy(self, period: Period, period_after_deploy: PeriodAfterDeploy, can_extract: bool) -> None:
        """Connects this period to a period that might hold salmon deployed in this period

        args:
            - period: 'Period' The period that might hold salmon deployed in this period
            - period_after_deploy: 'PeriodAfterDeploy' Collection of data related to the input period for salmon deployed in this period
            - can_extract: 'bool' Whether salmon deployed in this period can be extracted in the input period, either for harvest or as post-smolt
        """

        self.periods_after_deploy.append(period_after_deploy)
        period.deploy_periods.append(self)
        if can_extract:
            period.deploy_periods_for_extract.append(self)
            self.extract_periods.append(period)
        else:
            self.nonextract_periods.append(period)

    def add_transfer_period(self, period: Period) -> None:
        """Adds a possible transfer period for salmon deployed in this period

        args:
            - period: 'Period' The period that might transfer salmon deployed in this period
        """

        self.transfer_periods.append(period)
        period.deploy_periods_for_transfer.append(self)

    def add_postsmolt_extract_period(self, period: Period) -> None:
        """Adds a possible post-smolt extract period for salmon deployed in this period

        args:
            - period: 'Period' The period that might extract post-smolt from salmon deployed in this period
        """

        self.postsmolt_extract_periods.append(period)

    def add_harvest_period(self, period: Period) -> None:
        """Adds a possible harvest period for salmon deployed in this period

        args:
            - period: 'Period' The period that might harvest salmon deployed in this period
        """
        self.harvest_periods.append(period)

class PeriodAfterDeploy:
    """
    Collection of data for a period P1 for salmon deployed in period P0
    """

    period : Period
    """The period (P1) for the data"""

    expected_weight : float
    """Expected weight for salmon in P1 originally deployed in P0"""

    feed_cost : float
    """Feed cost in P1 for salmon deployed in P0 (NOK/kg)"""

    oxygen_cost : float
    """Oxygen cost in P1 for salmon deployed in P0 (NOK/kg)"""

    growth_factor : float
    """Growth factor for salmon in P1 originally deployed in P0"""

    transferred_growth_factor : float
    """Growth factor for salmon transferred in P1 originally deployed in P0"""

    weight_distribution : list[float]
    """Fraction of weigth falling into each weight class for salmon in P1 originally deplyed in P0. One entry for each weight class."""

    def __init__(self, period: Period, expected_weight: float, feed_cost: float, oxygen_cost: float, growth_factor: float, transferred_growth_factor: float, weight_distribution: list[float]) -> None:
        self.period = period
        self.expected_weight = expected_weight
        self.feed_cost = feed_cost
        self.oxygen_cost = oxygen_cost
        self.growth_factor = growth_factor
        self.transferred_growth_factor = transferred_growth_factor
        self.weight_distribution = weight_distribution