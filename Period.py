from __future__ import annotations
from PeriodAfterDeploy import PeriodAfterDeploy

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

    periods_after_deploy : dict[int, PeriodAfterDeploy]
    """Information related to specific periods for salmon deployed this period. Key is period index"""

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
        self.periods_after_deploy = {}
        self.initial_weights = {}

    def add_after_deploy(self, period: Period, period_after_deploy: PeriodAfterDeploy, can_extract: bool) -> None:
        """Connects this period to a period that might hold salmon deployed in this period

        args:
            - period: 'Period' The period that might hold salmon deployed in this period
            - period_after_deploy: 'PeriodAfterDeploy' Collection of data related to the input period for salmon deployed in this period
            - can_extract: 'bool' Whether salmon deployed in this period can be extracted in the input period, either for harvest or as post-smolt
        """

        self.periods_after_deploy[period.index] = period_after_deploy
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
