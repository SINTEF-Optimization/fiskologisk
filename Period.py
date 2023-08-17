from __future__ import annotations
from PeriodAfterDeploy import PeriodAfterDeploy

class Period:
    """
    A period of one month either before or in the planning horizon.
    All months are expected to have 30 days for simplicity.
    """

    index : int
    """A unique index among all periods. The time between two periods is the number of months between their indices"""

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
    """Planning horizon periods when salmon deployed in this period can not be extracted, neither for harvest nor as post-smolt"""

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
