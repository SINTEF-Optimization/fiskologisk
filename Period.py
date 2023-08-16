from Year import Year
from Period import Period

class Period:
    """
    A period of one month either before or in the planning horizon.
    All months are expected to have 30 days for simplicity.

    Attributes:
        index                        A unique index among all periods, in chronological order, used in MIP variable names
        month                        The month number, in range 0-11
        year                         The year of this month
        deploy_periods               Possible deploy periods for salmon existing in this period
        deploy_periods_for_extract   Possible deploy periods for salmon extracted in this period
        deploy_periods_for_transfer  Possible deploy periods for salmon transferred in this period
        postsmolt_extract_periods    Possible post-smolt extract periods for salmon deployed in this period
        harvest_periods              Possible harvest periods for salmon deployed in this period
        extract_periods              Possible extract periods for salmon deployed in this period
        nonextract_periods           Planning horizon periods when salmon deployed in this period can not be extracted
        transfer_periods             Possible transfer periods for salmon deployed in this period
        initial_weights              Weight of salmon deployed in this period in each tank. Only used if this is a period prior to planning horizon. Part of initial conditions to the MIP problem.
    """
    index : int
    month : int
    year : Year
    deploy_periods : list[Period]
    deploy_periods_for_extract : list[Period]
    deploy_periods_for_transfer : list[Period]
    postsmolt_extract_periods : list[Period]
    harvest_periods : list[Period]
    extract_periods : list[Period]
    nonextract_periods : list[Period]
    transfer_periods : list[Period]
    initial_weights : dict[int, float]