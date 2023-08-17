from Module import Module
from Period import Period
from Year import Year
from Parameters import Parameters

class Environment:
    """
    The top class of the model used for building the MIP problem for landbased salmon farming

    Attributes:
        modules                   The modules (units of tanks) in the model
        weight_classes            The number of weight classes in the model
        periods                   The periods in the planning horizon
        release_periods           The periods in the planning horizon when salmon can be deployed/released
        preplan_release_periods   The periods prior to the planning horizon when salmon was deployed/released
        years                     The years in the planning horizon
        parameters                The global parameters in the MIP problem
    """
    modules : list[Module]
    weight_classes : int
    periods : list[Period]
    release_periods : list[Period]
    preplan_release_periods : list[Period]
    years : list[Year]
    parameters : Parameters

    def __init__(self) -> None:
        self.modules = []
        self.weight_classes = 0
        self.periods = []
        self.release_periods = []
        self.preplan_release_periods = []
        self.years = []
        self.parameters = Parameters()