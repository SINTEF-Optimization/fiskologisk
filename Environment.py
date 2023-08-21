from Module import Module
from Tank import Tank
from Period import Period
from Year import Year
from Parameters import Parameters
from WeightClass import WeightClass

class Environment:
    """
    The top class of the model used for building the MIP problem for landbased salmon farming
    """

    modules : list[Module]
    """The modules (units of tanks) in the model"""

    tanks : list[Tank]
    """The tanks in the model"""

    weight_classes : list[WeightClass]
    """The weight classes in the model"""

    periods : list[Period]
    """The periods in the planning horizon"""

    release_periods : list[Period]
    """The periods when salmon can be deployed/released, both prior to and in planning horizon"""

    plan_release_periods : list[Period]
    """The periods in the planning horizon when salmon can be deployed/released"""

    preplan_release_periods : list[Period]
    """The periods prior to the planning horizon when salmon was deployed/released"""

    years : list[Year]
    """The years in the planning horizon"""

    parameters : Parameters
    """The global parameters in the MIP problem"""

    def __init__(self) -> None:
        self.modules = []
        self.tanks = []
        self.weight_classes = []
        self.periods = []
        self.release_periods = []
        self.plan_release_periods = []
        self.preplan_release_periods = []
        self.years = []
        self.parameters = Parameters()