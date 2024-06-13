from typing import Dict
from fiskologisk.domain.Module import Module
from fiskologisk.domain.Tank import Tank
from fiskologisk.domain.Period import Period
from fiskologisk.domain.Year import Year
from fiskologisk.domain.Parameters import Parameters
from fiskologisk.domain.WeightClass import WeightClass

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

    period_indices : Dict[int, Period]
    """A map from the period index to the period object."""

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
        self.period_indices = {}

    def add_initial_populations(self, initial_populations) -> None:

        for init_pop in initial_populations:
            tank_index = init_pop["tank"]
            tank = next(t for t in self.tanks if t.index == tank_index)

            if "deploy_period" in init_pop:
                tank.initial_deploy_period = init_pop["deploy_period"]

            if "weight" in init_pop:
                tank.initial_weight = init_pop["weight"]

            if "in_use" in init_pop:
                tank.initial_use = init_pop["in_use"]

    def get_tanks(self, module_idx: int) -> list[Tank]:
        """Gets either the tanks of a specific module or all tanks in the model

        args:
            - module_idx: 'int' The index of the specific module to get the tanks for, or -1 to get all the tanks in the model

        returns:
            A list with the requested tanks
        """

        return self.tanks if module_idx == -1 else self.modules[module_idx].tanks

    def get_modules(self, module_idx: int) -> list[Module]:
        """Gets either a specific module or all modules in the model

        args:
            - module_idx: 'int' The index of the specific requested module, or -1 to get all the modules in the model

        returns:
            If module_idx is -1, the list of all modules. If module_idx >= 0, a singleton list with the requested module.
        """

        return self.modules if module_idx == -1 else [self.modules[module_idx]]
