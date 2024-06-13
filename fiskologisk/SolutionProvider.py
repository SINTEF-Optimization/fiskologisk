from fiskologisk.domain.Period import Period
from fiskologisk.domain.Tank import Tank
from fiskologisk.domain.Module import Module

class SolutionProvider:
    """Base class for a provider of the values of the variables in a solution to the MIP problem for landbased salmon farming if a solution is found."""

    def extract_weight_value(self, depl_period: Period, tank: Tank, period: Period) -> float:
        """Returns the value of the continous MIP variable for weight of extracted salmon

        args:
            - depl_period: 'Period' The period when the extracted salmon was deployed
            - tank: 'Tank' The tank the salmon was extracted from
            - period: 'Period' The period when the salmon was extracted
        """

        pass

    def population_weight_value(self, depl_period: Period, tank: Tank, period: Period) -> float:
        """Returns the value of the continous MIP variable for salmon weight in a tank

        args:
            - depl_period: 'Period' The period when the salmon was deployed
            - tank: 'Tank' The tank
            - period: 'Period' The period to get the salmon weight for
        """

        pass

    def transfer_weight_value(self, depl_period: Period, from_tank: Tank, to_tank: Tank, period: Period) -> float:
        """Returns the value of the continous MIP variable for weight of transferred salmon

        args:
            - depl_period: 'Period' The period when the transferred salmon was deployed
            - from_tank: 'Tank' The tank the salmon was transferred from
            - to_tank: 'Tank' The tank the salmon was transferred to
            - period: 'Period' The period when the salmon was transferred
        """

        pass

    def contains_salmon_value(self, tank: Tank, period: Period) -> float:
        """Returns the value of the binary MIP variable for whether a tanks holds salmon at a given period

        args:
            - tank: 'Tank' The tank
            - period: 'Period' The period
        """

        pass

    def smolt_deployed_value(self, module: Module, depl_period: Period) -> float:
        """Returns the value of the binary MIP variable for whether salmon has been deployed in a module at a given period

        args:
            - module: 'Module' The module
            - depl_period: 'Period' The deploy period
        """

        pass

    def salmon_extracted_value(self, tank: Tank, period: Period) -> float:
        """Returns the value of the binary MIP variable for whether salmon was extracted from a tank at the end of a given period

        args:
            - tank: 'Tank' The tank
            - period: 'Period' The period
        """

        pass

    def salmon_transferred_value(self, tank: Tank, period: Period) -> float:
        """Returns the value of the binary MIP variable for whether salmon was transferred to a tank at a given period

        args:
            - tank: 'Tank' The tank
            - period: 'Period' The period
        """

        pass

    def drop_positive_solution(self) -> None:

        pass