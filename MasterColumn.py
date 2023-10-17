class MasterColumn:
    """A column for the Master Problem from a solution in one of the column generation subproblems"""

    module_index: int
    """The index of the module in the subproblem"""

    objective_value: float
    """The objective value in the subproblem of the solution the column is built for"""

    extract_weight_values: dict[(int, int, int), float]
    """The values of the continous MIP variables for weight of salmon extracted from the tanks for post-smolt or harvesting. Key is deploy period, tank and extract period"""

    population_weight_values: dict[(int, int, int), float]
    """The values of the continous MIP variables for salmon weight in a tank at a period. Key is deploy period, tank and period the mass is given for"""

    transfer_weight_values: dict[(int, int, int, int), float]
    """The values of the continous MIP variables for weight of salmon transfered at a period. Key is deploy period, tank transferred from, tank transferred to and period when the transfer takes place"""

    contains_salmon_values: dict[(int, int), int]
    """The values of the binary MIP variables for whether tanks hold salmon at a given period. Key is tank and period"""

    smolt_deployed_values: dict[(int, int), int]
    """The values of the binary MIP variables for whether salmon has been deployed in a module at a given period. Key is module and deploy period within the planning horizon"""

    salmon_extracted_values: dict[(int, int), int]
    """The values of the binary MIP variables for whether salmon was extracted from a tank at the end of a given period. Key is tank and period"""

    salmon_transferred_values: dict[(int, int), int]
    """The values of the binary MIP variables for whether salmon was transferred to a tank at a given period. Key is tank transferred to and period"""

    def __init__(self, module_index: int, objective_value: float,
                 extract_weight_values: dict[(int, int, int), float], population_weight_values: dict[(int, int, int), float], transfer_weight_values: dict[(int, int, int, int), float],
                 contains_salmon_values: dict[(int, int), int], smolt_deployed_values: dict[(int, int), int], salmon_extracted_values: dict[(int, int), int], salmon_transferred_values: dict[(int, int), int]) -> None:

        self.module_index = module_index
        self.objective_value = objective_value
        self.extract_weight_values = extract_weight_values
        self.population_weight_values = population_weight_values
        self.transfer_weight_values = transfer_weight_values
        self.contains_salmon_values = contains_salmon_values
        self.smolt_deployed_values = smolt_deployed_values
        self.salmon_extracted_values = salmon_extracted_values
        self.salmon_transferred_values = salmon_transferred_values

    def deploy_periods(self) -> list[int]:
        """Returns the periods when salmon is deployed in the solution of this column.

        returns:
            A sorted list of the indices of the deploy periods
        """

        depl_periods = [dep_p for (_, dep_p), val in self.smolt_deployed_values.items() if val > 0.5]
        depl_periods.sort()
        return depl_periods

    def drop_positive_solution(self) -> None:
        """Prints the objective and the variable values of the column. Only positive values are printed."""
    
        print("objective = %s"%self.objective_value)
        for key, value in self.extract_weight_values.items():
            if value > 0.5:
                print("extract_weight(%s) = %s"%(key, value))
        for key, value in self.population_weight_values.items():
            if value > 0.5:
                print("population_weight(%s) = %s"%(key, value))
        for key, value in self.transfer_weight_values.items():
            if value > 0.5:
                print("transfer_weight(%s) = %s"%(key, value))
        for key, value in self.contains_salmon_values.items():
            if value > 0.001:
                print("contains_salmon(%s) = %s"%(key, value))
        for key, value in self.smolt_deployed_values.items():
            if value > 0.001:
                print("smolt_deployed(%s) = %s"%(key, value))
        for key, value in self.salmon_extracted_values.items():
            if value > 0.001:
                print("salmon_extracted(%s) = %s"%(key, value))
        for key, value in self.salmon_transferred_values.items():
            if value > 0.001:
                print("salmon_transferred(%s) = %s"%(key, value))
