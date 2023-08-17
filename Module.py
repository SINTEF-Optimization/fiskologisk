from Tank import Tank

class Module:
    """
    A unit of several tanks. Salmon can only be transferred between tanks within the same module.

    Attributes:
        index   Unique index among all modules, used in MIP variable names
        tanks   The tanks in this module
    """
    index : int
    tanks : list[Tank]

    def __init__(self, index: int) -> None:
        self.index = index
        self.tanks = []

    def connect_transfer_tanks(self, transf_from: int, transf_to: int) -> None:
        tank_from = self.tanks[transf_from]
        tank_to = self.tanks[transf_to]
        tank_from.transferable_to.append(tank_to)
        tank_to.transferable_from.append(tank_from)
