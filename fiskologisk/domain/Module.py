from fiskologisk.domain.Tank import Tank

class Module:
    """
    A unit of several tanks. Salmon can only be transferred between tanks within the same module.
    """

    index : int
    """Unique index among all modules"""

    tanks : list[Tank]
    """The tanks in this module"""

    def __init__(self, index: int) -> None:
        self.index = index
        self.tanks = []

    def connect_transfer_tanks(self, transf_from: int, transf_to: int) -> None:
        """Connects two tanks in the module that might transfer salmon
        
        args:
            - transf_from: 'int' The index among the module tanks of the tank salmon is transfered from
            - transf_to: 'int' The index among the module tanks of the tank salmon is transfered TO
        """
        tank_from = self.tanks[transf_from]
        tank_to = self.tanks[transf_to]
        tank_from.transferable_to.append(tank_to)
        tank_to.transferable_from.append(tank_from)
