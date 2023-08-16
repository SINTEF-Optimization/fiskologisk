from Module import Module
from Tank import Tank

class Tank:
    """
    A water tank that might contain salmon.

    Attributes:
        index               Unique index among all tanks, used in MIP variable names
        module              The module this tank belongs to
        transferable_to     The tanks salmon in this tank might be transered to
        transferable_from   The tanks that might transfer salmon to this tank
    """
    index : int
    module : Module
    transferable_to : list[Tank]
    transferable_from : list[Tank]

