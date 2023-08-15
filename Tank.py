from dataclasses import dataclass
from Module import Module
from Tank import Tank

@dataclass
class Tank:
    """
    A water tank that might contain salmon.

    Attributes:
        module              The module this tank belongs to
        transferable_to     The tanks salmon in this tank might be transered to
        transferable_from   The tanks that might transfer salmon to this tank
    """
    module : Module
    transferable_to : list[Tank]
    transferable_from : list[Tank]

