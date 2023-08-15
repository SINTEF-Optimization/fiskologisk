from dataclasses import dataclass
from Tank import Tank

@dataclass
class Module:
    """
    A unit of several tanks. Salmon can only be transferred between tanks within the same module.

    Attributes:
        tanks   The tanks in this module
    """
    tanks : list[Tank]