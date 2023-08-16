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