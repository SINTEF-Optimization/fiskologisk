from __future__ import annotations

class Tank:
    """
    A water tank that might contain salmon.

    Attributes:
        index               Unique index among all tanks, used in MIP variable names
        transferable_to     The tanks salmon in this tank might be transered to
        transferable_from   The tanks that might transfer salmon to this tank
        inverse_volume          1.0 divided by the tank volume (1/m3)
        initial_use         Whether this tank was in use in the last period prior to the planning horizon. Part of initial conditions to the MIP problem.
    """
    index : int
    transferable_to : list[Tank]
    transferable_from : list[Tank]
    inverse_volume : float
    initial_use : bool

    def __init__(self, index: int, inverse_volume : float) -> None:
        self.index = index
        self.transferable_to = []
        self.transferable_from = []
        self.inverse_volume = inverse_volume
        self.initial_use = False

