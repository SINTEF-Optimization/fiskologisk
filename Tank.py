from __future__ import annotations

class Tank:
    """
    A water tank that might contain salmon.
    """
    index : int
    """Unique index among all tanks"""

    transferable_to : list[Tank]
    """The tanks salmon in this tank might be transered to"""

    transferable_from : list[Tank]
    """The tanks that might transfer salmon to this tank"""

    inverse_volume : float
    """1.0 divided by the tank volume (1/m3)"""

    initial_weight : float
    """Weight of salmon in this tank at the start of the planning horizon, deployed in a preplanning period. If zero, tank was empty at start of planning horizon. Part of initial conditions to the MIP problem."""

    initial_deploy_period : int
    """Index of preplanning deploy period for the salmon at the start of the planning horizon, only relevant if initial_weight is positive. Part of initial conditions to the MIP problem."""

    def __init__(self, index: int, inverse_volume : float) -> None:
        self.index = index
        self.transferable_to = []
        self.transferable_from = []
        self.inverse_volume = inverse_volume
        self.initial_weight = 0.0
        self.initial_deploy_period = 0
