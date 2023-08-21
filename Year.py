from Period import Period

class Year:
    """
    A year in the planning horizon.
    A year will start in March since this is the first in a group of release periods.
    Each year will have 360 days, distributed on 12 months of 30 days each.
    """

    year : int
    """The number of the year, like 2023"""

    periods : list[Period]
    """The periods/months of the year, starting with March"""

    def __init__(self, year: int) -> None:
        self.year = year
        self.periods = []
