from Period import Period

class Year:
    """
    A year before or in the planning horizon.
    A year will start in March since this is the first in a group of release periods.
    Each year will have 360 days, distributed on 12 months of 30 days each.

    Attributes:
        year       The number of the year, like 2023
        periods    The periods/months of the year, starting with March
    """
    year : int
    periods : list[Period]