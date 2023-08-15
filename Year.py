from dataclasses import dataclass
from Period import Period

@dataclass
class Year:
    year : int
    periods : list[Period]