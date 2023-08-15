from dataclasses import dataclass
from Module import Module
from WeightClass import WeightClass
from Period import Period
from Year import Year

@dataclass
class Environment:
    modules : list[Module]
    weight_classes : list[WeightClass]
    periods : list[Period]
    release_periods : list[Period]
    preplan_release_periods : list[Period]
    years : list[Year]