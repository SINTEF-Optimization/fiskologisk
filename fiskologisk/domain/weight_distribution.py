import math
from fiskologisk.domain.WeightClass import WeightClass

def get_weight_distributions(weight_classes: list[WeightClass], expected_weights: list[list[float]], variance_portion: float, max_weight: float) -> list[list[list[float]]]:
    """Calculates the weight distributions into weight classes for different periods after different deploy periods

    args:
        - weight_classes: 'list[WeightClass]' The weight classes in the model
        - expected_weights: 'list[list[float]]' The expected weight in a period. Outermost list is deploy months, innermost list is months same as and after deploy month
        - variance_portion: 'float' The normal distribution variance divided by the expected weight, used in the normal distribution function into weight classes
        - max_weight: 'float' Upper limit of the expected weight for the periods when salmon can be harvested.

    returns:
        A three dimensional table of distributions.
        The outermost list is for each deploy period during the year, starting with March in position 0.
        All 12 months are represented, also months that are outside the deploy seasons.
        The list on the second level is for each period after the deploy period, starting the deploy period.
        The list goes up to the latest period salmon can be harvested.
        The innemost list is the distribution factor for each weight class for the given month after the given deploy month.
        The list has a non-negative number for each weight class, the distributions should sum up to 1.
    """
    
    distributions = []

    for exp_weights_by_deploy in expected_weights:
        distributions_by_deploy = []
        for mean_weight in exp_weights_by_deploy:
            if mean_weight < max_weight:
                distributions_by_deploy.append(get_weight_distributions_for_mass(mean_weight, weight_classes, variance_portion))
        distributions.append(distributions_by_deploy)
    
    return distributions


def get_weight_distributions_for_mass(mean_weight: float, weight_classes: list[WeightClass], variance_portion: float) -> list[float]:
    """Calculates the weight distributions into weight classes for salmon of a given expected mean weight

    args:
        - mean_weight: 'float' The mean weight of the individual salmon
        - weight_classes: 'list[WeightClass]' The weight classes in the model
        - variance_portion: 'float' The normal distribution variance divided by the expected weight, used in the normal distribution function into weight classes

    returns:
        A list is of the distribution factors for each weight class.
        The list has a non-negative number for each weight class, the distributions should sum up to 1.
    """

    variance = mean_weight * variance_portion
    distributions = []
    for weight_class in weight_classes:
        weight = weight_class.weight
        diff = (weight - mean_weight) / variance
        distributions.append(weight * math.exp(-0.5 * diff * diff))
    sum_distr = sum(distributions)

    normalized_distributions = []
    for distr in distributions:
        normalized_distributions.append(distr / sum_distr)
    
    return normalized_distributions
