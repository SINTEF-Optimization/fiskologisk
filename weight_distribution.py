import math
from WeightClass import WeightClass

def get_weight_distributions(weight_classes: list[WeightClass], expected_weights: list[list[float]], variance_portion: float, max_weight: float) -> list[list[list[float]]]:
    
    distributions = []

    for exp_weights_by_deploy in expected_weights:
        distributions_by_deploy = []
        for mean_weight in exp_weights_by_deploy:
            if mean_weight < max_weight:
                distributions_by_deploy.append(get_weight_distributions_for_mass(mean_weight, weight_classes, variance_portion))
        distributions.append(distributions_by_deploy)
    
    return distributions


def get_weight_distributions_for_mass(mean_weight: float, weight_classes: list[WeightClass], variance_portion: float) -> list[float]:

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
