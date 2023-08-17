import math

def get_weight_distributions(weight_classes: list[float], expected_weights: list[list[float]], variance_portion: float, max_weight) -> list[list[list[float]]]:
    
    distributions = []

    for exp_weights_by_deploy in expected_weights:
        distributions_by_deploy = []
        for mean_weight in exp_weights_by_deploy:
            if mean_weight < max_weight:
                distributions_by_deploy.append(get_weight_distributions_for_mass(mean_weight, weight_classes, variance_portion))
        distributions.append(distributions_by_deploy)
    
    return distributions


def get_weight_distributions_for_mass(mean_weight: float, weight_classes: list[float], variance_portion: float) -> list[float]:

    variance = mean_weight * variance_portion
    normal_distributions = []
    for weight in weight_classes:
        diff = (weight - mean_weight) / variance
        normal_distributions.append(math.exp(-0.5 * diff * diff))
    sum_distr = sum(normal_distributions)

    distributions = []
    for i in range(len(weight_classes)):
        distributions.append(normal_distributions[i] * weight_classes[i] / (sum_distr * mean_weight))
    
    return distributions

