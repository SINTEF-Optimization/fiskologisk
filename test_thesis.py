from Environment import Environment
from Period import Period
from WeightClass import WeightClass
from read_problem import read_problem

def read_and_test_thesis() -> None:
    environment = read_problem("Data\\Foesund_Strandkleiv_thesis\CoreProblem.json")
    print_weight_classes(environment.weight_classes)
    print_deploy_weights("Pre-planning deploy periods", environment.preplan_release_periods)
    print_deploy_weights("Planning horizon deploy periods", environment.release_periods)
    print_weight_class_distributions(environment.release_periods)

def print_weight_classes(weight_classes: list[WeightClass]):
    print()
    print("Weight classes")
    for w_cl in weight_classes:
        print("W=" + str(w_cl.weight) + " Harv=" + str(w_cl.harvest_revenue) + " PS/ind=" + str(w_cl.post_smolt_revenue * w_cl.weight) + " PS/kg=" + str(w_cl.post_smolt_revenue))

def print_deploy_weights(heading: str, deploy_periods: list[Period]) -> None:
    print()
    print(heading + ":")
    for depl_p in deploy_periods:
        line = "Month " + str(depl_p.index) + " => "
        growths = []
        for period_idx in sorted(depl_p.periods_after_deploy):
            growths.append(str(period_idx) + ":" + str(depl_p.periods_after_deploy[period_idx].growth_factor))
        line += ", ".join(growths)
        print(line)

def print_weight_class_distributions(deploy_periods: list[Period]) -> None:
    print()
    print("Weight class distributions")
    for depl_p in deploy_periods:
        for period_idx in sorted(depl_p.periods_after_deploy):
            per_after_depl = depl_p.periods_after_deploy[period_idx]
            print("Dep-p=" + str(depl_p.index) + " p=" + str(period_idx) + " ExpW = " + str(per_after_depl.expected_weight) + " Distr-sum: " + str(sum(per_after_depl.weight_distribution)))
            print("Dep-p=" + str(depl_p.index) + " p=" + str(period_idx) + " Distr: " + str(per_after_depl.weight_distribution))

if __name__ == "__main__":
    read_and_test_thesis()