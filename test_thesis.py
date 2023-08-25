import gurobipy as gp
from Environment import Environment
from Period import Period
from WeightClass import WeightClass
from GurobiProblemGenerator import GurobiProblemGenerator
from read_problem import read_problem
from run_problem import run_problem

def read_and_test_problem(file_path: str) -> None:
    environment = read_problem(file_path)
    print_weight_classes(environment.weight_classes)
    print_deploy_weights("Pre-planning deploy periods", environment.preplan_release_periods)
    print_deploy_weights("Planning horizon deploy periods", environment.plan_release_periods)
    print_weight_class_distributions(environment.plan_release_periods)

def read_and_test_problem_and_model(file_path: str) -> None:
    environment = read_problem(file_path)
    gpm = GurobiProblemGenerator(environment)
    model = gpm.build_model()
    
def build_and_solve_problem(file_path: str) -> None:
    environment = read_problem(file_path)
    gpm = GurobiProblemGenerator(environment)
    model = gpm.build_model()
    model.optimize()
    model.write("test_thesis.lp")
    print_variables(list(gpm.extract_weight_variables.values()), 0.5)
    print_variables(list(gpm.population_weight_variables.values()), 0.5)
    print_variables(list(gpm.transfer_weight_variables.values()), 0.5)
    print_variables(list(gpm.contains_salmon_variables.values()), 0.5)
    print_variables(list(gpm.smolt_deployed_variables.values()), 0.5)
    print_variables(list(gpm.salmon_extracted_variables.values()), 0.5)
    print_variables(list(gpm.salmon_transferred_variables.values()), 0.5)
    
def print_variables(variables: list[gp.Var], min_val: float) -> None:
    for v in variables:
        if v.X > min_val:
            print(v.VarName + " = " + str(v.X))

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
        for period in depl_p.periods_after_deploy:
            growths.append(str(period.index) + ":" + str(depl_p.periods_after_deploy_data[period.index].growth_factor))
        line += ", ".join(growths)
        print(line)

def print_weight_class_distributions(deploy_periods: list[Period]) -> None:
    print()
    print("Weight class distributions")
    for depl_p in deploy_periods:
        for period in depl_p.periods_after_deploy:
            per_after_depl_data = depl_p.periods_after_deploy_data[period.index]
            print("Dep-p=" + str(depl_p.index) + " p=" + str(period.index) + " ExpW = " + str(per_after_depl_data.expected_weight) + " Distr-sum: " + str(sum(per_after_depl_data.weight_distribution)))
            print("Dep-p=" + str(depl_p.index) + " p=" + str(period.index) + " Distr: " + str(per_after_depl_data.weight_distribution))

if __name__ == "__main__":
    #read_and_test_problem("Data\Foesund_Strandkleiv_thesis\Iteration_0.json")
    #read_and_test_problem_and_model("Data\Foesund_Strandkleiv_thesis\Iteration_0.json")
    #read_and_test_problem("Data\One_module_three_years\Iteration_0.json")
    #build_and_solve_problem("Data\M1_T1_Y2_E0_P0\Iteration_0.json")
    #build_and_solve_problem("Data\M1_T2_Y2_E0_P0\Iteration_0.json")
    #build_and_solve_problem("Data\M1_T4_Y2_E0_P0\Iteration_0.json")
    #build_and_solve_problem("Data\M1_T4_Y3_E14_P18\Iteration_0.json")
    run_problem("Data\M1_T4_Y2_E14_P18\Iteration3.json")
    #run_problem("Data\M1_T4_Y3_E14_P18\Iteration1.json")
    #run_problem("Data\M1_T4_Y4_E14_P18\Iteration0.json")
    #run_problem("Data\M2_T4_Y2_E14_P18\Iteration1.json")
    #run_problem("Data\M2_T4_Y3_E14_P18\Iteration1.json")
    #run_problem("Data\M2_T4_Y3_E14_P18\Iteration1.json")
