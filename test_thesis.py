from Environment import Environment
from problem_setup import read_environment

def read_and_test_thesis() -> None:
    environment = read_environment("Data\\Foesund_Strandkleiv_thesis\Parameters.json")
    print ("Pumping cost:" + str(environment.parameters.marginal_increase_pumping_cost))
    print ("Nmb of modules:" + str(len(environment.modules)))
    print ("module[2].index:" + str(environment.modules[2].index))
    print ("Nmb of module[2] tanks:" + str(len(environment.modules[2].tanks)))
    print ("M[2]T[1].index:" + str(environment.modules[2].tanks[1].index))
    print ("Nmb of M[2]T[0].transf to:" + str(len(environment.modules[2].tanks[0].transferable_to)))

if __name__ == "__main__":
    read_and_test_thesis()