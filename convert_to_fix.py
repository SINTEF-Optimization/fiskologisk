import sys
import json
import os

def convert_to_fix(from_file: str, to_file_local: str) -> None:

    file_dir = os.path.dirname(from_file)
    input_file = open(from_file, "r")
    data_json = json.load(input_file)

    start_dep_by_module = {} #production_cycles
    first_period = data_json["planning_horizon"]["first_period"]
    for dep_cyc in data_json["production_cycles"]:
        mod = dep_cyc["module"]
        dep_p = dep_cyc["deploy_period"]
        if dep_p >= first_period:
            if not mod in start_dep_by_module:
                start_dep_by_module[mod] = []
            start_dep_by_module[mod].append(dep_p)

    deploy_periods = []
    for mod, periods in sorted(start_dep_by_module.items()):
        deploy_periods.append({"module": mod, "start_periods": periods})


    out_json = { "deploy_periods": deploy_periods }

    to_file_path = os.path.join(file_dir, to_file_local)
    json_object = json.dumps(out_json, indent=4)
    with open(to_file_path, "w") as outfile:
        outfile.write(json_object)

if __name__ == "__main__":
    convert_to_fix(sys.argv[1], sys.argv[2])
