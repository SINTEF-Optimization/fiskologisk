import json

with open("process_status.json") as f:
        process = json.load(f)

run_id = process["run_id"]

with open("dist\instances\M2_T4_Y4_E14_P18\M2_T4_Y4_I1.json") as f:
        solution = json.load(f)

with open(f"results-{run_id}.json","w") as f:
        json.dump(solution, f)