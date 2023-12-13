import json
import os
import random
import subprocess
from flask import Flask, request, make_response
from flask_cors import CORS, cross_origin

app = Flask(__name__)
CORS(app, support_credentials=True)

def check_pid(pid):
    """ Check For the existence of a unix pid. """
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    else:
        return True


@app.get("/parameters")
def get_parameter():
    with open("parameters.json") as f:
        return json.load(f)

@app.post("/parameters")
@cross_origin(supports_credentials=True)
def set_parameters():
    try:
        parameters = request.get_json()
        print("Recieved POST to parameters endpoint!")
        print(parameters)
        with open("parameters.json","w") as f:
            json.dump(parameters, f)
        return ('',204)
    except:
        return ('',500)

@app.post("/start")
def start():
    run_id = random.randint(0,10000)
    process = subprocess.Popen(["python","optimize.py", f"results-{run_id}.json"])
    with open("process_status.json","w") as f:
        json.dump({"run_id": run_id, "pid": process.pid}, f)
    return ('',204)


@app.get("/results")
def results():
    with open("process_status.json") as f:
        process = json.load(f)

    pid = process["pid"]
    run_id = process["run_id"]

    if check_pid(pid):
        return {"status": "computing"}
    else:
        results_filename = f"results-{run_id}.json"
        if os.path.isfile(results_filename):
            with open(results_filename) as f:
                return json.load(f)
        else:
            return {"status": "failed"}


