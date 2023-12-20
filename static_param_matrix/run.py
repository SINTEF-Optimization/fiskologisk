import os, json, subprocess, multiprocessing

cases = []

for n_modules in [2,3,4,5,6,7]:
    for smolt_price in [140, 160, 180]:
        for tank_volume in [2500,3500,5000]:
            #for psp in [0.8, 1.0, 1.2]:
                #for hp in [0.8, 1.0, 1.2]:

                name=f"static_param_matrix_mod{n_modules}_spr{smolt_price}_tankvol{tank_volume}"
                cases.append((name,{
                    "modules/modules": n_modules,
                    # ??? "parameters/tanks_in_regulations": n_modules*4,
                    "parameters/smolt_price": smolt_price,
                    "modules/tank_volume": tank_volume,
                    #"weight_classes/post_smolt_revenue": 
                    #    [[0.5, psp*40.0], [1.0, psp*50.0]],
                    #"weight_classes/harvest_revenue_pr_kg":
                    #    [[2.0, hp*47.0],[3.0, hp*53.0],[4.0,hp*55.0],[5.0,hp*57.0],[6.0,hp*60],[7.0,hp*60.0],[8.0,hp*61.0]],
                }))
        

def read(filename):
    with open(filename,"r") as f:
        return json.load(f)

def write(filename, object):
    with open(filename,"w") as f:
        json.dump(object, f, indent=2)

src_corefile = "base_case/CoreProblem.json"
src_iterfile = "base_case/Iteration0.json"

def run_case(case):
    name,params = case
    print("Generating case", name)

    corefile = f"../Data/{name}/CoreProblem.json"
    def iterfile(n):
        return f"../Data/{name}/Iteration{n}.json"

    os.makedirs(f"../Data/{name}", exist_ok=True)
    core = read(src_corefile)
    iter_config = read(src_iterfile)
    for k,v in params.items():
        k1,k2 = k.split("/")
        print(f"setting {k1} / {k2} to {v}")
        core[k1][k2] = v

    write(corefile, core)
    write(iterfile(0), iter_config)

    n_modules = params["modules/modules"] 
    decomposition_method :int
    if n_modules <= 1:
        decomposition_method = 0
    elif n_modules <= 2:
        decomposition_method = 1
    else:
        decomposition_method = 2


    for i in [0,1,2,3]:
        print(f"iterfile: {iterfile(i)}")
        cmd = f"python ../run_iteration.py {iterfile(i)} -d {decomposition_method} | tee {iterfile(i)}.output.txt"
        print("executing cmd:",cmd)
        subprocess.run(cmd, shell=True, check=True)
    
    
with multiprocessing.Pool(24) as p:
    p.map(run_case, cases)

    
