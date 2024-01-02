import { useEffect, useState } from "react";
import { SalmonPlanSolution } from "../components/productionPlanView/models";
import { ProductionPlanPresenter } from "../components/productionPlanPresenter/productionPlanPresenter";
import { DotPulse } from "@uiball/loaders";

export enum ResultState {
    Loading,
    Loaded,
}

interface ProblemSpecDir {
    dir: string,
    modules: number,
    smolt_price: number,
    tank_volume: number
}

export interface ProblemSpec {
    modules: number,
    smolt_price: number,
    tank_volume: number
}

export const HomePage = () => {
    const [resultState, setResultState] = useState<ResultState>(ResultState.Loading);
    const [resultData, setResultData] = useState<Map<string,SalmonPlanSolution>>(new Map<string,SalmonPlanSolution>())
    const [moduleValues,setModuleValues] = useState<Array<number>>([]);
    const [smoltPriceValues,setSmoltPriceValues] = useState<Array<number>>([]);
    const [tankVolumeValues,setTankVolumeValues] = useState<Array<number>>([]);

    useEffect(() => {
        const getData = async () => {

            const problemFetch = await fetch("problems.json");
            const problems = await problemFetch.json() as Array<ProblemSpecDir>

            const problemMap: Map<string,SalmonPlanSolution> = new Map<string,SalmonPlanSolution>();
            const modVals: Array<number> = [];
            const smoltVals: Array<number> = [];
            const volVals: Array<number> = [];
            for (var problem of problems) {
                const response = await fetch(`data/${problem.dir}/results_iter3.json`);
                const json = await response.json();
                problemMap.set(`${problem.modules}-${problem.smolt_price}-${problem.tank_volume}`, json as SalmonPlanSolution);
                if (!modVals.includes(problem.modules)) modVals.push(problem.modules);
                if (!smoltVals.includes(problem.smolt_price)) smoltVals.push(problem.smolt_price);
                if (!volVals.includes(problem.tank_volume)) volVals.push(problem.tank_volume);
            }

            setModuleValues(modVals);
            setSmoltPriceValues(smoltVals);
            setTankVolumeValues(volVals);
            setResultData(problemMap);
            setResultState(ResultState.Loaded);
        }
        getData();
    },[]);

    switch (resultState) {
        case ResultState.Loaded:
            return <ProductionPlanPresenter data={resultData} moduleValues={moduleValues} smoltPriceValues={smoltPriceValues} tankVolumeValues={tankVolumeValues}  />
        case ResultState.Loading:
        default:
            return (<div><DotPulse/></div>)
    }
}