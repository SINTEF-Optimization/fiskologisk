import { useEffect, useState } from "react";
import { FiskologiskOptimizationForm } from "../components/fiskologiskOptimizationForm"
import { FiskologiskResults } from "../components/fiskologiskResults"
import axios from "axios";
import { SalmonPlanSolution } from "../components/productionPlanView/models";

export enum ResultState {
    Loading,
    Loaded,
    Infeasible
}

export const HomePage = () => {
    const [resultState, setResultState] = useState<ResultState>(ResultState.Loading);
    const [resultData, setResultData] = useState<any | null>(null)

    useEffect(() => {
        if (resultState === ResultState.Loading) {
            let timerId = setTimeout(fetchData, 1000);
            return () => clearInterval(timerId);
        }
    });

    const fetchData = () => {
        if (resultData !== null) {
            return
        }
        console.log("Polling for fiskologisk results data");

        // TODO: replace this with proper endpoint
        axios.get(`https://api.chucknorris.io/jokes/random`).then((response) => {
            if (response.data["status"] === "computing") {
                setTimeout(fetchData, 1000)
            }
            else if (response.data["status"] === "failed") {
                setResultState(ResultState.Infeasible);
            }
            else {
                // Load static data. TODO: replace this with proper logic once backend returns real data
                fetch("M2_T4_Y4_I1.json").then(res => res.json()).then((result) => {
                    const mockresponse = result as SalmonPlanSolution;
                    console.log(mockresponse);
                    setResultState(ResultState.Loaded);
                    setResultData(mockresponse);
                });
            }
        });
    }

    const onNewCalculationSubmitted = () => {
        setResultData(null);
        setResultState(ResultState.Loading);
    }

    const triggerInfeasibleSolution = () => {
        setResultData(null);
        setResultState(ResultState.Infeasible)
    }

    return (
        <div>
            <FiskologiskOptimizationForm onNewCalculationSubmitted={onNewCalculationSubmitted} triggerInfeasibleSolution={triggerInfeasibleSolution}/>
            <FiskologiskResults resultState={resultState} resultData={resultData}/>
        </div>
    )
}