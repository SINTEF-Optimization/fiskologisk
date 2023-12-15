import { useEffect, useRef, useState } from "react";
import { FiskologiskOptimizationForm } from "../components/fiskologiskOptimizationForm"
import { FiskologiskResults } from "../components/fiskologiskResults"
import axios from "axios";
import { SalmonPlanSolution } from "../components/productionPlanView/models";
import { useFiskologiskApiService } from "../services/fiskologiskApi/fiskologiskApiService";

export enum ResultState {
    Loading,
    Loaded,
    Infeasible
}

export const HomePage = () => {
    const [resultState, setResultState] = useState<ResultState>(ResultState.Loading);
    const [resultData, setResultData] = useState<any | null>(null)
    const activePollingTimer = useRef<NodeJS.Timeout>(); // In case we exit or something strange happens while waiting for a polling call or whatever
    const fiskologiskApi = useFiskologiskApiService();

    useEffect(() => {
        if (resultState === ResultState.Loading) {
            activePollingTimer.current = setTimeout(fetchData, 1000);
            return () => clearInterval(activePollingTimer.current);
        }
    });

    const fetchData = () => {
        if (resultData !== null) {
            return
        }
        console.log("Polling for fiskologisk results data");

        fiskologiskApi.ResultsApi.get().then((response) => {
            if (response["status"] === "computing") {
                activePollingTimer.current = setTimeout(fetchData, 1000)
            }
            else if (response["status"] === "failed") {
                setResultState(ResultState.Infeasible);
            }
            else {
                setResultState(ResultState.Loaded);
                setResultData(response);
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