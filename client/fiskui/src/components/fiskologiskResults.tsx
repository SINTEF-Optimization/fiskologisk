import { ResultState } from "../pages/home";
import { ProductionPlanView } from "./productionPlanView/productionPlanView";

export interface FiskologiskResultsProps {
    resultState: ResultState,
    resultData: any | null
}

export const FiskologiskResults = (props: FiskologiskResultsProps) => {
    switch (props.resultState) {
        case ResultState.Loading:
            return <h3>LOADING!</h3>
        case ResultState.Loaded:
            return <ProductionPlanView solution={props.resultData}/>
        case ResultState.Infeasible:
        default:
            return <h3>Unfeasible solution or something went wrong. Please try again with different parameters.</h3>
    }
}