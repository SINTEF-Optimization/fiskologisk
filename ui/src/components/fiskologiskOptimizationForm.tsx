import { useFiskologiskApiService } from "../services/fiskologiskApi/fiskologiskApiService";
import { SetParameterRequest } from "../services/fiskologiskApi/requests/setParametersRequest";
import { NumberInput } from "./numberInput";
import { useFormik } from "formik";

interface FormDataModel {
    numberOfTanks: number
    biomass: number
}

export interface FiskologiskOptimizationFormProps {
    onNewCalculationSubmitted: () => void
    triggerInfeasibleSolution: () => void
}

export const FiskologiskOptimizationForm = (props: FiskologiskOptimizationFormProps) => {
    const fiskologiskApi = useFiskologiskApiService();

    const initialValues: FormDataModel = {
        numberOfTanks: 4,
        biomass: 1000
    }

    const handleSubmission = async (values: FormDataModel) => {
        const request: SetParameterRequest = {
            numberOfTanks: values.numberOfTanks,
            biomassInKgs: values.biomass
        }
        await fiskologiskApi.ParametersApi.set(request);
    }

    const form = useFormik({
        initialValues,
        onSubmit: handleSubmission
    })

    const startOptimization = async () => {
        const response = await fiskologiskApi.RunOptimizationApi.start();
        if (response) {
            props.onNewCalculationSubmitted();
        }
    }

    return (
        <div>
            <form onSubmit={form.handleSubmit}>
                <NumberInput
                    name="numberOfTanks"
                    label="Number of tanks: "
                    value={form.values.numberOfTanks}
                    onChange={form.handleChange}
                    onBlur={form.handleBlur}
                />
                <NumberInput
                    name="biomass"
                    label="Max biomass in kgs: "
                    value={form.values.biomass}
                    onChange={form.handleChange}
                    onBlur={form.handleBlur}
                />
                <div>
                    <button type="submit">Set parameters</button>
                </div>
            </form>
            <button onClick={startOptimization}>Start Optimization</button>
            <button onClick={props.triggerInfeasibleSolution}>DEBUG: Trigger infeasible optimization</button>
        </div>
    )
}