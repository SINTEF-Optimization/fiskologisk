import React from "react"
import {
    createContext,
    FC,
    useContext,
  } from "react"
import axios from "axios"
import { RunOptimizationApi } from "./constituentApis/runOptimizationApi"
import { ParametersApi } from "./constituentApis/parameterApi"
import { ResultsApi } from "./constituentApis/getResultsApi"

export interface FiskologiskApiService {
    RunOptimizationApi: RunOptimizationApi
    ParametersApi: ParametersApi
    ResultsApi: ResultsApi
}

export interface FiskologiskApiServiceProviderProps {
    children: React.ReactNode
}

const FiskologiskApiServiceContext = createContext<FiskologiskApiService | undefined>(undefined)

export const FiskologiskApiServiceProvider: FC<FiskologiskApiServiceProviderProps> = ({ children }) => {
    const client = axios.create({
        baseURL: 'http://127.0.0.1:5000'}) //TODO: replace this with actual IP of webserver on deployment
    const fiskologiskApiService: FiskologiskApiService = {
        RunOptimizationApi: new RunOptimizationApi(client),
        ParametersApi: new ParametersApi(client),
        ResultsApi: new ResultsApi(client)
    }

    return (
        <FiskologiskApiServiceContext.Provider value={fiskologiskApiService}>
        {children}
        </FiskologiskApiServiceContext.Provider>
    )
}

export const useFiskologiskApiService = () => {
    const context = useContext<FiskologiskApiService | undefined>(FiskologiskApiServiceContext)
    if (!context) {
        const serviceName = Object.keys({ FiskologiskApiServiceContext: FiskologiskApiServiceContext })[0]
        throw new Error(serviceName + " was not provided. "
        + "Make sure the component is a child of the required service provider")
    }
    return context
}
