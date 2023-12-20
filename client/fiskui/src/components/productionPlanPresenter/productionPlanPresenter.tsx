import { useEffect, useState } from "react";
import { ProblemSpec } from "../../pages/home"
import { SalmonPlanSolution } from "../productionPlanView/models"
import React from "react";
import { ProductionPlanView } from "../productionPlanView/productionPlanView";

export interface ProductionPlanPresenterProps {
    data: Map<string,SalmonPlanSolution>,
    moduleValues: Array<number>,
    smoltPriceValues: Array<number>,
    tankVolumeValues: Array<number>
}

export const ProductionPlanPresenter = (props: ProductionPlanPresenterProps) => {
    const [activeSpec,setActiveSpec] = useState<ProblemSpec>({
        modules:props.moduleValues[0],
        smolt_price:props.smoltPriceValues[0],
        tank_volume:props.tankVolumeValues[0]});

    useEffect(()=>{console.log(props.data)},[]);

    const onModuleChange = (event: React.ChangeEvent<HTMLSelectElement>) => {
        const newModValue = parseInt(event.target.value,10);
        setActiveSpec({...activeSpec, modules:newModValue});
    }

    const onSmoltChange = (event: React.ChangeEvent<HTMLSelectElement>) => {
        const newSmoltValue = parseInt(event.target.value,10);
        setActiveSpec({...activeSpec, smolt_price:newSmoltValue});
    }

    const onTankVolumeChange = (event: React.ChangeEvent<HTMLSelectElement>) => {
        const newTankValue = parseInt(event.target.value,10);
        setActiveSpec({...activeSpec, tank_volume:newTankValue});
    }

    return (<div>
            <label>
                {"  Number of modules:  "}
                <select value={activeSpec.modules} onChange={onModuleChange}>
                    {React.Children.toArray(props.moduleValues.map(mv => {return <option value={mv}>{mv}</option>}))}
                </select>
            </label>
            <label>
                {"  Smolt price:  "}
                <select value={activeSpec.smolt_price} onChange={onSmoltChange}>
                    {React.Children.toArray(props.smoltPriceValues.map(sv => {return <option value={sv}>{sv}</option>}))}
                </select>
            </label>
            <label>
                {"  Tank volume:  "}
                <select value={activeSpec.tank_volume} onChange={onTankVolumeChange}>
                    {React.Children.toArray(props.tankVolumeValues.map(tv => {return <option value={tv}>{tv}</option>}))}
                </select>
            </label>
            {props.data.get(`${activeSpec.modules}-${activeSpec.smolt_price}-${activeSpec.tank_volume}`) && <ProductionPlanView solution={props.data.get(`${activeSpec.modules}-${activeSpec.smolt_price}-${activeSpec.tank_volume}`)!}></ProductionPlanView> }
        </div>);
}