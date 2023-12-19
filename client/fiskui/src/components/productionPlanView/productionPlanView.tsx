import * as d3 from "d3";
import { EndCause, SalmonPlanSolution, StartCause } from "./models";
import { useEffect } from "react";

export interface ProductionPlanViewProps {
    solution: SalmonPlanSolution
}

export const ProductionPlanView = (props: ProductionPlanViewProps) => {
    useEffect(() => {
        draw(props.solution);
    });

    const draw = (solution: SalmonPlanSolution) => {
        // set the dimensions and margins of the graph
        const margin = { top: 10, right: 30, bottom: 30, left: 60 },
            legendWidth = 300,
            heightBiomass = 300,
            width = 1200 - margin.left - margin.right,
            height = 600 - margin.top - margin.bottom;


        // clear any current graphs before drawing a new one: (TODO: optimize this in future. Only redraw if current data has changed in a meaningful way etc)
        d3.select(".prodPlanView").selectAll("svg").remove();

        // append the svg object to the div we return
        const svg = d3.select(".prodPlanView")
            .append("svg")
            .attr("width", width + margin.left + margin.right + legendWidth)
            .attr("height", height + margin.top + 2*margin.bottom + heightBiomass)
            .append("g")
            .attr("transform",
            "translate(" + margin.left + "," + margin.top + ")");

        // Timescale to display months
        const startTime = new Date(2021,2,1)
        const endTime = new Date((new Date(2021,2,1)).setMonth(startTime.getMonth()+48));
        const xScale = d3.scaleTime()
          .domain([startTime,endTime])
          //.nice()
          .range([0, width]);

        // Linear scale to display symbols
        const xScaleLinear = d3.scaleLinear()
            .domain([solution.planning_horizon.first_period - 1, solution.planning_horizon.first_period + solution.planning_horizon.years * 12 + 1])
            .range([0, width]);

        // Append x scale
        svg.append("g").attr("transform", "translate(0," + height + ")").call(d3.axisBottom(xScale));

        const tankRefToName = ({ module_idx, tank_idx } : {module_idx: number, tank_idx: number}) => `m${module_idx}-tank${tank_idx}`;

        enum Symbol {
            Empty,
            Deploy,
            Growing,
            PostSmolt,
            Harvest,
            TransferOut,
            TransferIn,
            BeforePlanningHorizon,
            AfterPlanningHorizon,
        }

        const yDomain = solution.modules.flatMap((m) =>
            m.tank_indices.map((t) =>
            tankRefToName({ module_idx: m.module_index, tank_idx: t })));

        const yScale = d3.scalePoint(
            yDomain, [margin.top, height - margin.bottom]);

        // Append y scale
        svg.append("g").attr("transform", "translate(-10,0)").call(d3.axisLeft(yScale));

        const symbolsMap: Map<string, { x: number, y: string, symbol: Symbol }> = new Map();
        const setSymbol = (x: number, y: string, symbol: Symbol) => symbolsMap.set(`${x},${y}`, { x, y, symbol });
        const maybeSetSymbol = (x: number, y: string, symbol: Symbol) => {
            if (!symbolsMap.has(`${x},${y}`)) setSymbol(x, y, symbol);

        };

        for (const cycle of solution.production_cycles) {
            for (const tank_cycle of cycle.tank_cycles) {
            const this_tank = tankRefToName({ module_idx: cycle.module, tank_idx: tank_cycle.tank });

            // Add start cause symbol.
            if (tank_cycle.start_cause === StartCause.Transfer) {
                const transfer = tank_cycle.transfer!;
                const from_tank = tankRefToName({ module_idx: cycle.module, tank_idx: transfer.from_tank });
                setSymbol(transfer.period, from_tank, Symbol.TransferOut);
                setSymbol(transfer.period, this_tank, Symbol.TransferIn);
            } else if (tank_cycle.start_cause === StartCause.PrePlanningDeploy) {
                setSymbol(tank_cycle.start_period - 1, this_tank, Symbol.BeforePlanningHorizon);
            } else if (tank_cycle.start_cause === StartCause.Deploy) {
                setSymbol(tank_cycle.start_period, this_tank, Symbol.Deploy);
            }

            // Add end cause symbol
            if (tank_cycle.end_cause === EndCause.Harvest) {
                setSymbol(tank_cycle.end_period, this_tank, Symbol.Harvest);
            } else if (tank_cycle.end_cause === EndCause.PostSmolt) {
                setSymbol(tank_cycle.end_period, this_tank, Symbol.PostSmolt);
            } else if (tank_cycle.end_cause === EndCause.PlanningHorizonExtension) {
                setSymbol(tank_cycle.end_period + 1, this_tank, Symbol.AfterPlanningHorizon);
            }

            // Add active/growing symbols to all active periods that have no other symbol
            for (const period of tank_cycle.period_biomasses) {
                maybeSetSymbol(period.period,this_tank, Symbol.Growing);
            }
            }
        }

        const dots = Array.from(symbolsMap.values());

        const shape = (pt: Symbol) => {

            if (pt === Symbol.Deploy) {
            return { shape: d3.symbol().size(80).type(d3.symbolCross)(), color: "green" }
            } else if (pt === Symbol.Growing) {
            return { shape: d3.symbol().size(20).type(d3.symbolCircle)(), color: "black" }
            } else if (pt === Symbol.Harvest) {
            return { shape: d3.symbol().size(80).type(d3.symbolCircle)(), color: "red" }
            } else if (pt === Symbol.PostSmolt) {
            return { shape: d3.symbol().size(80).type(d3.symbolCircle)(), color: "blue" }
            } else if (pt === Symbol.TransferOut) {
            return { shape: d3.symbol().size(80).type(d3.symbolTriangle)(), color: "darkred", transform: "rotate(180)" }
            } else if (pt === Symbol.TransferIn) {
            return { shape: d3.symbol().size(80).type(d3.symbolTriangle)(), color: "black", transform: "rotate(0)" }
            } else if (pt === Symbol.BeforePlanningHorizon) {
            return { shape: d3.symbol().size(80).type(d3.symbolTriangle)(), color: "gray", transform: "rotate(90)" }
            } else if (pt === Symbol.AfterPlanningHorizon) {
            return { shape: d3.symbol().size(80).type(d3.symbolTriangle)(), color: "gray", transform: "rotate(90)" }
            } else if (pt === Symbol.Empty) {
            return { shape: "", color: "" };
            } else {
            throw "unknown symbol type";
            }
            // const s =  d3.symbol().type(d3.symbolAsterisk)();
            // return s;
        }

        const legendText = (s: Symbol) => {
            switch (s) {
                case Symbol.Deploy:
                    return "Deploy";
                case Symbol.AfterPlanningHorizon:
                case Symbol.BeforePlanningHorizon:
                    return "Before/After Planning Horizon";
                case Symbol.Growing:
                    return "Growing";
                case Symbol.Harvest:
                    return "Harvest";
                case Symbol.PostSmolt:
                    return "Post Smolt";
                case Symbol.TransferIn:
                    return "Transfer In";
                case Symbol.TransferOut:
                    return "Transfer Out";
                case Symbol.Empty:
                default:
                    return "";
            }
        }

        const symbols = svg
            .append("g")
                .attr("stroke-width", 1)
            .selectAll("path")
            .data(dots)
            .join("path")
                .attr(
                "transform",
                d => `translate(${xScaleLinear(d.x)}, ${yScale(d.y)}) ` + (shape(d.symbol).transform ?? "")
                )
                .attr("fill", d => shape(d.symbol).color)
                .attr("stroke", d => shape(d.symbol).color)
                .attr("d", d => shape(d.symbol).shape);


        // ************************************** LEGEND *****************************************************

        // Get an array from 0 to number of values in Symbol enum (then remove 0 to avoid the empty symbol)
        const enumArray = Array.from(Array((Object.keys(Symbol).length / 2)-2).keys()).map(i => i+1);

        // Create the group for the legend
        const legend = svg
            .append("g")
                .attr("transform", "translate(" + width + ",-12)")
                //.attr("style", "outline: thick solid grey");

        // Add the symbols
        legend.selectAll("path")
        .data(enumArray)
        .join("path")
            .attr(
                "transform",
                d => `translate(${margin.right}, ${d*(24)}) ` + (shape(d).transform ?? "")
                )
            .attr("fill", d => shape(d).color)
            .attr("stroke", d => shape(d).color)
            .attr("d", d => shape(d).shape);

        // Add text
        legend.selectAll("text")
        .data(enumArray)
        .join("text")
            .attr(
                "transform",
                d => `translate(${margin.right + 24}, ${d*(24)}) `
                )
            .attr("fill", d => shape(d).color)
            .attr("stroke", d => shape(d).color)
            .attr("dy", ".35em")
            .text(function(d) { return legendText(d); });


        // ************************************** BIOMASS GRAPH *****************************************************

        // Keep track of total biomass. First track biomass for each period for each tank (in each module). Structure is Map<tankid, Array<number>>.
        const tankBiomassData: Map<number, Array<number>> = new Map();

        // Total mass is just an array of length 48. (since we know we will have exactly 48 months)
        const totalBiomassData = Array<number>(48).fill(0);
        for (const cycle of solution.production_cycles) {
            for (const tank_cycle of cycle.tank_cycles) {
                const existingTankArray = tankBiomassData.get(tank_cycle.tank) ?? Array<number>(48).fill(0);
                for (const period_masses of tank_cycle.period_biomasses) {
                    // For tank
                    existingTankArray[period_masses.period-24] = period_masses.biomass;

                    // For total
                    totalBiomassData[period_masses.period-24] = totalBiomassData[period_masses.period-24] + period_masses.biomass;
                }
                tankBiomassData.set(tank_cycle.tank, existingTankArray);
            }
        }

        // Create grouph for the biomass graph
        const biomass = svg
            .append("g")
            .attr("transform", "translate(0," + (height+margin.bottom) + ")");

        // Append x scale
        biomass.append("g").attr("transform", "translate(0," + heightBiomass + ")").call(d3.axisBottom(xScale));

        // Append y scale
        const yScaleBiomass = d3.scaleLinear()
        .domain([0, Math.max(...totalBiomassData)])
        .range([heightBiomass, 0]);
        // Append y scale
        const yAxisBiomass = biomass.append("g")
            .attr("transform", "translate(-10,0)")
            .call(d3.axisLeft(yScaleBiomass));

        // prepare a helper function
        var graphBiomass = d3.line()
        .x(function(d) { return xScaleLinear(d[0]) })
        .y(function(d) { return yScaleBiomass(d[1]) })

        biomass.append("path")
                .attr("fill", "none")
                .attr("stroke", "steelblue")
                .attr("stroke-width", 1.5)
                .attr("d", graphBiomass(totalBiomassData.map((value,index) => { return [index+24, value]})));



        // ************************************** DYNAMICALLY UPDATE GRAPH BASED ON MOUSE POSITION *****************************************************

        // Handles mouse event: checks what tank is "selected" and updates data in biomass graph

        const onMouseover = (tankIndex: number) => {
            const biomassData = tankBiomassData.get(tankIndex);
            if (biomassData){
                // Update scale
                yScaleBiomass.domain([0, Math.max(...biomassData)]) //TODO: the vertical line disappears after update for some reason..?
                yAxisBiomass.call(d3.axisLeft(yScaleBiomass));

                // Remove old data and graph new
                biomass.selectAll("path")
                    .remove();
                biomass.append("path")
                    .attr("fill", "none")
                    .attr("stroke", "steelblue")
                    .attr("stroke-width", 1.5)
                    .attr("d", graphBiomass(biomassData.map((value,index) => { return [index+24, value]})));
            }
        }

        const onMouseOut = () => {
            // Update scale
            yScaleBiomass.domain([0, Math.max(...totalBiomassData)])
            yAxisBiomass.call(d3.axisLeft(yScaleBiomass));

            // Remove old data and graph new
            biomass.selectAll("path")
                .remove();
            biomass.append("path")
                .attr("fill", "none")
                .attr("stroke", "steelblue")
                .attr("stroke-width", 1.5)
                .attr("d", graphBiomass(totalBiomassData.map((value,index) => { return [index+24, value]})));
        }

        // One rectangle for each row. First
        // First row:
        svg.append("rect")
            .attr('height', height/(2*tankBiomassData.size-1))
            .attr("width", width)
            .attr('fill', 'none')
            .attr('opacity', '0.5')
            .attr('pointer-events', 'all')
            .on('mouseover', function () {
                d3.select(this).attr('fill','maroon');
                onMouseover(0);
            })
            .on('mouseout', function () {
                d3.select(this).attr('fill','none');
                onMouseOut();
            });

        for (let i = 0; i<tankBiomassData.size-1; i++) {
            svg.append("rect")
                .attr('height', height/(tankBiomassData.size-1))
                .attr("width", width)
                .attr('fill','none')
                .attr('opacity', '0.5')
                .attr('y', i*(height/(tankBiomassData.size-1))+height/(2*tankBiomassData.size-1)-i*3.2) //TODO: This is pretty hacky. Should use the y-scale to properly extract dimensions.
                .attr('pointer-events', 'all')
                .on('mouseover', function () {
                    d3.select(this).attr('fill','maroon');
                    onMouseover(i+1);
                })
                .on('mouseout', function () {
                    d3.select(this).attr('fill','none');
                    onMouseOut();
                });
        }
    }

    return <div className="prodPlanView"/>
}
