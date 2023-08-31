import * as d3 from "d3";
import { EndCause, SalmonPlanSolution, StartCause } from "./model";

(async () => {

  const problem = await fetch("instances/M2_T4_Y4_E14_P18/CoreProblem.json").then(res => res.json());
  const solution = await fetch("instances/M2_T4_Y4_E14_P18/M2_T4_Y4_I1.json").then(res => res.json()) as SalmonPlanSolution;

  console.log(problem);
  console.log(solution);


  // set the dimensions and margins of the graph
  const margin = { top: 10, right: 30, bottom: 30, left: 60 },
    width = 1200 - margin.left - margin.right,
    height = 600 - margin.top - margin.bottom;

  // append the svg object to the body of the page
  const svg = d3.select("body")
    .append("svg")
    .attr("width", width + margin.left + margin.right)
    .attr("height", height + margin.top + margin.bottom)
    .append("g")
    .attr("transform",
      "translate(" + margin.left + "," + margin.top + ")");

  // const timeDomain = d3.extent(data.periods.flatMap(p => [p.start_date, p.end_date])) as [Date, Date]

  // const xScale = d3.scaleTime()
  //   .domain(timeDomain)
  //   .nice()
  //   .range([0, width]);

  const xScale = d3.scaleLinear()
    .domain([solution.planning_horizon.first_period - 1, solution.planning_horizon.first_period + solution.planning_horizon.years * 12 + 1])
    .range([0, width]);

  svg.append("g").attr("transform", "translate(0," + height + ")").call(d3.axisBottom(xScale));

  const tankRefToName = ({ module_idx, tank_idx }) => `m${module_idx}-tank${tank_idx}`;

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

  svg.append("g").attr("transform", "translate(-10,0)").call(d3.axisLeft(yScale));

  // TODO background color on deploy periods.

  const symbolsMap: Map<string, { x: number, y: string, symbol: Symbol }> = new Map();
  const setSymbol = (x: number, y: string, symbol: Symbol) => symbolsMap.set(`${x},${y}`, { x, y, symbol });
  const maybeSetSymbol = (x: number, y: string, symbol: Symbol) => {
    if (!symbolsMap.has(`${x},${y}`)) setSymbol(x, y, symbol);

  };

  for (const cycle of solution.production_cycles) {
    for (const tank_cycle of cycle.tank_cycles) {
      const this_tank = tankRefToName({ module_idx: cycle.module, tank_idx: tank_cycle.tank });

      // Add start cause symbol.
      if (tank_cycle.start_cause == StartCause.Transfer) {
        const transfer = tank_cycle.transfer!;
        const from_tank = tankRefToName({ module_idx: cycle.module, tank_idx: transfer.from_tank });
        setSymbol(transfer.period, from_tank, Symbol.TransferOut);
        setSymbol(transfer.period, this_tank, Symbol.TransferIn);
      } else if (tank_cycle.start_cause == StartCause.PrePlanningDeploy) {
        setSymbol(tank_cycle.start_period - 1, this_tank, Symbol.BeforePlanningHorizon);
      } else if (tank_cycle.start_cause == StartCause.Deploy) {
        setSymbol(tank_cycle.start_period, this_tank, Symbol.Deploy);
      }

      // Add end cause symbol
      if (tank_cycle.end_cause == EndCause.Harvest) {
        setSymbol(tank_cycle.end_period, this_tank, Symbol.Harvest);
      } else if (tank_cycle.end_cause == EndCause.PostSmolt) {
        setSymbol(tank_cycle.end_period, this_tank, Symbol.PostSmolt);
      } else if (tank_cycle.end_cause == EndCause.PlanningHorizonExtension) {
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

    if (pt == Symbol.Deploy) {
      return { shape: d3.symbol().size(80).type(d3.symbolCross)(), color: "green" }
    } else if (pt == Symbol.Growing) {
      return { shape: d3.symbol().size(20).type(d3.symbolCircle)(), color: "black" }
    } else if (pt == Symbol.Harvest) {
      return { shape: d3.symbol().size(80).type(d3.symbolCircle)(), color: "red" }
    } else if (pt == Symbol.PostSmolt) {
      return { shape: d3.symbol().size(80).type(d3.symbolCircle)(), color: "blue" }
    } else if (pt == Symbol.TransferOut) {
      return { shape: d3.symbol().size(80).type(d3.symbolTriangle)(), color: "darkred", transform: "rotate(180)" }
    } else if (pt == Symbol.TransferIn) {
      return { shape: d3.symbol().size(80).type(d3.symbolTriangle)(), color: "black", transform: "rotate(0)" }
    } else if (pt == Symbol.BeforePlanningHorizon) {
      return { shape: d3.symbol().size(80).type(d3.symbolTriangle)(), color: "gray", transform: "rotate(90)" }
    } else if (pt == Symbol.AfterPlanningHorizon) {
      return { shape: d3.symbol().size(80).type(d3.symbolTriangle)(), color: "gray", transform: "rotate(90)" }
    } else if (pt == Symbol.Empty) {
      return { shape: "", color: "" };
    } else {
      throw "unknown symbol type";
    }
    // const s =  d3.symbol().type(d3.symbolAsterisk)();
    // return s;
  }

  const symbols = svg
    .append("g")
    .attr("stroke-width", 1)
    .selectAll("path")
    .data(dots)
    .join("path")
    .attr(
      "transform",
      d => `translate(${xScale(d.x)}, ${yScale(d.y)}) ` + (shape(d.symbol).transform ?? "")
    )
    .attr("fill", d => shape(d.symbol).color)
    .attr("stroke", d => shape(d.symbol).color)
    .attr("d", d => shape(d.symbol).shape);

})();
