import * as d3 from "d3";
import { example1 } from "./example";
import { SalmonPlan, TankPeriod, TankRef } from "./model";

// set the dimensions and margins of the graph
const margin = { top: 10, right: 30, bottom: 30, left: 60 },
  width = 800 - margin.left - margin.right,
  height = 400 - margin.top - margin.bottom;

// append the svg object to the body of the page
const svg = d3.select("body")
  .append("svg")
  .attr("width", width + margin.left + margin.right)
  .attr("height", height + margin.top + margin.bottom)
  .append("g")
  .attr("transform",
    "translate(" + margin.left + "," + margin.top + ")");

const data = example1;

console.log(data);

const timeDomain = d3.extent(data.periods.flatMap(p => [p.start_date, p.end_date])) as [Date, Date]

const xScale = d3.scaleTime()
  .domain(timeDomain)
  .nice()
  .range([0, width]);

svg.append("g").attr("transform", "translate(0," + height + ")").call(d3.axisBottom(xScale));

const tankRefToName = ({ module_idx, tank_idx }) => `m${module_idx}-tank${tank_idx}`;

enum TankPeriodType {
  Empty,
  Deploy,
  Growing,
  PostSmolt,
  Harvest,
  TransferOut,
  TransferIn,
}

const isZero = (x: number) => Math.abs(x) <= 1e-5;

const tankPeriodToType = (data: SalmonPlan, tankperiod: TankPeriod) => {
  if (isZero(tankperiod.salmon_weight)) {
    return TankPeriodType.Empty;
  } else if (!isZero(tankperiod.salmon_deployed)) {
    return TankPeriodType.Deploy;
  } else if (!isZero(tankperiod.salmon_extracted)) {
    if (tankperiod.salmon_classes.every(c => data.weight_classes[c.class].individual_weight_ub <= 2.0)) {
      return TankPeriodType.PostSmolt;
    } else {
      return TankPeriodType.Harvest;
    }
  } else if (!isZero(tankperiod.salmon_transfer_out)) {
    return TankPeriodType.TransferOut;
  } else if (tankperiod.salmon_transferred_to_this_tank.find(t => !isZero(t.weight))) {
    return TankPeriodType.TransferIn;
  } else {
    return TankPeriodType.Growing;
  }
};

const yDomain = data.modules.flatMap((m, module_idx) =>
  m.tanks.map((_t, tank_idx) =>
    tankRefToName({ module_idx, tank_idx })));


const yScale = d3.scalePoint(
  yDomain, [margin.top, height - margin.bottom]);

svg.append("g").attr("transform", "translate(-10,0)").call(d3.axisLeft(yScale));

const dots: any[] = [];
for (const [module_idx, module] of data.modules.entries()) {
  for (const [tank_idx, tank] of module.tanks.entries()) {
    for (const tankperiod of tank.periods) {
      dots.push({
        x: data.periods[tankperiod.period].start_date,
        y: tankRefToName({ module_idx, tank_idx }),
        periodType: tankPeriodToType(data, tankperiod),
      });
    }
  }
}

const shape = (pt: TankPeriodType) => {

  if (pt == TankPeriodType.Deploy) {
    return { shape: d3.symbol().size(80).type(d3.symbolCross)(), color: "green" }
  } else if (pt == TankPeriodType.Growing) {
    return { shape: d3.symbol().size(20).type(d3.symbolCircle)(), color: "black" }
  } else if (pt == TankPeriodType.Harvest) {
    return { shape: d3.symbol().size(80).type(d3.symbolCircle)(), color: "red" }
  } else if (pt == TankPeriodType.PostSmolt) {
    return { shape: d3.symbol().size(80).type(d3.symbolCircle)(), color: "blue" }
  } else if (pt == TankPeriodType.TransferOut) {
    return { shape: d3.symbol().size(80).type(d3.symbolTriangle)(), color: "darkred" }
  } else if (pt == TankPeriodType.TransferIn) {
    return { shape: d3.symbol().size(80).type(d3.symbolTriangle)(), color: "black", transform: "rotate(180)" }
  } else {
    return { shape: "", color: "" };
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
    d => `translate(${xScale(d.x)}, ${yScale(d.y)}) ` + (shape(d.periodType).transform ?? "")
  )
  .attr("fill", d => shape(d.periodType).color)
  .attr("stroke", d => shape(d.periodType).color)
  .attr("d", d => shape(d.periodType).shape);
