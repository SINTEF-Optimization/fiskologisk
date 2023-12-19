const margin = { top: 10, right: 30, bottom: 30, left: 80 };
const legendWidth = 300;
const heightBiomass = 300;
const heightBiomassTitle = 100;
const width = 1200 - margin.left - margin.right;
const height = 600 - margin.top - margin.bottom;

// append the svg object to the body
const svg = d3.select("body")
    .append("svg")
    .attr("width", width + margin.left + margin.right + legendWidth)
    .attr("height", height + margin.top + 2*margin.bottom + heightBiomass + heightBiomassTitle)
    .append("g")
    .attr("transform",
    "translate(" + margin.left + "," + margin.top + ")");

// This stateless component renders a static "wheel" made of circles,
// and rotates it depending on the value of props.angle.
var wheel = d3.component("g")
    .create(function (selection){
    var minRadius = 3,
        maxRadius = 8,
        numDots = 10,
        wheelRadius = 30,
        rotation = 0,
        rotationIncrement = 3,
        radius = d3.scaleLinear()
            .domain([0, numDots - 1])
            .range([maxRadius, minRadius]),
        angle = d3.scaleLinear()
            .domain([0, numDots])
            .range([0, Math.PI * 2]);
    selection
        .selectAll("circle").data(d3.range(numDots))
        .enter().append("circle")
        .attr("cx", d => Math.round(Math.sin(angle(d)) * wheelRadius))
        .attr("cy", d => Math.round(Math.cos(angle(d)) * wheelRadius))
        .attr("r", d => Math.round(radius(d)));
    })
    .render(function (selection, d){
    selection.attr("transform", "rotate(" + d + ")");
    });

// This component with a local timer makes the wheel spin.
var spinner = (function (){
    var timer = d3.local();
    return d3.component("g")
    .create(function (selection, d){
        timer.set(selection.node(), d3.timer(function (elapsed){
        selection.call(wheel, elapsed * d.speed);
        }));
    })
    .render(function (selection, d){
        selection.attr("transform", "translate(" + d.x + "," + d.y + ")");
    })
    .destroy(function(selection, d){
        timer.get(selection.node()).stop();
        return selection
            .attr("fill-opacity", 1)
        //.transition().duration(1000)
            .attr("transform", "translate(" + d.x + "," + d.y + ") scale(0.01)")
            .attr("fill-opacity", 0);
    });
}());

async function main() {
    // Create spinner
    svg.call(spinner, {
        x: width / 2,
        y: height / 2,
        speed: 0.2
    });

    var i = 0;
    while (i<1000) {
        console.log(i);
    }

    // Kill spinner
    svg.call(spinner,[]);
}

main();
