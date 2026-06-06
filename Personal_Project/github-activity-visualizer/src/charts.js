const colors = ["#191713", "#ff5a2f", "#ccff00", "#2f6bff", "#8b5cf6", "#00a884", "#f4b400", "#ff2e88"];

export function renderHeatmap(target, days) {
  const cell = 13;
  const gap = 4;
  const width = 53 * (cell + gap) + 44;
  const height = 7 * (cell + gap) + 28;
  const max = d3.max(days, (day) => day.count) || 1;
  const color = d3.scaleSequential([0, max], d3.interpolateYlOrRd);

  const root = d3.select(target).html("");
  const svg = root.append("svg").attr("viewBox", `0 0 ${width} ${height}`).attr("width", width).attr("height", height).attr("role", "img");

  svg.selectAll("rect")
    .data(days)
    .join("rect")
    .attr("class", "day-cell")
    .attr("x", (day) => d3.timeWeek.count(d3.timeYear(day.date), day.date) * (cell + gap) + 32)
    .attr("y", (day) => day.date.getDay() * (cell + gap) + 16)
    .attr("width", cell)
    .attr("height", cell)
    .attr("fill", (day) => day.count ? color(day.count) : "rgba(25, 23, 19, 0.08)")
    .append("title")
    .text((day) => `${day.key}: ${day.count} commits`);

  svg.selectAll("text.month")
    .data(d3.timeMonths(d3.timeMonth(days[0].date), days[days.length - 1].date))
    .join("text")
    .attr("class", "axis-label")
    .attr("x", (date) => d3.timeWeek.count(d3.timeYear(date), date) * (cell + gap) + 32)
    .attr("y", 10)
    .text((date) => d3.timeFormat("%b")(date));
}

export function renderStreakBars(target, data) {
  const rows = [
    { label: "Current", value: data.current },
    { label: "Best", value: data.best },
    { label: "Active weeks", value: data.activeWeeks },
  ];
  const width = 420;
  const height = 210;
  const x = d3.scaleLinear().domain([0, d3.max(rows, (row) => row.value) || 1]).range([0, 270]);

  const svg = d3.select(target).html("").append("svg").attr("viewBox", `0 0 ${width} ${height}`).attr("role", "img");
  const group = svg.append("g").attr("transform", "translate(110, 28)");

  group.selectAll("rect.bg").data(rows).join("rect")
    .attr("class", "bar-bg")
    .attr("x", 0).attr("y", (_, index) => index * 54)
    .attr("width", 270).attr("height", 28);

  group.selectAll("rect.bar").data(rows).join("rect")
    .attr("class", "bar")
    .attr("x", 0).attr("y", (_, index) => index * 54)
    .attr("width", (row) => x(row.value)).attr("height", 28);

  group.selectAll("text.label").data(rows).join("text")
    .attr("class", "chart-label")
    .attr("x", -100).attr("y", (_, index) => index * 54 + 19)
    .text((row) => row.label);

  group.selectAll("text.value").data(rows).join("text")
    .attr("class", "slice-label")
    .attr("x", (row) => x(row.value) + 10).attr("y", (_, index) => index * 54 + 19)
    .text((row) => row.value);
}

export function renderLanguageChart(target, languages) {
  const width = 420;
  const height = 250;
  const radius = 92;
  const pie = d3.pie().value((language) => language.bytes).sort(null);
  const arc = d3.arc().innerRadius(48).outerRadius(radius);
  const labelArc = d3.arc().innerRadius(radius + 16).outerRadius(radius + 16);

  const svg = d3.select(target).html("").append("svg").attr("viewBox", `0 0 ${width} ${height}`).attr("role", "img");
  const group = svg.append("g").attr("transform", `translate(${width / 2}, ${height / 2})`);

  if (!languages.length) {
    group.append("text").attr("class", "chart-label").attr("text-anchor", "middle").text("No language data found");
    return;
  }

  group.selectAll("path")
    .data(pie(languages))
    .join("path")
    .attr("d", arc)
    .attr("fill", (_, index) => colors[index % colors.length])
    .attr("stroke", "#f3ead8")
    .attr("stroke-width", 3);

  group.selectAll("text")
    .data(pie(languages).filter((slice) => slice.data.percent >= 5))
    .join("text")
    .attr("class", "slice-label")
    .attr("transform", (slice) => `translate(${labelArc.centroid(slice)})`)
    .attr("text-anchor", "middle")
    .text((slice) => `${slice.data.name} ${slice.data.percent}%`);
}
