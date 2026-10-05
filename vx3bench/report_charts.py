"""Chart specifications shared by the HTML and PDF reports.

A chart is described once, as data (a "spec"), and rendered twice:

* HTML -- by the small script in CHART_JS, entirely client-side, so the
  chart can be zoomed/panned on the time axis, series toggled, the Y axis
  switched between zero-based and fit-to-data, and every point inspected
  with a crosshair tooltip. Axis ticks are recomputed on every zoom, so a
  24 h run shows hours and zooming in switches to minutes/seconds.
* PDF  -- by report_pdf.py as vector graphics (see that module).

Time-axis units are chosen from the visible span (see time_unit()):
seconds up to 3 min, minutes up to 3 h, hours up to 3 days, then days.
"""
from __future__ import annotations

import colorsys
import html
import json
import math
from typing import Any, Iterable

BASE_COLORS = ["#48b8ff", "#6ce0b0", "#ffca63", "#ff7d8a", "#b38cff", "#55dce5",
               "#f59e5b", "#9bd35a", "#e879c9", "#7aa2ff"]

HTML_MAX_POINTS = 3000   # per series, embedded in the HTML report
PDF_MAX_POINTS = 4000    # per series, drawn in the PDF


def color(index: int) -> str:
    """Distinct colours for any number of series (30+ streams)."""
    if index < len(BASE_COLORS):
        return BASE_COLORS[index]
    hue = (index * 0.618033988749895) % 1.0
    r, g, b = colorsys.hls_to_rgb(hue, 0.66, 0.72)
    return "#%02x%02x%02x" % (int(r * 255), int(g * 255), int(b * 255))


# --------------------------------------------------------------------------- #
# Time axis
# --------------------------------------------------------------------------- #
TIME_UNITS = (("s", 1.0, "seconds"), ("min", 60.0, "minutes"), ("h", 3600.0, "hours"), ("d", 86400.0, "days"))


def time_unit(span_seconds: float) -> tuple[str, float, str]:
    span = abs(span_seconds or 0)
    if span <= 180:
        return TIME_UNITS[0]
    if span <= 3 * 3600:
        return TIME_UNITS[1]
    if span <= 72 * 3600:
        return TIME_UNITS[2]
    return TIME_UNITS[3]


def nice_step(span: float, target: int = 6) -> float:
    if span <= 0 or not math.isfinite(span):
        return 1.0
    raw = span / max(1, target)
    power = 10 ** math.floor(math.log10(raw))
    for factor in (1, 2, 2.5, 5, 10):
        if raw <= factor * power:
            return factor * power
    return 10 * power


def nice_ticks(lo: float, hi: float, target: int = 6) -> list[float]:
    if hi <= lo:
        hi = lo + 1
    step = nice_step(hi - lo, target)
    first = math.ceil(lo / step - 1e-9) * step
    ticks, value = [], first
    while value <= hi + step * 1e-9 and len(ticks) < 50:
        ticks.append(round(value, 10))
        value += step
    return ticks


def tick_label(seconds: float, suffix: str) -> str:
    """Label in the step unit; once the position passes the next larger unit
    it is written compound (e.g. '8h 30m' rather than '510min'), so zooming
    into the middle of a long run stays readable."""
    sec = round(seconds, 6)
    d, rem = divmod(sec, 86400)
    h, rem = divmod(rem, 3600)
    m, s = divmod(rem, 60)
    d, h, m = int(d), int(h), int(m)
    g = lambda v: f"{v:g}"
    if suffix == "d":
        return f"{g(sec / 86400)}d"
    if suffix == "h":
        return f"{g(sec / 3600)}h" if sec < 86400 else f"{d}d {g((sec - d * 86400) / 3600)}h"
    if suffix == "min":
        if sec < 3600:
            return f"{g(sec / 60)}min"
        return (f"{d}d " if d else "") + f"{h}h {m:02d}m" + (f" {g(s):0>2}s" if s else "")
    if sec < 60:
        return f"{g(sec)}s"
    return (f"{d}d " if d else "") + (f"{h}h " if (h or d) else "") + f"{m}m {g(s):0>2}s"


def time_ticks(x0: float, x1: float, target: int = 7) -> tuple[list[tuple[float, str]], str]:
    """Tick positions (in seconds) and labels, plus the axis title."""
    suffix, div, name = time_unit(x1 - x0)
    ticks = nice_ticks(x0 / div, x1 / div, target)
    return [(t * div, tick_label(t * div, suffix)) for t in ticks], f"Elapsed time ({name})"


def fmt_duration(seconds: float | None) -> str:
    if seconds is None or not math.isfinite(seconds):
        return "N/A"
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{seconds:.1f}s"
    total = int(round(seconds))
    d, rem = divmod(total, 86400)
    h, rem = divmod(rem, 3600)
    m, s = divmod(rem, 60)
    if d:
        return f"{d}d {h:02d}h {m:02d}m"
    if h:
        return f"{h}h {m:02d}m {s:02d}s"
    return f"{m}m {s:02d}s"


def fmt_value(value: float | None, unit: str = "") -> str:
    if value is None:
        return "N/A"
    text = f"{value:,.2f}"
    if not unit:
        return text
    return f"{text}{unit}" if unit in ("%", "°C") else f"{text} {unit}"


# --------------------------------------------------------------------------- #
# Specs
# --------------------------------------------------------------------------- #
def decimate(points: list[tuple[float, float]], max_points: int) -> list[tuple[float, float]]:
    """Min/max bucket decimation: keeps each bucket's lowest and highest
    point (in time order), so short dips and spikes survive downsampling."""
    if len(points) <= max_points:
        return points
    buckets = max(1, max_points // 2)
    size = len(points) / buckets
    out = []
    for b in range(buckets):
        chunk = points[int(b * size):int((b + 1) * size)]
        if not chunk:
            continue
        lo = min(chunk, key=lambda p: p[1])
        hi = max(chunk, key=lambda p: p[1])
        out.extend(sorted({lo, hi}, key=lambda p: p[0]))
    return out


def make_spec(title: str, unit: str, series: Iterable[dict[str, Any]], *, target: float | None = None,
              target_label: str = "", y_from_zero: bool = True, group: str = "", note: str = "") -> dict | None:
    """series items: {"name", "points": [(x_seconds, y)], "color"?}"""
    cleaned = []
    for i, s in enumerate(series):
        pts = [(float(x), float(y)) for x, y in s.get("points", [])
               if x is not None and y is not None and math.isfinite(float(x)) and math.isfinite(float(y))]
        if pts:
            pts.sort(key=lambda p: p[0])
            cleaned.append({"name": str(s.get("name", f"Series {i + 1}")), "color": s.get("color") or color(i), "points": pts})
    if not cleaned:
        return None
    return {"title": title, "unit": unit, "series": cleaned, "target": target, "target_label": target_label,
            "y_from_zero": y_from_zero, "group": group or title, "note": note}


def _compact(points, max_points):
    return [[round(x, 3), float(f"{y:.6g}")] for x, y in decimate(points, max_points)]


_chart_counter = [0]


def html_chart(spec: dict | None, *, height: int = 300) -> str:
    if not spec:
        return ""
    _chart_counter[0] += 1
    cid = f"chart{_chart_counter[0]}"
    payload = {
        "title": spec["title"], "unit": spec["unit"], "target": spec.get("target"),
        "targetLabel": spec.get("target_label", ""), "yFromZero": spec.get("y_from_zero", True),
        "height": height,
        "series": [{"name": s["name"], "color": s["color"], "points": _compact(s["points"], HTML_MAX_POINTS)}
                   for s in spec["series"]],
    }
    data = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")
    note = f'<p class="muted chart-note">{html.escape(spec["note"])}</p>' if spec.get("note") else ""
    return (f'<article class="chart ichart" id="{cid}"><h3>{html.escape(spec["title"])}</h3>{note}'
            f'<div class="ichart-host"><noscript><p class="muted">Enable JavaScript to view this chart.</p></noscript></div>'
            f'<script type="application/json" class="ichart-data">{data}</script></article>')


CHART_CSS = """
.ichart h3{margin:0 0 6px}
.chart-note{margin:0 0 6px;font-size:12px}
.ichart-toolbar{display:flex;flex-wrap:wrap;align-items:center;gap:6px;margin:4px 0 8px}
.ichart-toolbar button{padding:4px 9px;background:#203b55;color:#fff;border:1px solid #41617e;border-radius:5px;cursor:pointer;font-size:12px}
.ichart-toolbar button.on{background:#48b8ff;color:#04111c}
.ichart-toolbar .hint{color:#91a7bd;font-size:11px;margin-left:4px}
.ichart-plot{position:relative}
.ichart svg{width:100%;height:auto;display:block;user-select:none;touch-action:none;cursor:crosshair}
.ichart svg text{fill:#91a7bd;font-size:11px}
.ichart svg .grid{stroke:#29425e;stroke-width:1}
.ichart svg .axis{stroke:#52718e;stroke-width:1}
.ichart svg .axis-title{fill:#b7c9db;font-size:11px}
.ichart svg .target{stroke:#ffca63;stroke-dasharray:7 5;stroke-width:1.5}
.ichart svg .target-label{fill:#ffca63}
.ichart svg .cross{stroke:#edf5ff;stroke-width:1;opacity:.55}
.ichart svg path.series.dim{opacity:.18}
.ichart svg path.series.focus{stroke-width:3.5}
.ichart-tip{position:absolute;pointer-events:none;background:#06101cf0;border:1px solid #41617e;border-radius:8px;padding:7px 9px;font-size:12px;min-width:150px;max-width:340px;z-index:5;display:none}
.ichart-tip b{display:block;margin-bottom:4px;color:#65d6ff}
.ichart-tip div{display:flex;justify-content:space-between;gap:12px;white-space:nowrap}
.ichart-tip i{font-style:normal}
.ichart-legend{display:flex;flex-wrap:wrap;gap:5px;margin-top:8px;max-height:130px;overflow:auto}
.ichart-legend button{display:flex;align-items:center;gap:6px;padding:3px 8px;background:#081625;border:1px solid #29425e;border-radius:99px;color:#edf5ff;font-size:12px;cursor:pointer}
.ichart-legend button.off{opacity:.35;text-decoration:line-through}
.ichart-legend button span{width:10px;height:10px;border-radius:50%}
"""

# Client-side renderer. Mirrors time_unit()/nice_step() above.
CHART_JS = r"""
(function(){
const NS='http://www.w3.org/2000/svg';
const UNITS=[['s',1,'seconds'],['min',60,'minutes'],['h',3600,'hours'],['d',86400,'days']];
function unitFor(span){span=Math.abs(span);return span<=180?UNITS[0]:span<=10800?UNITS[1]:span<=259200?UNITS[2]:UNITS[3]}
function niceStep(span,n){if(!(span>0))return 1;const raw=span/n,p=Math.pow(10,Math.floor(Math.log10(raw)));for(const f of [1,2,2.5,5,10])if(raw<=f*p)return f*p;return 10*p}
function ticks(lo,hi,n){if(hi<=lo)hi=lo+1;const st=niceStep(hi-lo,n),out=[];for(let v=Math.ceil(lo/st-1e-9)*st;v<=hi+st*1e-9&&out.length<60;v+=st)out.push(+v.toFixed(10));return out}
function fmtNum(v){const a=Math.abs(v);return a>=1000?v.toLocaleString(undefined,{maximumFractionDigits:0}):a>=100?v.toFixed(1):a>=1?v.toFixed(2):v.toPrecision(3)}
function fmtUnit(v,u){if(v==null)return 'N/A';const t=fmtNum(v);return !u?t:(u==='%'||u==='°C')?t+u:t+' '+u}
function tickUnit(v,u){const t=Math.abs(v)>=1000?Math.round(v).toLocaleString():String(+v.toFixed(4));return !u?t:(u==='%'||u==='°C')?t+u:t+' '+u}
function dur(s){s=Math.max(0,s);if(s<60)return s.toFixed(1)+'s';s=Math.round(s);const d=Math.floor(s/86400),h=Math.floor(s%86400/3600),m=Math.floor(s%3600/60),x=s%60,p=n=>String(n).padStart(2,'0');return d?`${d}d ${p(h)}h ${p(m)}m`:h?`${h}h ${p(m)}m ${p(x)}s`:`${m}m ${p(x)}s`}
function tickLabel(sec,u){sec=+sec.toFixed(6);const d=Math.floor(sec/86400);let r=sec-d*86400;const h=Math.floor(r/3600);r-=h*3600;const m=Math.floor(r/60),s=+(r-m*60).toFixed(3),g=v=>String(+v.toFixed(4)),p=v=>g(v).padStart(2,'0');
  if(u==='d')return g(sec/86400)+'d';
  if(u==='h')return sec<86400?g(sec/3600)+'h':`${d}d ${g((sec-d*86400)/3600)}h`;
  if(u==='min')return sec<3600?g(sec/60)+'min':(d?d+'d ':'')+`${h}h ${p(m)}m`+(s?` ${p(s)}s`:'');
  return sec<60?g(sec)+'s':(d?d+'d ':'')+((h||d)?h+'h ':'')+`${m}m ${p(s)}s`}
function el(t,a,parent){const e=document.createElementNS(NS,t);for(const k in a)e.setAttribute(k,a[k]);if(parent)parent.appendChild(e);return e}
function bisect(pts,x){let lo=0,hi=pts.length-1;while(lo<hi){const m=(lo+hi)>>1;if(pts[m][0]<x)lo=m+1;else hi=m}return lo}

function build(root){
  const spec=JSON.parse(root.querySelector('.ichart-data').textContent);
  const host=root.querySelector('.ichart-host');host.innerHTML='';
  const all=spec.series.flatMap(s=>s.points);if(!all.length)return;
  const X0=Math.min(...spec.series.map(s=>s.points[0][0])),X1=Math.max(...spec.series.map(s=>s.points[s.points.length-1][0]));
  const S0=Math.min(0,X0),full=[S0,X1>S0?X1:S0+1];
  let view=[...full],yZero=spec.yFromZero!==false,hidden=new Set(),focus=-1;
  const W=900,H=spec.height||300,L=62,R=16,T=14,B=46,PW=W-L-R,PH=H-T-B;
  const bar=document.createElement('div');bar.className='ichart-toolbar';host.appendChild(bar);
  const btn=(label,fn,title)=>{const b=document.createElement('button');b.type='button';b.textContent=label;if(title)b.title=title;b.onclick=fn;bar.appendChild(b);return b};
  btn('Zoom in',()=>zoomAt((view[0]+view[1])/2,.5),'Zoom the time axis in');
  btn('Zoom out',()=>zoomAt((view[0]+view[1])/2,2),'Zoom the time axis out');
  btn('Reset',()=>{view=[...full];draw()},'Show the whole session');
  const yb=btn('Y: from 0',()=>{yZero=!yZero;draw()},'Toggle zero-based / fit-to-data Y axis');
  if(spec.series.length>1){btn('All',()=>{hidden.clear();draw()});btn('None',()=>{spec.series.forEach((_,i)=>hidden.add(i));draw()})}
  const hint=document.createElement('span');hint.className='hint';hint.textContent='Wheel/pinch: zoom time · drag: pan or select-zoom with Shift · double-click: reset · click legend to toggle';bar.appendChild(hint);
  const plot=document.createElement('div');plot.className='ichart-plot';host.appendChild(plot);
  const svg=el('svg',{viewBox:`0 0 ${W} ${H}`,role:'img','aria-label':spec.title},plot);
  const tip=document.createElement('div');tip.className='ichart-tip';plot.appendChild(tip);
  let legend=null;
  if(spec.series.length>1||spec.series[0].name){legend=document.createElement('div');legend.className='ichart-legend';host.appendChild(legend);
    spec.series.forEach((s,i)=>{const b=document.createElement('button');b.type='button';b.innerHTML=`<span style="background:${s.color}"></span>`;b.appendChild(document.createTextNode(s.name));
      b.onclick=ev=>{if(ev.altKey||ev.metaKey||ev.ctrlKey){hidden=new Set(spec.series.map((_,j)=>j).filter(j=>j!==i))}else{hidden.has(i)?hidden.delete(i):hidden.add(i)}draw()};
      b.onmouseenter=()=>{focus=i;draw()};b.onmouseleave=()=>{focus=-1;draw()};legend.appendChild(b)})}
  const layer=el('g',{},svg),cross=el('line',{class:'cross',y1:T,y2:T+PH,visibility:'hidden'},svg),sel=el('rect',{fill:'#48b8ff',opacity:.15,y:T,height:PH,visibility:'hidden'},svg);
  let yr=[0,1];
  const sx=x=>L+(x-view[0])/(view[1]-view[0])*PW,sy=y=>T+(1-(y-yr[0])/(yr[1]-yr[0]))*PH,ix=px=>view[0]+(px-L)/PW*(view[1]-view[0]);
  function visible(s){const pts=s.points;let a=Math.max(0,bisect(pts,view[0])-1),b=Math.min(pts.length,bisect(pts,view[1])+1);return pts.slice(a,b)}
  function draw(){
    yb.textContent=yZero?'Y: from 0':'Y: fit data';yb.classList.toggle('on',!yZero);
    layer.innerHTML='';
    const vis=spec.series.map((s,i)=>hidden.has(i)?[]:visible(s));
    let ys=vis.flat().map(p=>p[1]);if(spec.target!=null)ys.push(spec.target);if(!ys.length)ys=[0,1];
    let lo=Math.min(...ys),hi=Math.max(...ys);if(yZero)lo=Math.min(0,lo);if(hi<=lo){hi=lo+Math.max(1,Math.abs(lo)*.1)}
    const pad=(hi-lo)*.06;yr=[yZero&&lo===0?0:lo-pad,hi+pad];
    const yt=ticks(yr[0],yr[1],5);
    yt.forEach(v=>{const y=sy(v);if(y<T-1||y>T+PH+1)return;el('line',{class:'grid',x1:L,x2:L+PW,y1:y,y2:y},layer);const t=el('text',{x:L-6,y:y+4,'text-anchor':'end'},layer);t.textContent=tickUnit(v,spec.unit)});
    const u=unitFor(view[1]-view[0]);
    ticks(view[0]/u[1],view[1]/u[1],7).forEach(v=>{const x=sx(v*u[1]);if(x<L-1||x>L+PW+1)return;el('line',{class:'grid',x1:x,x2:x,y1:T,y2:T+PH},layer);const t=el('text',{x:x,y:T+PH+16,'text-anchor':'middle'},layer);t.textContent=tickLabel(v*u[1],u[0])});
    el('line',{class:'axis',x1:L,x2:L+PW,y1:T+PH,y2:T+PH},layer);el('line',{class:'axis',x1:L,x2:L,y1:T,y2:T+PH},layer);
    const at=el('text',{class:'axis-title',x:L+PW/2,y:H-6,'text-anchor':'middle'},layer);at.textContent=`Elapsed time (${u[2]})  ·  showing ${dur(view[0])} – ${dur(view[1])}`;
    const clip='clip'+root.id;const cp=el('clipPath',{id:clip},layer);el('rect',{x:L,y:T,width:PW,height:PH},cp);
    if(spec.target!=null){const y=sy(spec.target);el('line',{class:'target',x1:L,x2:L+PW,y1:y,y2:y},layer);const t=el('text',{class:'target-label',x:L+6,y:y+14},layer);t.textContent=spec.targetLabel||('Target '+fmtUnit(spec.target,spec.unit))}
    const g=el('g',{'clip-path':`url(#${clip})`},layer);
    vis.forEach((pts,i)=>{if(!pts.length)return;const s=spec.series[i];
      const d=pts.map((p,k)=>(k?'L':'M')+sx(p[0]).toFixed(1)+' '+sy(p[1]).toFixed(1)).join('');
      el('path',{d,class:'series'+(focus>=0?(focus===i?' focus':' dim'):''),fill:'none',stroke:s.color,'stroke-width':pts.length>1?2:4,'stroke-linecap':'round','stroke-linejoin':'round'},g)});
    if(legend)[...legend.children].forEach((b,i)=>b.classList.toggle('off',hidden.has(i)));
  }
  function zoomAt(cx,f){const span=Math.min(full[1]-full[0],Math.max(1,(view[1]-view[0])*f));let a=cx-(cx-view[0])*span/(view[1]-view[0]);a=Math.max(full[0],Math.min(a,full[1]-span));view=[a,a+span];draw()}
  const pt=ev=>{const p=svg.createSVGPoint();p.x=ev.clientX;p.y=ev.clientY;return p.matrixTransform(svg.getScreenCTM().inverse())};
  svg.addEventListener('wheel',ev=>{ev.preventDefault();const p=pt(ev);zoomAt(ix(Math.max(L,Math.min(L+PW,p.x))),ev.deltaY<0?.8:1.25)},{passive:false});
  let drag=null;
  svg.addEventListener('pointerdown',ev=>{const p=pt(ev);drag={x:p.x,view:[...view],select:ev.shiftKey};svg.setPointerCapture(ev.pointerId)});
  svg.addEventListener('pointermove',ev=>{const p=pt(ev);
    if(drag){if(drag.select){const a=Math.max(L,Math.min(drag.x,p.x)),b=Math.min(L+PW,Math.max(drag.x,p.x));sel.setAttribute('x',a);sel.setAttribute('width',b-a);sel.setAttribute('visibility','visible')}
      else{const dx=(p.x-drag.x)/PW*(drag.view[1]-drag.view[0]);let a=drag.view[0]-dx,span=drag.view[1]-drag.view[0];a=Math.max(full[0],Math.min(a,full[1]-span));view=[a,a+span];draw()}return}
    if(p.x<L||p.x>L+PW){cross.setAttribute('visibility','hidden');tip.style.display='none';return}
    const x=ix(p.x);cross.setAttribute('x1',p.x);cross.setAttribute('x2',p.x);cross.setAttribute('visibility','visible');
    const rows=[];spec.series.forEach((s,i)=>{if(hidden.has(i)||!s.points.length)return;const k=bisect(s.points,x);let q=s.points[Math.min(k,s.points.length-1)];if(k>0&&Math.abs(s.points[k-1][0]-x)<Math.abs(q[0]-x))q=s.points[k-1];rows.push([s,q])});
    rows.sort((a,b)=>b[1][1]-a[1][1]);const shown=rows.slice(0,14);
    tip.innerHTML=`<b>${dur(x)}</b>`+shown.map(([s,q])=>`<div><i style="color:${s.color}">● ${s.name.replace(/[&<>]/g,'')}</i><span>${fmtUnit(q[1],spec.unit)}</span></div>`).join('')+(rows.length>14?`<div><i>… ${rows.length-14} more</i></div>`:'');
    tip.style.display='block';const r=plot.getBoundingClientRect(),cx=ev.clientX-r.left;tip.style.left=(cx>r.width*.6?cx-tip.offsetWidth-14:cx+14)+'px';tip.style.top='8px'});
  svg.addEventListener('pointerup',ev=>{if(drag&&drag.select){const p=pt(ev),a=ix(Math.max(L,Math.min(drag.x,p.x))),b=ix(Math.min(L+PW,Math.max(drag.x,p.x)));sel.setAttribute('visibility','hidden');if(b-a>0.5){view=[a,b];draw()}}drag=null});
  svg.addEventListener('pointerleave',()=>{cross.setAttribute('visibility','hidden');tip.style.display='none'});
  svg.addEventListener('dblclick',()=>{view=[...full];draw()});
  draw();
}
document.querySelectorAll('.ichart').forEach(root=>{try{build(root)}catch(e){console.error(e)}});
})();
"""
