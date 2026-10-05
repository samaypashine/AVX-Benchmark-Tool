import html,json,math
from pathlib import Path
from .bottleneck import analyze
from .report_fps import fps_bar_chart, fps_history_spec, fps_min_spec
from .report_charts import make_spec, html_chart, CHART_CSS, CHART_JS, fmt_duration
C=["#48b8ff","#6ce0b0","#ffca63","#ff7d8a","#b38cff","#55dce5"]
def e(x):return html.escape(str(x))
def fmt(v,k=""):
    if v is None:return "N/A"
    if isinstance(v,(dict,list)):return json.dumps(v,indent=2)
    if isinstance(v,(int,float)):
        if "bytes" in k:return f"{v/2**30:,.2f} GiB"
        if k.endswith("_ms"):return f"{v:,.2f} ms"
        if k.endswith("_fps") or k=="fps":return f"{v:,.2f} FPS"
        if "percent" in k:return f"{v:,.2f}%"
        return f"{v:,.2f}"
    return str(v)
def long(v):return isinstance(v,(dict,list)) or len(str(v))>180 or str(v).count("\n")>3
def rows(d,exclude=()):
    out=[]
    for k,v in d.items():
        if k in exclude:continue
        text=fmt(v,k);content=f"<pre>{e(text)}</pre>" if long(text) else e(text)
        if long(text):content=f'<div class="clamp">{content}</div><button class="more">Show More</button>'
        out.append(f'<div class="row"><span>{e(k.replace("_"," ").title())}</span><div>{content}</div></div>')
    return ''.join(out)
def cards(d,defs):
    items=[]
    for k,label in defs:
        value=d.get(k)
        detail=""
        if isinstance(value,dict) and "mean" in value:
            detail=f'<small>min {e(fmt(value.get("min"),k))} · max {e(fmt(value.get("max"),k))}</small>'
            value=value.get("mean")
        items.append(f'<div class="metric"><span>{e(label)}</span><strong>{e(fmt(value,k))}</strong>{detail}</div>')
    return '<div class="cards">'+''.join(items)+'</div>'
def system_info(inv):
    if not inv:return '<p class="muted">No inventory captured.</p>'
    top=cards(inv,[("hostname","Hostname"),("os","OS"),("architecture","Architecture"),
                   ("cpu_model","CPU Model"),("physical_cpu_count","Physical Cores"),
                   ("logical_cpu_count","Logical Cores"),                   ("ram_total_bytes","RAM Total"),("memory_bus_speed_mhz","Memory Bus Speed"),
                   ("python","Python")])
    gpus=inv.get("gpus") or []
    gpu_html=''.join(
        f'<div class="metric"><span>{e(g.get("name") or "GPU " + str(g.get("index")))}</span><strong>{e(g.get("name","Unknown"))}</strong>'
        f'<small>{e(fmt(g.get("vram_total_bytes"),"vram_total_bytes"))} VRAM &middot; driver {e(g.get("driver_version","?"))}</small></div>'
        for g in gpus
    ) or '<div class="metric"><span>GPU</span><strong>None detected</strong></div>'
    disks=inv.get("disks") or []
    disk_html=''.join(
        f'<div class="metric"><span>{e(d.get("mountpoint"))}</span><strong>{e(fmt(d.get("used_bytes"),"used_bytes"))} / {e(fmt(d.get("total_bytes"),"total_bytes"))}</strong>'
        f'<small>{e(d.get("filesystem",""))} &middot; {d.get("percent",0):.1f}% used &middot; {e(d.get("device",""))}</small></div>'
        for d in disks
    ) or '<div class="metric"><span>Disk</span><strong>None detected</strong></div>'
    usb=inv.get("usb_devices") or []
    usb_html='<ul class="usb-list">'+(''.join(f'<li>{e(u.get("description",""))}</li>' for u in usb) or '<li class="muted">No USB devices detected or enumeration unavailable.</li>')+'</ul>'
    warnings=[w for w in (inv.get("gpu_inventory_warning"),inv.get("usb_inventory_warning")) if w]
    warn_html=''.join(f'<p class="muted">{e(w)}</p>' for w in warnings)
    return f'''{top}
        <h3>GPUs</h3><div class="cards">{gpu_html}</div>
        <h3>Storage</h3><div class="cards">{disk_html}</div>
        <h3>USB Devices</h3>{usb_html}
        {warn_html}'''
def points(samples,path,scale=1):
    r=[]
    for s in samples:
        try:
            v=s
            for k in path:v=v[k]
            if v is not None and math.isfinite(float(v)):r.append((float(s["elapsed_seconds"]),float(v)/scale))
        except Exception:pass
    return r
def chart(title,unit,samples,defs):
    """Chart spec (rendered by report_charts for HTML and report_pdf for PDF)."""
    return make_spec(title,unit.strip(),[{"name":name,"color":C[i%len(C)],"points":points(samples,path,scale)} for i,(name,path,scale) in enumerate(defs)])
def telemetry_summary_html(summary):
    if not summary:return '<p class="muted">No telemetry samples were recorded.</p>'
    blocks=[]
    for key,value in summary.items():
        if key=="gpus":
            continue
        if isinstance(value,dict) and value:
            blocks.append(f'<div class="summary-block"><h3>{e(key.replace("_"," ").title())}</h3><div class="summary-row">{summary_metric(value,"mean","Mean",key)}{summary_metric(value,"min","Minimum",key)}{summary_metric(value,"max","Maximum",key)}{summary_metric(value,"start","Start",key)}{summary_metric(value,"end","End",key)}</div></div>')
        else:
            blocks.append(f'<div class="summary-block"><h3>{e(key.replace("_"," ").title())}</h3><div class="summary-row">{summary_metric({"value":value},"value","Value",key)}</div></div>')
    for index,gpu in enumerate(summary.get("gpus",[])):
        blocks.append(gpu_summary_html(gpu,index))
    return ''.join(blocks)
def summary_metric(value,key,label,format_key):
    return f'<div class="summary-metric"><span>{e(label)}</span><strong>{e(fmt(value.get(key),format_key))}</strong></div>'
# Each GPU metric in the telemetry summary is a {min,mean,max,start,end} dict
# (same shape as every other telemetry key). It was previously passed straight
# to fmt(), which dumped the raw dict as JSON; it is now rendered with the same
# Mean/Minimum/Maximum/Start/End cards as the other sub-sections.
GPU_SUMMARY_METRICS=[("gpu_percent","GPU Utilization","%"),("memory_controller_percent","Memory Controller Utilization","%"),
    ("vram_used_bytes","VRAM Used","bytes"),("vram_percent","VRAM Utilization","%"),("temperature_c","Temperature","°C"),
    ("power_watts","Power Draw","W"),("graphics_clock_mhz","Graphics Clock","MHz"),("memory_clock_mhz","Memory Clock","MHz"),
    ("fan_percent","Fan Speed","%"),("pcie_tx_mbps","PCIe TX","MB/s"),("pcie_rx_mbps","PCIe RX","MB/s"),
    ("encoder_percent","Encoder Utilization","%"),("decoder_percent","Decoder Utilization","%")]
def fmt_unit(v,unit):
    if v is None:return "N/A"
    if unit=="bytes":return fmt(v,"bytes")
    if unit=="%":return f"{v:,.2f}%"
    return f"{v:,.2f} {unit}"
def gpu_summary_html(gpu,index):
    name=gpu.get("name") or f"GPU {index}"
    sections=[]
    for key,label,unit in GPU_SUMMARY_METRICS:
        stats=gpu.get(key)
        if not isinstance(stats,dict) or not stats:continue  # metric not reported by this GPU/driver
        cells=''.join(f'<div class="summary-metric"><span>{e(title)}</span><strong>{e(fmt_unit(stats.get(stat),unit))}</strong></div>' for stat,title in (("mean","Mean"),("min","Minimum"),("max","Maximum"),("start","Start"),("end","End")))
        sections.append(f'<div class="summary-block"><h4>{e(label)}</h4><div class="summary-row">{cells}</div></div>')
    body=''.join(sections) or '<p class="muted">No readings were recorded for this GPU.</p>'
    return f'<div class="summary-block gpu-summary"><h3>{e(name)} <small>GPU {e(index)}</small></h3>{body}</div>'
AI_STATE_LABELS={"target_met":("Target met","ok"),"saturated":("Saturated","warn"),"instance_failed":("Instance failed to start","bad"),"max_instances":("Max processes reached, target not met","bad"),
    "failed":("Failed","bad"),"scaling":("Still scaling at end of session","warn"),"fixed":("Fixed instance count (no target)","")}
def _n(v,d=2):return "N/A" if v is None else f"{v:,.{d}f}"
def model_summary_html(models):
    if not models:return '<p class="muted">No AI model workers were enabled for this session.</p>'
    out=[]
    for m in models:
        label,tone=AI_STATE_LABELS.get(m.get("state"),(m.get("state","unknown"),""))
        target=m.get("target_fps") or 0
        verdict=(f'{m.get("required_instances")} instance(s) required' if m.get("required_instances") else
                 f'{m.get("instance_count",0)} instance(s) running')
        inst_rows=''.join(
            f'<tr class="{"retired" if x.get("retired") else ""}"><td>#{e(x.get("instance"))}</td><td>{e("stopped" if x.get("retired") else x.get("state",""))}</td>'
            f'<td>{_n(x.get("assigned_fps") or x.get("target_rate_fps"))}</td><td>{_n(x.get("fps"))}</td><td>{_n(x.get("compute_fps"),1)}</td>'
            f'<td>{int(x.get("inferences") or 0):,}</td><td>{_n(x.get("latency_mean_ms"),3)}</td><td>{_n(x.get("latency_p95_ms"),3)}</td>'
            f'<td>{_n(x.get("latency_p99_ms"),3)}</td><td>{_n(x.get("cold_start_latency_ms"),1)}</td><td>{e(x.get("saturation_note") or x.get("retired_reason") or x.get("error") or "")}</td></tr>'
            for x in (m.get("instances") or [])+(m.get("retired_instances") or []))
        events=''.join(f'<tr><td>{e(fmt_duration(ev.get("elapsed_seconds")))}</td><td>{e(ev.get("action"))}</td><td>{e(ev.get("instances"))}</td>'
                       f'<td>{_n(ev.get("total_fps"))}</td><td>{e(ev.get("reason"))}</td></tr>' for ev in m.get("scaling_events") or [])
        steps=''.join(f'<tr><td>{e(x.get("instances"))}</td><td>{_n(x.get("total_fps"))}</td><td>{"N/A" if not target else f"{100*(x.get("total_fps") or 0)/target:.1f}%"}</td></tr>' for x in m.get("scaling_steps") or [])
        pct=(100*(m.get("fps") or 0)/target) if target else None
        out.append(f'''<div class="ai-model">
            <div class="ai-model-head"><h3>{e(m.get("name","Model"))}</h3><span class="pill {tone}">{e(label)}</span></div>
            <div class="cards">
              <div class="metric"><span>Target</span><strong>{_n(target) if target else "None"} FPS</strong><small>{_n(m.get("target_fps_per_stream"),1)} FPS × {e(m.get("stream_count",0))} AI-enabled stream(s)</small></div>
              <div class="metric"><span>Achieved (steady avg)</span><strong>{_n(m.get("fps"))} FPS</strong><small>{"" if pct is None else f"{pct:.1f}% of target"}</small></div>
              <div class="metric"><span>Instances</span><strong>{e(verdict)}</strong><small>{e(m.get("active_device") or m.get("provider") or m.get("requested_device",""))} · {"separate processes" if m.get("worker_mode")=="process" else "threads in one shared process"}</small></div>
              <div class="metric"><span>Latency</span><strong>{_n(m.get("latency_mean_ms"),2)} ms</strong><small>worst instance p95 {_n(m.get("latency_p95_ms"),2)} · p99 {_n(m.get("latency_p99_ms"),2)} ms</small></div>
            </div>
            <p class="muted">{e(m.get("scaling_reason") or "")}</p>
            {f'<p class="muted"><b>Device fallback:</b> {e(m["device_fallback"])}</p>' if m.get("device_fallback") else ""}
            <div class="table-wrap"><table class="data"><thead><tr><th>Instance</th><th>State</th><th>Assigned FPS</th><th>Avg FPS (lifetime)</th><th>Capacity FPS</th><th>Inferences</th><th>Mean ms</th><th>p95 ms</th><th>p99 ms</th><th>Cold start ms</th><th>Note</th></tr></thead><tbody>{inst_rows}</tbody></table></div>
            {f'<details open><summary>Settled total at each instance count</summary><div class="table-wrap"><table class="data"><thead><tr><th>Instances</th><th>Total FPS</th><th>% of target</th></tr></thead><tbody>{steps}</tbody></table></div></details>' if steps else ""}
            {f'<details><summary>Scaling decisions ({len(m.get("scaling_events") or [])})</summary><div class="table-wrap"><table class="data"><thead><tr><th>At</th><th>Action</th><th>Instances</th><th>Total FPS</th><th>Reason</th></tr></thead><tbody>{events}</tbody></table></div></details>' if events else ""}
        </div>''')
    return ''.join(out)
def model_fps_specs(models):
    """One chart per model: total throughput vs target, plus each instance."""
    specs=[]
    for m in models:
        series=[{"name":"Total","color":"#edf5ff","points":[(x.get("elapsed_seconds"),x.get("fps")) for x in m.get("fps_history") or []]}]
        for k,x in enumerate((m.get("instances") or [])+(m.get("retired_instances") or [])):
            series.append({"name":f'Instance #{x.get("instance")}'+(" (stopped)" if x.get("retired") else ""),"color":C[k%len(C)],
                           "points":[(h.get("elapsed_seconds"),h.get("fps")) for h in x.get("fps_history") or []]})
        target=m.get("target_fps") or None
        spec=make_spec(f'{m.get("name","Model")} FPS: Total vs Target, per Instance',"FPS",series,target=target,
                       target_label=f"Target {target:.1f} FPS" if target else "",
                       note="Total = sum of all instances (rolling 5 s). Each instance is paced to an equal share of the target; an instance below its share is running flat out.")
        if spec:specs.append(spec)
    return specs
def aux_rows(p):
    """(label, snapshot, target, target_kind) for the image workloads."""
    out=[]
    for x in p.get("image_persistence") or []:
        out.append((f'Persistence: {x.get("model")}',x,None,"model FPS"))
    return out
def aux_html(p):
    rows_=aux_rows(p)
    cards=[encode_html(p.get("encode_test"))] if p.get("encode_test") else []
    if not rows_ and not cards:return '<p class="muted">The encode test and image persistence were disabled for this session.</p>'
    for label,x,target,kind in rows_:
        if kind=="model FPS":
            ratio=x.get("keep_up_ratio")
            verdict=("Kept up with the model" if ratio and ratio>=0.98 else "Fell behind the model" if ratio else "No saves")
            tone="ok" if ratio and ratio>=0.98 else "bad"
            head=f'{_n(x.get("fps"))} saves/s for {_n(x.get("requested_fps_avg"))} inferences/s · {_pct(ratio)} kept up'
            where=f'{x.get("width")}×{x.get("height")} {str(x.get("format","")).upper()}, {(x.get("image_bytes") or 0)/1024:,.0f} KiB, fsync {"on" if x.get("fsync") else "off"} → {x.get("image_path","")}'
        else:
            met=bool(target) and (x.get("fps") or 0)>=target*0.98
            verdict="Target met" if met else ("Below target" if target else "Unpaced (max throughput)")
            tone="ok" if met or not target else "bad"
            head=f'{_n(x.get("fps"))} FPS avg' + (f' of {_n(target)} target' if target else '')
            where=f'{x.get("width")}×{x.get("height")} {str(x.get("format","")).upper()} q{x.get("quality")} · {x.get("backend","")}'
        if x.get("state")=="failed":verdict,tone=f'Failed: {x.get("error","")}',"bad"
        ops=f"{int(x.get('operations') or 0):,}"
        cards.append(f'''<div class="ai-model"><div class="ai-model-head"><h3>{e(label)}</h3><span class="pill {tone}">{e(verdict)}</span></div>
          <div class="cards">
            <div class="metric"><span>Throughput</span><strong>{e(head)}</strong><small>{ops} saves · {int(x.get("dropped_requests") or 0):,} dropped · {_n(x.get("throughput_mb_per_s"))} MB/s</small></div>
            <div class="metric"><span>Latency (mean)</span><strong>{_n(x.get("latency_mean_ms"),3)} ms</strong><small>p50 {_n(x.get("latency_p50_ms"),3)} · p95 {_n(x.get("latency_p95_ms"),3)} · p99 {_n(x.get("latency_p99_ms"),3)} · max {_n(x.get("latency_max_ms"),3)} ms</small></div>
            <div class="metric"><span>Capacity</span><strong>{_n(x.get("capacity_fps"),1)} FPS</strong><small>1000 / mean latency</small></div>
          </div><p class="muted">{e(where)}</p></div>''')
    return ''.join(cards)
def obs_note_html(o):
    if not o:return ""
    st=o.get("obs_version_stats") or {}
    msg=(f'Streams were received by OBS Studio (DistroAV) in scene "{o.get("scene")}", {o.get("sources",0)} NDI source(s). '
         f'Per-stream FPS is OBS Source Profiler "async input" (frames DistroAV delivered to OBS, averaged by OBS over ~5 s); frame counts are estimated from it. '
         f'OBS render {st.get("activeFps","N/A")} FPS, skipped render frames {st.get("renderSkippedFrames","N/A")}/{st.get("renderTotalFrames","N/A")}, OBS CPU {st.get("cpuUsage","N/A")}%.')
    if o.get("error"):msg+=f' Error: {o["error"]}'
    if o.get("profiler_error"):msg+=f' Source Profiler: {o["profiler_error"]}'
    return f'<p class="muted">{e(msg)}</p>'
def _pct(r):return "N/A" if r is None else f"{r*100:.1f}%"
def encode_html(enc):
    streams=enc.get("streams") or []
    n=enc.get("stream_count") or len(streams);t=enc.get("target_fps") or 30
    ok=[s for s in streams if (s.get("keep_up_ratio") or 0)>=0.98]
    failed=[s for s in streams if s.get("state")=="failed"]
    verdict,tone=(("Error","bad") if failed or enc.get("error") else ("All streams kept up","ok") if len(ok)==n else (f"{len(ok)} of {n} streams kept up","bad"))
    rows_="".join(f'<tr><td>#{i+1}</td><td>{e(s.get("encoder",""))}</td><td>{_n(s.get("fps"))}</td><td>{_pct(s.get("keep_up_ratio"))}</td>'
                  f'<td>{int(s.get("frames_encoded") or 0):,}</td><td>{int(s.get("frames_dropped") or 0):,}</td><td>{_n(s.get("bitrate_mbps"),1)}</td>'
                  f'<td>{_n(s.get("write_mean_ms"),2)}</td><td>{_n(s.get("write_p95_ms"),2)}</td><td>{_n(s.get("write_max_ms"),2)}</td>'
                  f'<td>{e(s.get("fallback_reason") or s.get("error") or "")}</td></tr>' for i,s in enumerate(streams))
    enc_label=("GPU: "+enc["gpu_encoder"]) if enc.get("gpu_encoder") else "CPU (libx264): no working GPU encoder found"
    return f'''<div class="ai-model"><div class="ai-model-head"><h3>Encode test: {e(n)} x {e(enc.get("resolution",""))}</h3><span class="pill {tone}">{e(verdict)}</span></div>
      <div class="cards">
        <div class="metric"><span>Total encoded</span><strong>{_n(enc.get("total_fps"))} FPS</strong><small>target {n} x {t} = {n*t} FPS</small></div>
        <div class="metric"><span>Streams keeping up</span><strong>{len(ok)} / {n}</strong><small>&ge; 98% of frames due at {t} FPS</small></div>
        <div class="metric"><span>Encoder</span><strong>{e(enc_label)}</strong><small>H.264, one encoder process per stream; feeders {"in one shared process" if enc.get("worker_mode","shared")=="shared" else "one process each"}{f', CPU encoder capped at {enc["x264_threads_per_stream"]} thread(s)/stream' if not enc.get("gpu_encoder") and enc.get("x264_threads_per_stream") else ""}</small></div>
      </div>
      {f'<p class="muted">{e(enc["error"])}</p>' if enc.get("error") else ""}
      <div class="table-wrap"><table class="data"><thead><tr><th>Stream</th><th>Encoder</th><th>Encoded FPS</th><th>Kept up</th><th>Frames encoded</th><th>Dropped</th><th>Bitrate Mbps</th><th>Hand-off mean ms</th><th>Hand-off p95 ms</th><th>Hand-off max ms</th><th>Note</th></tr></thead><tbody>{rows_}</tbody></table></div>
      <details><summary>GPU encoder detection</summary><p class="muted">{e(" · ".join(enc.get("probe") or []))}</p></details></div>'''
def aux_specs(p):
    specs=[]
    enc=p.get("encode_test")
    if enc and enc.get("streams"):
        t=enc.get("target_fps") or 30
        sp=make_spec(f'Encode Streams FPS Over Time ({enc.get("stream_count")} x {enc.get("resolution")}, rolling 5 s)',"FPS",
                     [{"name":f'Stream #{i+1} ({s.get("encoder","")})',"points":[(h.get("elapsed_seconds"),h.get("fps")) for h in s.get("fps_history") or []]}
                      for i,s in enumerate(enc["streams"])],target=t,target_label=f"Target {t} FPS per stream")
        if sp:specs.append(sp)
    for x in p.get("image_persistence") or []:
        h=x.get("fps_history") or []
        sp=make_spec(f'Image Persistence FPS vs Model FPS: {x.get("model")}',"FPS",
                     [{"name":"Saved images/s","color":"#6ce0b0","points":[(a.get("elapsed_seconds"),a.get("fps")) for a in h]},
                      {"name":"Model FPS (requested)","color":"#ffca63","points":[(a.get("elapsed_seconds"),a.get("requested_fps")) for a in h if a.get("requested_fps") is not None]}])
        if sp:specs.append(sp)
    return specs
def telemetry_specs(samples):
    g=2**30;out=[chart("CPU Utilization (system-wide)","%",samples,[("CPU",["cpu_percent"],1)]),chart("Python Process Utilization (total)","%",samples,[("All benchmark processes",["tool_cpu_percent"],1)]),chart("CPU Frequency","MHz",samples,[("Clock",["cpu_frequency_mhz"],1)]),chart("CPU Temperature","°C",samples,[("CPU",["cpu_temperature_c"],1)]),chart("RAM Utilization","%",samples,[("RAM",["ram_percent"],1)]),chart("Memory","GiB",samples,[("RAM",["ram_used_bytes"],g),("This tool (all processes)",["tool_rss_bytes"],g),("Engine process only",["process_rss_bytes"],g),("VMS",["process_vms_bytes"],g),("Swap",["swap_used_bytes"],g)]),chart("Memory Bus Speed","MT/s",samples,[("Configured bus speed",["memory_bus_speed_mhz"],1)]),chart("Tool Process Count","",samples,[("Processes",["tool_process_count"],1)]),chart("Network","Mbps",samples,[("TX",["network_tx_mbps"],1),("RX",["network_rx_mbps"],1)]),chart("Storage","MB/s",samples,[("Read",["disk_read_mbps"],1),("Write",["disk_write_mbps"],1)]),chart("Disk Capacity","GiB",samples,[("Used",["disk_used_bytes"],g),("Free",["disk_free_bytes"],g)])]
    process_names=sorted({p.get("name") for s in samples for p in s.get("processes",[]) if p.get("name")})
    for name in process_names:
        key="process_"+str(abs(hash(name)))
        normalized=[]
        for sample in samples:
            match=next((p for p in sample.get("processes",[]) if p.get("name")==name),{})
            normalized.append({**sample,key:match.get("cpu_percent"),key+"_rss":match.get("rss_bytes")})
        out.append(chart(f"Process CPU — {name}","%",normalized,[(name,[key],1)]))
        out.append(chart(f"Process Memory — {name}","GiB",normalized,[(name,[key+"_rss"],g)]))
    n=max((len(s.get("gpus",[])) for s in samples),default=0)
    if n:
        # Combined view across every GPU first (one glance number), then each
        # GPU's own dedicated breakdown below.
        gpu_names=sorted({g.get("name") for s in samples for g in s.get("gpus",[]) if g.get("name")})
        gpu_label=", ".join(gpu_names) or "Detected GPUs"
        out += [chart(f"Combined GPU Utilization ({gpu_label})","%",samples,[("Average",["gpu_combined","gpu_percent_avg"],1),("Peak",["gpu_combined","gpu_percent_max"],1)]),
                chart(f"Combined GPU VRAM ({gpu_label})","%",samples,[("Average",["gpu_combined","vram_percent_avg"],1),("Peak",["gpu_combined","vram_percent_max"],1)]),
                chart(f"Combined GPU Power ({gpu_label})","W",samples,[("Total",["gpu_combined","power_watts_total"],1)]),
                chart(f"Peak GPU Temperature ({gpu_label})","°C",samples,[("Peak",["gpu_combined","temperature_c_max"],1)])]
    for i in range(n):
        name=next((g.get("name") for s in samples for g in s.get("gpus",[]) if g.get("index")==i and g.get("name")),f"GPU {i}")
        out += [chart(f"{name} Utilization","%",samples,[("Compute",["gpus",i,"gpu_percent"],1),("Memory",["gpus",i,"memory_controller_percent"],1),("Encoder",["gpus",i,"encoder_percent"],1),("Decoder",["gpus",i,"decoder_percent"],1)]),chart(f"{name} VRAM","%",samples,[("VRAM",["gpus",i,"vram_percent"],1)]),chart(f"{name} Thermal/Fan","°C/%",samples,[("Temperature",["gpus",i,"temperature_c"],1),("Fan",["gpus",i,"fan_percent"],1)]),chart(f"{name} Power","W",samples,[("Power",["gpus",i,"power_watts"],1),("Limit",["gpus",i,"power_limit_watts"],1)]),chart(f"{name} Clocks","MHz",samples,[("Graphics",["gpus",i,"graphics_clock_mhz"],1),("Memory",["gpus",i,"memory_clock_mhz"],1)]),chart(f"{name} PCIe","MB/s",samples,[("TX",["gpus",i,"pcie_tx_mbps"],1),("RX",["gpus",i,"pcie_rx_mbps"],1)])]
    return [x for x in out if x]
def fitted_copy(spec,title):
    """PDF has no 'fit Y' button, so the stream FPS chart gets a second page
    with the Y axis fitted to the data, where small deviations are visible."""
    if not spec:return None
    ys=sorted(y for s in spec["series"] for _,y in s["points"])
    lo=ys[int(len(ys)*0.005)];hi=ys[min(len(ys)-1,int(len(ys)*0.995))]
    if spec.get("target") is not None:lo=min(lo,spec["target"]);hi=max(hi,spec["target"])
    pad=max((hi-lo)*0.15,0.05*max(1.0,abs(hi)));lo-=pad;hi+=pad
    return dict(spec,title=title,y_from_zero=False,y_range=[lo,hi],pdf_note="Same data as the previous page with the Y axis zoomed to the central 99% of readings (plus the target), so small deviations are visible. Excursions outside this range, such as the drops on the previous page, run off the plot edge.")
def write(out,p):
    a=analyze(p);
    p["bottleneck_analysis"]=a;
    s=p["run_summary"];
    samples=p.get("telemetry",[])
    inventory_gpus=p.get("inventory",{}).get("gpus",[])
    for sample in samples:
        for index,gpu in enumerate(sample.get("gpus",[])):
            if not gpu.get("name"):
                inventory_gpu=next((item for item in inventory_gpus if item.get("index", index)==index), None)
                gpu["name"]=(inventory_gpu or {}).get("name") or f"GPU {index}"

    stream_results = p.get("streams", [])
    config = p.get("config", {})
    fps_target = float(config.get("targets", {}).get("stream_fps_min", 30))

    fps_bar_html = obs_note_html(p.get("obs_ndi")) + fps_bar_chart(stream_results, fps_target)
    fps_spec = fps_history_spec(stream_results, fps_target)
    fps_min = fps_min_spec(stream_results, fps_target)
    ai_specs = model_fps_specs(p.get("ai_models", []))
    image_specs = aux_specs(p)
    telemetry_chart_specs = telemetry_specs(samples)
    fps_history_html = (html_chart(fps_spec, height=380) + html_chart(fps_min, height=300)) or '<p class="muted">No FPS history was recorded.</p>'

    findings=''.join(
        f"""
        <div class="finding">
            <b>{e(x["subsystem"])}</b>
            <p>{e(x["evidence"])}</p>
            <p>{e(x["recommendation"])}</p>
        </div>"""
        for x in a["observations"]) or """
            <p class="muted">No dominant threshold violation detected.</p>
        """

    streams=[]

    for st in p.get("streams",[]):
        width = st.get("width")
        height = st.get("height")
        resolution = f"{width} x {height}" if width and height else "Not recorded"

        streams.append(
            f"""
            <details class="stream">
                <summary>
                    <span>
                        { e(st["stream_id"]) }
                    </span>
                    <small>
                        { e(st.get("transport","")) }
                        · {fmt(st.get("fps"),"fps")}
                        · {st.get("frames",0)} frames
                        · {e(resolution)}
                    </small>
                </summary>

                <div class="stream-body">
                    {
                        cards(st,[("frames","Received"),
                                  ("instant_fps", "FPS (live)"),
                                  ("fps_mean", "Mean FPS"),
                                  ("fps_min", "Minimum FPS"),
                                  ("fps_max", "Maximum FPS"),
                                  ("fps","FPS (avg)"),
                                  ("dropped","Dropped")])
                    }

                    <div class="resolution-summary">
                        <div class="resolution-card">
                            <span>Stream Resolution</span>
                            <strong>{ e(resolution) }</strong>
                        </div>
                    </div>

                    <h3>All Stream Data</h3>
                    <div class="rows">
                        { rows(st, exclude=("fps_history",)) }
                    </div>
                </div>
            </details>""")

    hero={"status":"Attention Required" if a["observations"] else "No Dominant Bottleneck","primary":a["primary_bottleneck"],"duration":fmt_duration(s.get("actual_duration_seconds")),"streams":s.get("streams")}
    body=f'''
    <!doctype html>
    <meta charset="utf-8">
    <style>*{{box-sizing:border-box}}
    
        body
        {{margin:0;
        background:#06101c;
        color:#edf5ff;
        font:14px Segoe UI}}
        
        main
        {{max-width:1320px;
        margin:auto;
        padding:30px}}
        
        header,section,.stream,.chart
        {{background:#0d1d30;
        border:1px solid #29425e;
        border-radius:14px;
        padding:20px;
        margin:15px 0}}
        
        header
        {{background:linear-gradient(135deg,#173e65,#0b192a)}}
        
        h1{{font-size:40px}}
        h2{{color:#65d6ff}}

        .cards
        {{display:grid;
        grid-template-columns:repeat(4,minmax(0,1fr));
        gap:10px}}
        
        .metric
        {{background:#081625;
        border:1px solid #29425e;
        border-radius:9px;
        padding:12px;
        min-width:0}}
        
        .metric span
        {{display:block;
        color:#91a7bd;
        font-size:10px;
        text-transform:uppercase}}
        
        .metric strong
        {{display:block;
        font-size:18px;
        overflow-wrap:anywhere}}

        .metric small
        {{display:block;
        margin-top:6px;
        color:#91a7bd;
        font-size:11px}}
        .summary-block{{margin:12px 0}}
        .summary-block h3{{margin-bottom:6px}}
        .ai-model{{border:1px solid #29425e;border-radius:12px;padding:14px;margin:12px 0}}
        .ai-model-head{{display:flex;justify-content:space-between;align-items:center}}
        .pill{{padding:3px 10px;border-radius:99px;font-size:12px;background:#203b55}}
        .pill.ok{{background:#1f5a44;color:#bff5dd}}.pill.warn{{background:#5c4a17;color:#ffe6a6}}.pill.bad{{background:#5f2530;color:#ffd0d6}}
        .table-wrap{{overflow-x:auto}}
        table.data{{width:100%;border-collapse:collapse;font-size:12px;margin-top:8px}}
        table.data th,table.data td{{text-align:left;padding:5px 8px;border-bottom:1px solid #29425e;white-space:nowrap}}
        table.data th{{color:#91a7bd;font-weight:600}}
        table.data tr.retired{{opacity:.5}}
        .summary-block h4{{margin:10px 0 6px;color:#b7c9db;font-size:13px}}
        .gpu-summary{{border-left:3px solid #48b8ff;padding-left:14px;margin-top:22px}}
        .gpu-summary h3 small{{color:#91a7bd;font-weight:400;font-size:12px;margin-left:6px}}
        .summary-row{{display:flex;flex-wrap:wrap;gap:8px}}
        .summary-metric{{flex:1 1 130px;min-width:120px;background:#081625;border:1px solid #29425e;border-radius:9px;padding:10px}}
        .summary-metric span{{display:block;color:#91a7bd;font-size:10px;text-transform:uppercase}}
        .summary-metric strong{{display:block;font-size:16px;margin-top:3px}}
        .model-meter{{height:7px;background:#14283b;border-radius:5px;margin-top:9px;overflow:hidden}}
        .model-meter i{{display:block;height:100%;background:#65dda9;border-radius:5px}}

        .usb-list
        {{list-style:none;
        margin:10px 0 0;
        padding:0;
        display:grid;
        gap:6px}}

        .usb-list li
        {{background:#081625;
        border:1px solid #29425e;
        border-radius:8px;
        padding:9px 13px;
        font-size:12px}}
        
        .charts{{display:grid;
        grid-template-columns:1fr 1fr;
        gap:12px}}
        
        .chart
        {{margin:0;
        background:#081625}}

        .chart-toolbar{{display:flex;align-items:center;gap:6px;margin:6px 0 10px}}
        .chart-toolbar button{{padding:4px 9px;background:#203b55;color:#fff;border:1px solid #41617e;border-radius:5px;cursor:pointer}}
        .chart-toolbar span{{color:#91a7bd;font-size:11px;margin-left:5px}}
        .grid{{stroke:#29425e;stroke-width:1;opacity:.7}}
        .series{{cursor:pointer;transition:opacity .15s,stroke-width .15s}}
        .point{{cursor:crosshair;opacity:.05;transition:opacity .15s}}
        .point:hover{{opacity:1;r:5}}
        .series:hover,.series.selected{{stroke-width:5;opacity:1}}
        .interactive-chart .series:not(.selected){{opacity:.85}}
        
        svg
        {{width:100%;
        height:225px}}
        
        svg text
        {{fill:#91a7bd}}
        
        .legend
        {{color:#91a7bd}}
        
        /* Per-stream FPS bars */

        .fps-bars {{
        display: grid;
        gap: 14px;
        margin-top: 18px;
        }}

        .fps-row {{
        padding: 12px;
        background: #081625;
        border: 1px solid #29425e;
        border-radius: 10px;
        }}

        .fps-label {{
        display: flex;
        justify-content: space-between;
        gap: 14px;
        margin-bottom: 7px;
        }}

        .fps-label strong {{
        min-width: 0;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
        }}

        .fps-label span {{
        flex-shrink: 0;
        color: #65d6ff;
        font-family: Consolas, monospace;
        font-weight: 700;
        }}

        .fps-track {{
        position: relative;
        height: 16px;
        overflow: visible;
        background: #14283b;
        border: 1px solid #35516b;
        border-radius: 999px;
        }}

        .fps-fill {{
        height: 100%;
        border-radius: 999px;
        }}

        .fps-track i {{
        position: absolute;
        top: -5px;
        bottom: -5px;
        width: 2px;
        background: #edf5ff;
        opacity: 0.9;
        }}

        .fps-row small {{
        display: block;
        margin-top: 7px;
        color: #91a7bd;
        }}

        .fps-target-description {{
        padding: 12px 14px;
        margin-bottom: 15px;
        color: #b7c9db;
        background: #081625;
        border-left: 4px solid #65d6ff;
        border-radius: 7px;
        }}

        /* Resolution */

        .resolution-summary {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 14px;
        align-items: center;
        margin: 18px 0;
        }}

        .resolution-card {{
        padding: 14px;
        background: #081625;
        border: 1px solid #29425e;
        border-radius: 10px;
        }}

        .resolution-card span {{
        display: block;
        color: #91a7bd;
        font-size: 10px;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        }}

        .resolution-card strong {{
        display: block;
        margin-top: 5px;
        font-size: 18px;
        }}


        .stream
        {{padding:0;
        overflow:hidden}}
        
        .stream summary
        {{cursor:pointer;
        padding:17px 20px;
        display:flex;
        justify-content:space-between;
        background:#10243a}}
        
        .stream summary span
        {{font-size:18px;
        font-weight:700}}
        
        .stream summary small
        {{color:#91a7bd}}
        
        .stream-body
        {{padding:20px}}
        
        .rows
        {{display:grid;
        grid-template-columns:1fr}}
        
        .row
        {{display:grid;
        grid-template-columns:260px 1fr;
        border-bottom:1px solid #29425e;
        padding:9px}}
        
        .row>span
        {{color:#91a7bd}}
        
        pre
        {{margin:0;
        white-space:pre-wrap;
        word-break:break-word}}
        
        .clamp
        {{max-height:90px;
        overflow:hidden}}
        
        .clamp.expanded
        {{max-height:none}}
        
        button.more
        {{background:#203b55;
        color:white;
        border:1px solid #41617e;
        border-radius:7px;
        padding:6px;
        margin-top:5px}}
        
        .finding
        {{border-left:4px solid #ffca63;
        padding:10px 15px;
        background:#081625;
        margin:10px 0}}
        
        .muted
        {{color:#91a7bd}}
        
        #topButton
        {{position:fixed;
        right:22px;
        bottom:22px;
        width:48px;
        height:48px;
        border:0;
        border-radius:50%;
        background:#48b8ff;
        color:#04111c;
        font-size:22px;
        font-weight:bold;
        box-shadow:0 8px 28px #0008;
        cursor:pointer;
        z-index:99}}

        {CHART_CSS}
        @media(max-width:850px)
        {{.cards,.charts{{grid-template-columns:1fr 1fr}}.row{{grid-template-columns:1fr}}}}
        @media(max-width:520px)
        {{.cards,.charts{{grid-template-columns:1fr}}}}
    </style>
    
    <main id="top">
        <header>
            <p>VX3 NDI STREAM QUALIFICATION</p>
            <h1>{e(s.get("scenario","Benchmark"))}</h1>
            {
                cards(hero,[("status","Status"),
                            ("primary","Primary Bottleneck"),
                            ("duration","Duration"),
                            ("streams","Streams")])
            }
        </header>
        
        <section>
            <h2>Bottleneck Analysis and Summary</h2>
            <p><b>{e(a["summary"])}</b></p>
            <p>Confidence: {e(a["confidence"])}</p>
            {findings}
        </section>
        
        <section>
            <h2>Hardware and Software Inventory</h2>
            {system_info(p.get("inventory",{}))}
        </section>
        
        <section>
            <h2>Benchmark Configuration</h2>
            <div class="clamp">
                <pre>{e(json.dumps(p.get("config",{}),indent=2))}</pre>
            </div>
            <button class="more">Show More</button>
        </section>

        <section>
            <h2>Independent AI Model Workload</h2>
            <p class="muted">Each model's target is the configured AI FPS per stream multiplied by the number of AI-enabled input streams. The benchmark starts one instance (OS process) per model, paces every instance to an equal share of the target, and adds instances one at a time until the total sustains the target -- so the reported instance count is the minimum that meets it. The search ends when the target is met; when an added instance raises the measured total by less than the configured minimum gain (always enforced for every added instance; all started instances keep running, and the result is "target met" or "saturated" depending on the total); when the configured maximum number of processes per model is reached; or when a new instance fails to start. The table of totals at each instance count shows where throughput stopped scaling. Each instance runs in its own OS process with its own ONNX Runtime session and repeatedly infers the same in-memory image. It is detached from NDI capture and does not consume stream frames. Only the inference call is timed; on GPU the input and outputs stay on the device (IOBinding) and results are synchronized inside the timed region. Warm-up inferences are excluded (the first one is reported as cold-start latency). FPS is completed inferences divided by the measurement window from the end of warm-up to the last inference, excluding the tool's own publishing time. Capacity FPS is 1000 / mean latency, the ceiling for an instance with no pacing.</p>
            {model_summary_html(p.get("ai_models", []))}
            <div class="charts" style="grid-template-columns:1fr">{''.join(html_chart(x) for x in ai_specs)}</div>
            {''.join(f'<div class="model-result"><h3>{e(model.get("name", "Model"))} details</h3><div class="rows">{rows(model, exclude=("name", "path", "input_shape", "fps_history", "instances", "retired_instances", "scaling_events"))}</div></div>' for model in p.get("ai_models", []))}
        </section>

        <section>
            <h2>Encode Test &amp; Image Persistence</h2>
            <p class="muted">The encode test runs N live H.264 encode streams (1080p30 or 4K30), each in its own process, fed with moving YUV frames at exactly 30 FPS; it uses a hardware encoder (NVIDIA NVENC, Intel Quick Sync or AMD AMF) when one works on this machine, otherwise the CPU (libx264). Encoded FPS comes from the encoder's own frame counter (2 s warm-up excluded); a frame slot the encoder could not accept in time is dropped, as with a live camera; hand-off time is the time to pass one raw frame to the encoder, which grows when it back-pressures. Image persistence runs one process per enabled AI model that repeatedly saves the same pre-encoded image to the same file, driven by that model's measured inference rate (one save request per inference, queued so a slow save is caught up afterwards; requests older than 2 s are dropped and counted); with fsync on, every save is forced to the storage device. Only open, write, flush, fsync and close are timed; percentiles are exact to 1 µs.</p>
            {aux_html(p)}
            <div class="charts" style="grid-template-columns:1fr">{''.join(html_chart(x) for x in image_specs)}</div>
        </section>
        
        <section>
            <h2>Per-Stream FPS Summary</h2>
            <div class="fps-target-description">
                The configured stream FPS target is
                <strong>{fps_target:.2f} FPS</strong>.
                Green bars meet or exceed the target,
                amber bars are within 10 percent of the
                target, and red bars are below that range.
            </div>
            {fps_bar_html}
        </section>
        
        <section>
            <h2>FPS Over Time</h2>
            <p class="muted">Every stream across the complete session on one shared time axis. Click legend entries to hide/show streams (Ctrl/Alt-click to isolate one); hover a stream in the legend to highlight it.</p>
            {fps_history_html}
        </section>


        <h2>Per-Stream Results</h2>
        <p class="muted">Streams are collapsed by default. Select a stream to expand its full data.</p>
        {''.join(streams)}
        
        <section>
            <h2>Telemetry Summary</h2>
            <p class="muted">Aggregates are calculated from recorded samples. Missing readings are excluded rather than treated as zero.</p>
            {telemetry_summary_html(p.get("telemetry_summary",{}))}
        </section>
        
        <h2>Full-Session Telemetry Graphs</h2>
        <p class="muted">"CPU Utilization (system-wide)" is everything running on the machine; "CPU Used By This Tool" is only this benchmark's own processes (the engine plus every stream process it spawned). If the two track closely, system load is genuinely this workload. If system-wide runs noticeably higher than the tool's own share — for example when running over Remote Desktop — something else (often RDP's own screen capture/encoding) is consuming the difference; disconnecting the Remote Desktop session (not closing/logging off) after starting a run lets it keep going in the background without that overhead, since the benchmark runs as an independent process regardless of whether the browser is being viewed.</p>
        <div class="charts">
            {''.join(html_chart(x) for x in telemetry_chart_specs)}
        </div>
        
        <section>
            <h2>Raw Engineering Data</h2>
            <p>All raw samples and measurements are preserved in results.json.</p>
        </section>
    </main>
    
    <button id="topButton" title="Return to top" aria-label="Return to top">↑</button>
    
    <script>{CHART_JS}</script>
    <script>
        document.querySelectorAll('.interactive-chart').forEach(chart=>{{
            const svg=chart.querySelector('svg');
            const vb=svg.getAttribute('viewBox').split(/\\s+/).map(Number);
            const original={{x:vb[0],y:vb[1],w:vb[2],h:vb[3]}};
            let view={{...original}}, drag=null;
            const render=()=>svg.setAttribute('viewBox',`${{view.x}} ${{view.y}} ${{view.w}} ${{view.h}}`);
            const zoom=f=>{{const cx=view.x+view.w/2,cy=view.y+view.h/2;view={{x:cx-view.w*f/2,y:cy-view.h*f/2,w:view.w*f,h:view.h*f}};render()}};
            chart.querySelectorAll('[data-zoom]').forEach(b=>b.onclick=()=>zoom(b.dataset.zoom==='1'?.8:1.25));
            chart.querySelector('[data-reset]').onclick=()=>{{view={{...original}};render()}};
            svg.addEventListener('wheel',ev=>{{ev.preventDefault();zoom(ev.deltaY<0?.8:1.25)}},{{passive:false}});
            svg.addEventListener('pointerdown',ev=>{{drag={{x:ev.clientX,y:ev.clientY,view:{{...view}}}};svg.setPointerCapture(ev.pointerId)}});
            svg.addEventListener('pointermove',ev=>{{if(!drag)return;const r=svg.getBoundingClientRect();view.x=drag.view.x-(ev.clientX-drag.x)/r.width*drag.view.w;view.y=drag.view.y-(ev.clientY-drag.y)/r.height*drag.view.h;render()}});
            svg.addEventListener('pointerup',()=>drag=null);
            svg.querySelectorAll('.series').forEach(line=>line.onclick=()=>{{svg.querySelectorAll('.series').forEach(x=>x.classList.remove('selected'));line.classList.add('selected')}});
        }});
        document.querySelectorAll('.more').forEach(b=>b.onclick=()=>
        {{let x=b.previousElementSibling;x.classList.toggle('expanded');b.textContent=x.classList.contains('expanded')?'Show Less':'Show More'}});
        document.getElementById('topButton').onclick=()=>window.scrollTo({{top:0,behavior:'smooth'}});
    </script>'''
    out=Path(out);
    out.mkdir(parents=True,exist_ok=True);
    (out/"report.html").write_text(body,encoding="utf-8");
    (out/"results.json").write_text(json.dumps(p,indent=2),encoding="utf-8")
    # PDF companion: same content, vector (losslessly zoomable) charts with a
    # toggleable layer per series, plus the interactive HTML, results.json and
    # CSV data embedded as attachments. Optional: a missing dependency or a
    # rendering problem must never cost the operator the HTML report.
    try:
        from .report_pdf import write_pdf
        write_pdf(out/"report.pdf", p, fps_target=fps_target,
                  chart_sections=[("Stream FPS", [x for x in (fps_spec, fitted_copy(fps_spec, "Per-Stream FPS Over Time (Y axis fitted to data)"), fps_min) if x]),
                                  ("AI Models", ai_specs),
                                  ("Encode Test & Image Persistence", image_specs),
                                  ("Telemetry", telemetry_chart_specs)],
                  attachments={"report.html": out/"report.html", "results.json": out/"results.json"})
        print(f"[report] PDF written: {out/'report.pdf'}", flush=True)
    except ImportError as exc:
        print(f"[report] PDF skipped -- install reportlab and pypdf ({exc})", flush=True)
    except Exception as exc:
        import traceback
        print(f"[report] PDF generation failed: {exc}", flush=True)
        traceback.print_exc()
