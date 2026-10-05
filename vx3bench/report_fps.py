"""Per-stream FPS visualization helpers for the qualification report."""
import html
from .report_charts import color, make_spec


def e(x): return html.escape(str(x))


def _avg_fps(s):
    return float(s.get("fps_mean", s.get("fps", 0)) or 0)


def fps_bar_chart(streams, target=30.0):
    if not streams: return '<p class="muted">No stream FPS data was recorded.</p>'
    maximum = max([target] + [_avg_fps(s) for s in streams]) * 1.1 or 1
    rows = []
    for s in streams:
        fps = _avg_fps(s); pct = min(100, 100 * fps / maximum); target_pct = min(100, 100 * target / maximum)
        c = "#6ce0b0" if fps >= target else "#ffca63" if fps >= target * .9 else "#ff7d8a"
        lo, hi = s.get("fps_min"), s.get("fps_max")
        span = f" · 1 s min {lo:.2f} / max {hi:.2f} FPS" if lo is not None and hi is not None else ""
        if s.get("obs_input_name"):
            span += f" · OBS input {s['obs_input_name']}, rendered {(s.get('obs_rendered_fps') or 0):.2f} FPS"
        rows.append(f'''<div class="fps-row"><div class="fps-label"><strong>{e(s.get('stream_id','stream'))}</strong><span>{fps:.2f} FPS</span></div><div class="fps-track"><div class="fps-fill" style="width:{pct:.2f}%;background:{c}"></div><i style="left:{target_pct:.2f}%" title="Target {target:.2f} FPS"></i></div><small>{s.get('width','?')}×{s.get('height','?')} received{span}</small></div>''')
    return '<div class="fps-bars">' + ''.join(rows) + '</div>'


def fps_history_spec(streams, target=30.0):
    """Whole-session FPS for every stream on one shared time axis. Each point
    is the exact receive rate over one sampling interval (1 s by default)."""
    series = []
    for i, s in enumerate(streams):
        pts = [(x.get("elapsed_seconds"), x.get("fps")) for x in s.get("fps_history", []) or []]
        series.append({"name": str(s.get("stream_id", f"stream {i+1}")), "color": color(i), "points": pts})
    spec = make_spec("Per-Stream FPS Over Time", "FPS", series, target=target,
                     target_label=f"Target {target:.1f} FPS", y_from_zero=True,
                     note="Each point is the exact receive rate over one sampling interval. Hover for values, "
                          "wheel to zoom the time axis, toggle 'Y: fit data' to see small deviations.")
    if spec:
        spec["pdf_note"] = ("Each point is the exact receive rate over one sampling interval. Use the viewer's "
                            "Layers panel to show/hide individual streams; the next page repeats this chart with a zoomed Y axis.")
    return spec


def fps_min_spec(streams, target=30.0):
    """Worst interval in each merged bucket -- only differs from the mean
    curve on long runs where history was compacted."""
    series = []
    for i, s in enumerate(streams):
        hist = s.get("fps_history", []) or []
        if not any(x.get("fps_min") is not None and x.get("fps_min") != x.get("fps") for x in hist):
            continue
        series.append({"name": str(s.get("stream_id")), "color": color(i),
                       "points": [(x.get("elapsed_seconds"), x.get("fps_min")) for x in hist]})
    return make_spec("Per-Stream Worst-Interval FPS (compacted history)", "FPS", series, target=target,
                     target_label=f"Target {target:.1f} FPS") if series else None
