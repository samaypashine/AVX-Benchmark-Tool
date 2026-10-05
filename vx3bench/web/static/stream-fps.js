/* Add <section class="panel"><h2>Live Stream FPS</h2><div id="streamFps"></div></section>
 * to the right panel, then call renderStreamFps(s.stream_metrics) from status(). */
function renderStreamFps(payload) {
  const streams = payload?.streams || [];
  const target = Number(cur?.targets?.stream_fps_min || 30);
  const max = Math.max(target, ...streams.map(s => Number(s.instant_fps || 0)), 1) * 1.1;
  document.getElementById('streamFps').innerHTML = streams.length ? streams.map(s => {
    const fps = Number(s.instant_fps || 0);
    const width = Math.min(100, fps / max * 100);
    const targetPosition = Math.min(100, target / max * 100);
    const color = fps >= target ? '#6ce0b0' : fps >= target * .9 ? '#ffca63' : '#ff7d8a';

    return `<div class="fps-row">
      <div class="fps-label"><strong>${esc(s.stream_id)}</strong><span>${fps.toFixed(2)} FPS</span></div>
      <div class="fps-track"><div class="fps-fill" style="width:${width}%;background:${color}"></div><i style="left:${targetPosition}%"></i></div>
      <small>${s.width || '?'}×${s.height || '?'} · ${s.frames_estimated ? '~' + (s.frames || 0) + ' frames (estimated)' : (s.frames || 0) + ' unique frames'}${s.dropped ? ` · ${s.dropped} dropped` : ''}${s.emulated_camera ? ' · <b style="color:#ffd57a">emulated camera</b>' : ''}${s.transport === 'synthetic' && s.obs_input_name ? ' · <b style="color:#ffd57a">synthetic</b>' : ''}${s.obs_input_name ? ` · via OBS (${s.obs_input_name}) · rendered ${Number(s.obs_rendered_fps || 0).toFixed(2)} FPS · render ${Number(s.obs_render_avg_ms || 0).toFixed(2)} ms` : ''}</small>
    </div>`;
  }).join('') : '<p class="muted">Waiting for stream frames...</p>';
}
