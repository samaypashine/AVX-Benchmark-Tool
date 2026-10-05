const $ = x => document.getElementById(x),
    C = ['#42b7ff', '#65dda9', '#ffc95c', '#ff7684'],
    GiB = 2 ** 30;

let cur, orig, mode = 'form', locked = false;

async function api(p, o) {
    let r = await fetch(p, o),
        d = await r.json();
    if (!r.ok) throw Error(d.error);
    return d
}

const esc = s => String(s).replace(/[&<>"']/g, c => ({
    '&': '&amp;',
    '<': '&lt;',
    '>': '&gt;',
    '"': '&quot;',
    "'": '&#39;'
} [c]));

function setp(o, p, v) {
    let x = o;
    for (let i = 0; i < p.length - 1; i++) x = x[p[i]];
    x[p.at(-1)] = v
}

// -- Parametric form --------------------------------------------------------
// Crucial fields (stream transport, NDI bandwidth) are rendered as
// dropdowns instead of free text, so an operator can't set them to an
// invalid value.

// Display-only label formatting: turns a raw JSON key like
// "discovery_timeout_seconds" into "Discovery Timeout Seconds". The
// underlying key (used for data-p / setp/getp) never changes -- only what
// the operator sees as the field's name.
const LABEL_ACRONYMS = new Set(['id', 'cpu', 'gpu', 'gpus', 'vram', 'ram', 'fps',
    'ndi', 'url', 'os', 'usb', 'pcie', 'mhz', 'ghz', 'mb', 'gb', 'kb', 'tb', 'rss', 'vms']);
function prettyLabel(k) {
    return String(k).split('_').filter(Boolean).map(w =>
        LABEL_ACRONYMS.has(w.toLowerCase()) ? w.toUpperCase() : (w.charAt(0).toUpperCase() + w.slice(1))
    ).join(' ');
}

// Must match model_workers.DEVICE_CHOICES ("gpu" in older configs is
// normalized to "cuda" by ensureModelFields).
// Any GPU choice falls back down the chain (ending with the CPU) if it is not
// installed or not usable on this machine; the card and report say so.
const MODEL_DEVICE_OPTIONS = [
    ['auto', 'Auto (best available: CUDA / ROCm / DirectML / CoreML, else CPU)'],
    ['cuda', 'GPU — CUDA (NVIDIA)'],
    ['tensorrt', 'GPU — TensorRT (NVIDIA)'],
    ['directml', 'GPU — DirectML (Windows, any GPU incl. ARM)'],
    ['coreml', 'CoreML (Apple Silicon)'],
    ['cpu', 'CPU']
];

// Friendlier labels, hints and limits for specific settings. Keys are exact
// paths, or patterns where [^.]+ stands for any model name.
const FIELD_META = [
    [/^ai\.target_fps_per_stream$/, { label: 'AI FPS Target per Stream', min: 0, step: 'any',
        hint: 'Each enabled model must sustain this × the number of AI-enabled streams (e.g. 15 × 13 streams = 195 FPS). 0 = no target: processes run flat out.' }],
    [/^ai\.worker_mode$/, { label: 'AI Worker Mode',
        hint: 'Shared (recommended): all instances of a model run as threads in ONE process, sharing one GPU context so their GPU work can overlap, and splitting the CPU cores between them. Process: one OS process per instance (the GPU time-slices between processes).' }],
    [/^ai\.models\.[^.]+\.worker_mode$/, { label: 'Worker Mode (this model)' }],
    [/^encode_test\.worker_mode$/, { label: 'Feeder Worker Mode',
        hint: 'Shared: one process feeds all streams (threads, one shared set of test frames). Each stream always keeps its own encoder process.' }],
    [/^image_persistence\.worker_mode$/, { label: 'Worker Mode', hint: 'Shared: every model\'s saver runs as a thread in one process.' }],
    [/^ai\.autoscale$/, { label: 'AI Process Scaling' }],
    [/^ai\.autoscale\.enabled$/, { label: 'Auto-scale Processes', hint: 'Add processes one at a time until the target is met.' }],
    [/^ai\.autoscale\.max_instances$/, { label: 'Max Processes per Model', min: 1, step: 1,
        hint: 'Scaling stops here if the target still is not met.' }],
    [/^ai\.autoscale\.settle_seconds$/, { label: 'Settle Time (s)', min: 0, step: 'any',
        hint: 'Steady running time after a process warms up before the total is judged.' }],
    [/^ai\.autoscale\.tolerance_percent$/, { label: 'Target Tolerance (%)', min: 0, step: 'any',
        hint: 'A total within this % of the target counts as met.' }],
    [/^ai\.autoscale\.min_gain_percent$/, { label: 'Min Gain per Added Process (%)', min: 0.1, step: 'any',
        hint: 'Always on. Every added process must raise the measured total FPS by at least this much; if one does not, no more processes are added (all started processes keep running). Must be greater than 0.' }],
    [/^inputs\.\d+\.stream_count$/, { label: 'Number of Streams', min: 1, step: 1,
        hint: 'NDI: how many streams OBS should receive -- each gets its own NDI Source in OBS scene "VX3 NDI", assigned a different source found on the network (shortfall can be filled with synthetic streams). Synthetic: how many test streams; when an NDI input is present they are played inside the same OBS scene.' }],
    [/^inputs\.\d+\.fill_with_synthetic$/, { label: 'Fill Missing NDI Streams with Synthetic',
        hint: 'If fewer NDI sources are found than the number of streams, add synthetic streams to the OBS scene to make up the difference (e.g. 12 real + 8 synthetic = 20).' }],
    [/^inputs\.\d+\.synthetic_format$/, { label: 'Synthetic Stream Format',
        hint: 'Synthetic streams play a looping SpeedHQ clip (the codec full-bandwidth NDI uses, ~100 Mbit/s at 1080p30) in OBS, so OBS decodes them like a real NDI source.' }],
    [/^inputs\.\d+\.source_filter$/, { label: 'Only Use NDI Sources Containing', hint: 'Optional text filter (e.g. a machine name). Empty = any NDI source OBS finds. OBS\'s own NDI outputs are always skipped.' }],
    [/^inputs\.\d+\.discovery_timeout_seconds$/, { label: 'NDI Discovery Wait (s)', min: 1, step: 1, hint: 'How long OBS may search for enough NDI sources before the session starts with the ones found.' }],
    [/^inputs\.\d+\.hardware_acceleration$/, { label: 'Allow NDI Hardware Acceleration (DistroAV)' }],
    [/^inputs\.\d+\.obs_path$/, { label: 'OBS Executable', hint: 'Empty = auto-detect (C:\\Program Files\\obs-studio\\bin\\64bit\\obs64.exe). Needs OBS 30.1+ with DistroAV and the NDI Runtime. Close any running OBS first.' }],
    [/^inputs\.\d+\.websocket_port$/, { label: 'obs-websocket Port', min: 1024, step: 1, hint: 'A one-time password is generated for every session.' }],
    [/^inputs\.\d+\.minimize$/, { label: 'Start OBS Minimized' }],
    [/^inputs\.\d+\.arrange_grid$/, { label: 'Tile Sources in a Grid', hint: 'Like OBS multiview: every source visible on the program scene.' }],
    [/^inputs\.\d+\.keep_obs_open$/, { label: 'Keep OBS Open After the Session' }],
    [/^encode_test$/, { label: 'Encode Test' }],
    [/^encode_test\.enabled$/, { label: 'Enabled', hint: 'Encodes N live H.264 streams, each in its own process. Uses the GPU encoder (NVIDIA NVENC, Intel Quick Sync or AMD AMF) when one works on this machine, otherwise the CPU (libx264).' }],
    [/^encode_test\.resolution$/, { label: 'Stream Format' }],
    [/^encode_test\.stream_count$/, { label: 'Number of Encode Streams', min: 1, step: 1 }],
    [/^image_persistence$/, { label: 'Image Persistence Workload' }],
    [/^image_persistence\.enabled$/, { label: 'Enabled', hint: "One process per enabled AI model, saving the same image to the same file at that model's measured FPS." }],
    [/^image_persistence\.directory$/, { label: 'Directory', hint: 'Where the images are written. Empty = the session output folder. Point it at the disk you want to test.' }],
    [/^image_persistence\.width$/, { label: 'Width (px)', min: 16, step: 1 }],
    [/^image_persistence\.height$/, { label: 'Height (px)', min: 16, step: 1 }],
    [/^image_persistence\.quality$/, { label: 'Quality (1-100)', min: 1, step: 1 }],
    [/^image_persistence\.fsync$/, { label: 'Force to Disk (fsync)', hint: 'On: every save is flushed to the storage device (a true disk test). Off: writes may only reach the OS cache.' }],
    [/^ai\.models\.[^.]+\.instances$/, { label: 'Starting Processes', min: 1, step: 1,
        hint: 'Processes to start with; auto-scaling adds more if needed.' }],
    [/^ai\.models\.[^.]+\.target_fps_per_stream$/, { label: 'AI FPS Target per Stream (this model)', min: 0, step: 'any' }]
];
const fieldMeta = path => (FIELD_META.find(([re]) => re.test(path)) || [null, {}])[1];

function leaf(v, p, k) {
    let path = p.join('.'),
        opts = null,
        meta = fieldMeta(path),
        label = meta.label || prettyLabel(k),
        hint = meta.hint ? `<small class=field-hint>${esc(meta.hint)}</small>` : '';
    if (/^ai\.models\.[^.]+\.path$/.test(path))
        return `<label class=field>${label}<div class=model-path><input data-p="${path}" data-t=string list=modelPaths value="${esc(v)}"><input type=file accept=".onnx" class=modelFile><button type=button class=modelBrowse>Browse</button></div></label>`;
    if (path === 'session.duration.unit') opts = [
        ['seconds', 'seconds'],
        ['minutes', 'minutes'],
        ['hours', 'hours'],
        ['days', 'days']
    ];
    if (/^inputs\.\d+\.transport$/.test(path)) opts = [
        ['synthetic', 'Synthetic (generated test pattern)'],
        ['ndi', 'NDI (received in OBS)']
    ];
    if (/^inputs\.\d+\.bandwidth$/.test(path)) opts = [
        ['highest', 'Highest (full quality, more CPU/network per stream)'],
        ['lowest', 'Lowest (NDI preview quality, lighter per stream)']
    ];
    if (/^ai\.models\.[^.]+\.device$/.test(path)) opts = MODEL_DEVICE_OPTIONS;
    if (/(^ai|^ai\.models\.[^.]+|^encode_test|^image_persistence)\.worker_mode$/.test(path)) opts = [['shared', 'Shared process (threads) — recommended'], ['process', 'One process per worker']];
    if (/^inputs\.\d+\.synthetic_format$/.test(path)) opts = [['720p30', '720p30'], ['1080p30', '1080p30'], ['1080p60', '1080p60'], ['4k30', '4K30']];
    if (/^image_persistence\.format$/.test(path)) opts = [['jpeg', 'JPEG'], ['png', 'PNG'], ['webp', 'WebP']];
    if (/^encode_test\.resolution$/.test(path)) opts = [['1080p30', '1080p30 (1920×1080, 30 FPS)'], ['4k30', '4K30 (3840×2160, 30 FPS)']];
    if (opts) {
        // A value the dropdown doesn't know (hand-edited JSON) is shown as
        // such instead of silently displaying the first option while the
        // config still holds something else.
        if (!opts.some(x => x[0] === v)) opts = [[v, `${v} (unsupported)`], ...opts];
        return `<label class=field>${label}<select data-p="${path}" data-t=string>${opts.map(x => `<option value="${esc(x[0])}" ${v === x[0] ? 'selected' : ''}>${esc(x[1])}</option>`).join('')}</select></label>`;
    }
    if (v === null) return `<label class=field>${label}<input readonly value=null></label>`;
    let t = typeof v;
    const limits = t === 'number' ? `${meta.min != null ? ` min="${meta.min}"` : ''}${meta.step != null ? ` step="${meta.step}"` : ''}` : '';
    return `<label class=field>${label}<input data-p="${path}" data-t="${t}" type="${t === 'boolean' ? 'checkbox' : t === 'number' ? 'number' : 'text'}"${limits} ${t === 'boolean' ? (v ? 'checked' : '') : `value="${esc(v)}"`}>${hint}</label>`
}

function node(v, p, k) {
    let path = p.join('.');
    if (path === 'inputs') return renderInputs(v);
    if (Array.isArray(v)) return `<div class=group><div class=title>${prettyLabel(k)}</div>${v.map((x, i) => `<div class=array>${node(x, [...p, i], k + ' ' + (i + 1))}</div>`).join('')}</div>`;
    if (v && typeof v === 'object') {
        let l = [],
            c = [];
        Object.entries(v).forEach(x => (x[1] && typeof x[1] === 'object' ? c : l).push(x));
        // An empty k means "this object's own container already has a
        // header" (used by renderInputs, whose array-item header replaces
        // the generic group title) -- skip the redundant title.
        let titleHtml = k ? `<div class=title>${fieldMeta(path).label || prettyLabel(k)}</div>` : '';
        return `<div class=group>${titleHtml}<div class=fields>${l.map(([n, x]) => leaf(x, [...p, n], n)).join('')}</div>${c.map(([n, x]) => node(x, [...p, n], n)).join('')}</div>`
    }
    return leaf(v, p, k)
}

// Input streams get their own dynamic add/remove UI instead of the generic
// array rendering, so a scenario can be built up entirely from the WebUI
// without hand-editing JSON.
function renderInputs(list) {
    let items = (list || []).map((item, i) => `
        <div class="array-item">
            <div class="array-item-head">
                <b>${esc(item.id || ('input ' + (i + 1)))} <small>(${item.transport === 'ndi' ? 'NDI · received in OBS' : esc(item.transport || '')})</small></b>
                <button type=button class="removeBtn" data-remove="inputs.${i}">Remove</button>
            </div>
            ${node(item, ['inputs', i], '')}
        </div>`).join('');
    return `<div class=group><div class=title>Input Streams</div>${items || '<p class="muted">No input streams yet.</p>'}
        <button type=button id="addInputBtn">+ Add Input Stream</button></div>`;
}

// Fills in the fields a given input's transport actually needs, without
// overwriting anything already set. Synthetic and NDI inputs share the same
// "stream_count" field name (how many total streams this input represents),
// so switching the transport dropdown never leaves behind a field the
// engine doesn't read.
function ensureTransportFields(item) {
    if (!item) return;
    if (item.stream_count == null) item.stream_count = 1;
    const ndi = { fill_with_synthetic: true, synthetic_format: '1080p30', source_filter: '', discovery_timeout_seconds: 15, bandwidth: 'highest', hardware_acceleration: true,
                  obs_path: '', websocket_port: 4455, minimize: true, arrange_grid: true, keep_obs_open: false };
    const syn = { width: 1920, height: 1080, framerate: 30 };
    // NDI is received by OBS only: settings of the removed in-tool receiver
    // are dropped from older scenarios.
    ['receive_mode', 'color_format', 'streams_per_process', 'fallback_framerate'].forEach(k => delete item[k]);
    if (item.transport === 'ndi') {
        Object.keys(syn).forEach(k => delete item[k]);
        Object.keys(ndi).forEach(k => { if (item[k] == null) item[k] = ndi[k]; });
    } else if (item.transport === 'synthetic') {
        Object.keys(ndi).forEach(k => delete item[k]);
        Object.keys(syn).forEach(k => { if (item[k] == null) item[k] = syn[k]; });
    }
}

// Scenario-level AI throughput target and the instance autoscaler
// (defaults match model_workers.DEFAULT_TARGET_FPS_PER_STREAM / DEFAULT_AUTOSCALE).
function ensureAiFields(config) {
    if (!config || typeof config !== 'object') return;
    config.ai = config.ai || { models: {} };
    if (config.ai.target_fps_per_stream == null) config.ai.target_fps_per_stream = 15;
    if (config.ai.worker_mode == null) config.ai.worker_mode = 'shared';
    const a = config.ai.autoscale = config.ai.autoscale || {};
    const d = { enabled: true, max_instances: 16, settle_seconds: 8, tolerance_percent: 2,
                min_gain_percent: 5 };
    delete a.stop_low_gain_instance;   // removed: the minimum-gain rule always applies and never stops processes
    Object.keys(d).forEach(k => { if (a[k] == null) a[k] = d[k]; });
    if (!(Number(a.min_gain_percent) > 0)) a.min_gain_percent = d.min_gain_percent;   // always enabled
}

function ensureImageWorkloadFields(config) {
    if (!config || typeof config !== 'object') return;
    // Older scenarios kept OBS settings in a separate "obs_ndi" section:
    // fold them into the NDI input (one combined Input Streams section).
    config.inputs = config.inputs || [];
    if (config.obs_ndi) {
        const ndiItem = config.inputs.find(x => x && x.transport === 'ndi');
        if (ndiItem) Object.entries(config.obs_ndi).forEach(([k, v]) => { if (k !== 'enabled' && ndiItem[k] == null) ndiItem[k] = v; });
        delete config.obs_ndi;
    }
    config.inputs.forEach(item => ensureTransportFields(item));
    const fill = (key, d) => { const o = config[key] = config[key] || {}; Object.keys(d).forEach(k => { if (o[k] == null) o[k] = d[k]; }); };
    delete config.image_encode;   // replaced by the multi-stream encode test
    fill('encode_test', { enabled: false, resolution: '1080p30', stream_count: 1, worker_mode: 'shared' });
    config.encode_test.resolution = String(config.encode_test.resolution).toLowerCase();
    fill('image_persistence', { enabled: false, directory: '', width: 1920, height: 1080, format: 'jpeg', quality: 90, fsync: true, worker_mode: 'shared' });
}

function ensureModelFields(models) {
    Object.values(models || {}).forEach(model => {
        if (!model || typeof model !== 'object') return;
        // inference_time (an idle gap between inferences) was removed: pacing
        // now comes from the per-stream AI FPS target, which made it a no-op.
        delete model.inference_time;
        model.device = model.device == null ? 'auto' : String(model.device).trim().toLowerCase();
        if (model.device === 'gpu') model.device = 'cuda';
        if (model.warmup_iterations == null) model.warmup_iterations = 10;
        if (model.instances == null) model.instances = 1;
    });
}

function ensureImageCaptureFields() {
    cur.image_capture = cur.image_capture || {};
    if (cur.image_capture.enabled == null) cur.image_capture.enabled = false;
    if (cur.image_capture.snapshot_polling_rate == null) cur.image_capture.snapshot_polling_rate = 1000;
}

function render() {
    // Defensive backfill so an older or hand-edited scenario always has a
    // correctly-fielded set of inputs, no matter what shape it arrived in.
    cur.inputs = cur.inputs || [];
    cur.inputs.forEach(ensureTransportFields);
    ensureAiFields(cur);
    ensureImageWorkloadFields(cur);
    ensureModelFields(cur.ai?.models);
    ensureImageCaptureFields();

    let r = $('form');
    r.innerHTML = Object.entries(cur).map(([k, v]) => node(v, [k], k)).join('') + '<datalist id=modelPaths></datalist>';
    bindFormEvents();
    loadModelPaths();
    applyLockToForm();
}

function bindFormEvents() {
    let r = $('form');
    r.querySelectorAll('[data-p]').forEach(i => i.onchange = i.oninput = () => {
        let v = i.dataset.t === 'boolean' ? i.checked : i.dataset.t === 'number' ? +i.value : i.value;
        setp(cur, i.dataset.p.split('.').map(x => /^\d+$/.test(x) ? +x : x), v);
        $('valid').textContent = 'Unsaved';
        // Switching an input's transport changes which fields it needs
        // (e.g. NDI's discovery timing vs synthetic's width/height) —
        // re-render immediately so those fields show up right away instead
        // of only after a save/reload.
        if (/^inputs\.\d+\.transport$/.test(i.dataset.p)) render();
    });

    r.querySelectorAll('.removeBtn').forEach(b => b.onclick = () => {
        let parts = b.dataset.remove.split('.');
        if (parts[0] === 'inputs') cur.inputs.splice(+parts[1], 1);
        $('valid').textContent = 'Unsaved';
        render();
    });
    r.querySelectorAll('.modelBrowse').forEach(b => b.onclick = () => b.previousElementSibling.click());
    r.querySelectorAll('.modelFile').forEach(file => file.onchange = async () => {
        if (!file.files.length) return;
        try {
            const data = new FormData();
            data.append('model', file.files[0], file.files[0].name);
            const result = await api('/api/models/upload', { method: 'POST', body: data });
            const input = file.previousElementSibling;
            input.value = result.path;
            input.dispatchEvent(new Event('change', { bubbles: true }));
            $('valid').textContent = 'Model selected';
        } catch (e) { $('error').textContent = e.message }
    });

    let addInputBtn = r.querySelector('#addInputBtn');
    if (addInputBtn) addInputBtn.onclick = () => {
        cur.inputs = cur.inputs || [];
        let n = cur.inputs.length + 1;
        let item = { id: `stream-${n}`, transport: 'synthetic', enabled: true };
        ensureTransportFields(item);
        cur.inputs.push(item);
        $('valid').textContent = 'Unsaved';
        render();
    };
}

// -- Session lock -------------------------------------------------------
// While a session is running/stopping, the form, JSON editor, and config
// picker are locked so an operator can't edit the config a live session is
// actively using out from under it.
function applyLockToForm() {
    $('form').querySelectorAll('input,select,button').forEach(el => el.disabled = locked);
    $('form').classList.toggle('locked', locked);
}

function setFormLocked(isLocked) {
    if (isLocked === locked) { applyLockToForm(); return }
    locked = isLocked;
    $('config').disabled = locked;
    $('newConfig').disabled = locked;
    $('jsonTab').disabled = locked;
    $('formTab').disabled = locked;
    $('editor').readOnly = locked;
    $('discard').disabled = locked;
    $('save').disabled = locked;
    $('start').disabled = locked;
    applyLockToForm();
}

async function selected() {
    let d = await api('/api/configs/' + encodeURIComponent($('config').value));
    cur = structuredClone(d.data);
    orig = structuredClone(cur);
    $('desc').textContent = cur.description || '';
    render();
    $('editor').value = JSON.stringify(cur, null, 2)
}

async function load() {
    let d = await api('/api/configs');
    let previous = $('config').value;
    $('config').innerHTML = d.configs.map(x => `<option value="${x.id}">${x.name}</option>`).join('');
    if (previous && d.configs.some(x => x.id === previous)) $('config').value = previous;
    await selected()
}

async function loadModelPaths() {
    try {
        const result = await api('/api/models');
        const list = $('modelPaths');
        if (list) list.innerHTML = result.entries.filter(x => !x.directory)
            .map(x => `<option value="${esc('../models/' + x.path)}">${esc(x.name)}</option>`).join('');
    } catch (e) { /* browsing is unavailable when no models directory exists */ }
}

$('config').onchange = selected;

$('newConfig').onclick = async () => {
    let name = prompt('New scenario name:');
    if (!name) return;
    name = name.trim();
    if (!name) return;
    try {
        $('error').textContent = '';
        let r = await api('/api/configs', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name })
        });
        await load();
        $('config').value = r.id;
        await selected();
    } catch (e) { $('error').textContent = e.message }
};

$('jsonTab').onclick = () => {
    $('form').hidden = true;
    $('editor').hidden = false;
    mode = 'json'
};

$('formTab').onclick = () => {
    try {
        cur = JSON.parse($('editor').value)
    } catch (e) { $('error').textContent = 'Invalid JSON: ' + e.message; return }
    render();
    $('form').hidden = false;
    $('editor').hidden = true;
    mode = 'form'
};

$('discard').onclick = () => {
    cur = structuredClone(orig);
    render();
    $('editor').value = JSON.stringify(cur, null, 2);
    $('valid').textContent = '';
    $('error').textContent = ''
};

async function saveCurrent() {
    if (mode === 'json') cur = JSON.parse($('editor').value);
    await api('/api/configs/validate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ data: cur })
    });
    await api('/api/configs/' + encodeURIComponent($('config').value), {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ data: cur })
    });
    orig = structuredClone(cur);
    $('editor').value = JSON.stringify(cur, null, 2);
    $('valid').textContent = 'Saved'
}

$('save').onclick = async () => {
    $('error').textContent = '';
    try { await saveCurrent() } catch (e) { $('error').textContent = e.message }
};

// Starting a session always saves the current form first, so whatever the
// operator just changed is what actually runs — no separate "did I save?"
// step before Start.
$('start').onclick = async () => {
    $('error').textContent = '';
    try {
        await saveCurrent();
        await api('/api/session/start', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ config: $('config').value })
        });
        await status();
    } catch (e) { $('error').textContent = e.message }
};

$('stop').onclick = async () => {
    $('error').textContent = '';
    try { await api('/api/session/stop', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{}' }) }
    catch (e) { $('error').textContent = e.message }
};

function val(o, p) {
    for (let k of p) {
        if (o == null) return null;
        o = o[k]
    }
    return Number.isFinite(+o) ? +o : null
}

// Time-axis labels scale with the session: seconds for short runs, then
// minutes, hours and days (same thresholds as the final report).
function timeUnit(span) {
    span = Math.abs(span);
    return span <= 180 ? ['s', 1] : span <= 10800 ? ['min', 60] : span <= 259200 ? ['h', 3600] : ['d', 86400];
}
function timeAxis(xmax, x0 = 35, w = 365, y = 145) {
    const [suffix, div] = timeUnit(xmax), raw = xmax / div / 4, p = Math.pow(10, Math.floor(Math.log10(raw || 1)));
    const step = [1, 2, 2.5, 5, 10].map(f => f * p).find(v => v >= raw) || 10 * p;
    let out = '';
    for (let v = 0; v <= xmax / div + 1e-9; v += step) {
        const x = x0 + v * div / xmax * w;
        out += `<line class=grid x1=${x.toFixed(1)} y1=8 x2=${x.toFixed(1)} y2=128 /><text x=${(x - 8).toFixed(1)} y=${y}>${+v.toFixed(4)}${suffix}</text>`;
    }
    return out;
}

function chart(t, u, s, ds) {
    let lines = ds.map((d, i) => ({
        n: d[0],
        c: C[i % C.length],
        p: s.map(x => [x.elapsed_seconds, val(x, d[1]) / (d[2] || 1)]).filter(x => x[1] != null)
    })).filter(x => x.p.length);
    if (!lines.length) return '';
    let all = lines.flatMap(x => x.p),
        xx = Math.max(...all.map(x => x[0]), 1),
        yy = Math.max(...all.map(x => x[1]), 1),
        paths = lines.map(x => `<path d="${x.p.map((p, i) => `${i ? 'L' : 'M'} ${35 + p[0] / xx * 365} ${8 + (1 - p[1] / yy) * 120}`).join(' ')}" fill=none stroke="${x.c}" stroke-width=2.5/>`).join('');
    return `<div class=chart><b>${t}</b><svg viewBox="0 0 410 150">${timeAxis(xx)}<text x=2 y=12>${yy.toFixed(1)}${u}</text>${paths}</svg></div>`
}

// Live telemetry: CPU (incl. thermal), RAM, Disk, Network, Storage, and every
// detected GPU — dedicated AND integrated alike — by index (utilization,
// VRAM, thermal, power, clocks) — both as KPI tiles and as full graphs,
// matching what the final report shows.
function gpuShortTag(g) {
    if (!g) return '';
    return g.integrated === true ? ' (iGPU)' : g.integrated === false ? ' (dGPU)' : '';
}
function gpuChartLabel(g, i) {
    if (!g) return `GPU ${i}`;
    let name = g.name || `GPU ${i}`;
    let tag = g.integrated === true ? 'Integrated' : g.integrated === false ? 'Dedicated' : '';
    return tag ? `${name} — ${tag}` : name;
}

function telemetry(t) {
    let s = t.samples || [],
        last = s.at(-1);
    let kpis = [];
    if (last) {
        kpis.push(['CPU (system-wide)', last.cpu_percent, '%']);
        kpis.push(['CPU (this tool)', last.tool_cpu_percent, '%']);
        kpis.push(['CPU Temp', last.cpu_temperature_c, '°C']);
        kpis.push(['RAM', last.ram_percent, '%']);
        kpis.push(['RAM Bus Speed', last.memory_bus_speed_mhz, ' MT/s']);
        kpis.push(['Disk', last.disk_percent, '%']);
        kpis.push(['Tool Processes', last.tool_process_count, '']);
        const gc = last.gpu_combined;
        if (gc && gc.gpu_count) {
            kpis.push(['GPUs Combined Util', gc.gpu_percent_avg, '%']);
            kpis.push(['GPUs Combined VRAM', gc.vram_percent_avg, '%']);
            kpis.push(['GPUs Peak Temp', gc.temperature_c_max, '°C']);
        }
        (last.gpus || []).forEach(g => {
            const label = `${g.name || `GPU ${g.index}`}${gpuShortTag(g)}`;
            kpis.push([`${label} Util`, g.gpu_percent, '%']);
            kpis.push([`${label} VRAM`, g.vram_percent, '%']);
            kpis.push([`${label} Temp`, g.temperature_c, '°C']);
        });
    }
    $('kpis').innerHTML = kpis.map(x => `<div class=kpi><span>${esc(x[0])}</span><b>${x[1] == null ? 'N/A' : (+x[1]).toFixed(1) + x[2]}</b></div>`).join('');

    let h = chart('CPU Utilization (system-wide)', '%', s, [
        ['CPU', ['cpu_percent'], 1]
    ]) + chart('Python Process Utilization (total)', '%', s, [
        ['This tool', ['tool_cpu_percent'], 1]
    ]) + chart('CPU Temperature', '°C', s, [
        ['CPU', ['cpu_temperature_c'], 1]
    ]) + chart('RAM', '%', s, [
        ['RAM', ['ram_percent'], 1]
    ]) + chart('Memory', 'GiB', s, [
        ['RAM', ['ram_used_bytes'], GiB],
        ['This tool (all processes)', ['tool_rss_bytes'], GiB],
        ['Engine process only', ['process_rss_bytes'], GiB]
    ]) + chart('Memory Bus Speed', 'MT/s', s, [
        ['Configured bus speed', ['memory_bus_speed_mhz'], 1]
    ]) + chart('Network', 'Mbps', s, [
        ['TX', ['network_tx_mbps'], 1],
        ['RX', ['network_rx_mbps'], 1]
    ]) + chart('Storage', 'MB/s', s, [
        ['Read', ['disk_read_mbps'], 1],
        ['Write', ['disk_write_mbps'], 1]
    ]);

    // Combined view across every GPU -- dedicated and integrated together --
    // (one glance number), shown before the per-GPU breakdown below.
    const hasGpus = s.some(x => (x.gpus?.length || 0) > 0);
    if (hasGpus) {
        h += chart('All GPUs Combined Utilization', '%', s, [
            ['Average', ['gpu_combined', 'gpu_percent_avg'], 1],
            ['Peak', ['gpu_combined', 'gpu_percent_max'], 1]
        ]) + chart('All GPUs Combined VRAM', '%', s, [
            ['Average', ['gpu_combined', 'vram_percent_avg'], 1],
            ['Peak', ['gpu_combined', 'vram_percent_max'], 1]
        ]) + chart('All GPUs Combined Power', 'W', s, [
            ['Total', ['gpu_combined', 'power_watts_total'], 1]
        ]) + chart('All GPUs Peak Temperature', '°C', s, [
            ['Peak', ['gpu_combined', 'temperature_c_max'], 1]
        ]);
    }

    let n = Math.max(...s.map(x => x.gpus?.length || 0), 0);
    for (let i = 0; i < n; i++) {
        const g = last?.gpus?.[i];
        const label = esc(gpuChartLabel(g, i));
        h += chart(`${label} Utilization`, '%', s, [
            ['Compute', ['gpus', i, 'gpu_percent'], 1],
            ['Mem Ctrl', ['gpus', i, 'memory_controller_percent'], 1],
            ['Encoder', ['gpus', i, 'encoder_percent'], 1],
            ['Decoder', ['gpus', i, 'decoder_percent'], 1]
        ]) + chart(`${label} VRAM`, '%', s, [
            ['VRAM', ['gpus', i, 'vram_percent'], 1]
        ]) + chart(`${label} Thermal`, '°C', s, [
            ['Temp', ['gpus', i, 'temperature_c'], 1]
        ]) + chart(`${label} Power`, 'W', s, [
            ['Power', ['gpus', i, 'power_watts'], 1],
            ['Limit', ['gpus', i, 'power_limit_watts'], 1]
        ]) + chart(`${label} Clocks`, 'MHz', s, [
            ['Graphics', ['gpus', i, 'graphics_clock_mhz'], 1],
            ['Memory', ['gpus', i, 'memory_clock_mhz'], 1]
        ]);
    }
    $('charts').innerHTML = h
}

const AI_STATES = {
    scaling: ['Scaling', 'warn'], target_met: ['Target met', 'ok'], saturated: ['Saturated', 'warn'], instance_failed: ['Instance failed', 'bad'],
    max_instances: ['Max processes reached', 'bad'], failed: ['Failed', 'bad'], fixed: ['Fixed', ''], idle: ['Idle', '']
};
const fnum = (v, d = 2) => v == null || isNaN(v) ? 'N/A' : Number(v).toFixed(d);

function aiInstanceRow(x, retired) {
    const fps = Number(x.instant_fps || 0), share = Number(x.assigned_fps || x.target_rate_fps || 0);
    const scale = Math.max(share, fps, 1) * 1.15;
    const tone = retired ? 'off' : x.state !== 'running' ? 'warn' : share && fps < share * 0.98 ? 'warn' : 'ok';
    return `<div class="ai-inst ${retired ? 'retired' : ''}">
        <span class=ai-inst-name>#${esc(x.instance)}</span>
        <div class="ai-track small"><div class="ai-fill ${tone}" style="width:${Math.min(100, fps / scale * 100).toFixed(1)}%"></div>${share ? `<i style="left:${(share / scale * 100).toFixed(1)}%" title="Assigned share ${fnum(share)} FPS"></i>` : ''}</div>
        <span class=ai-inst-val>${fnum(fps)}${share ? ' / ' + fnum(share) : ''} FPS</span>
        <small>${retired ? esc(x.retired_reason || 'stopped') : `${esc(x.state || '')} · ${x.inferences || 0} inf · ${fnum(x.latency_mean_ms)} ms mean${x.latency_p95_ms == null ? '' : ' · ' + fnum(x.latency_p95_ms) + ' ms p95'}${x.compute_fps ? ' · ' + fnum(x.compute_fps, 0) + ' FPS capacity' : ''}${x.saturation_note ? ' · <b class=warn-text>' + esc(x.saturation_note) + '</b>' : ''}${x.error ? ' · ' + esc(x.error) : ''}`}</small>
    </div>`;
}

function auxCard(title, x, target, targetLabel, extra) {
    const fps = Number(x.instant_fps || 0), scale = Math.max(target || 0, fps, 1) * 1.1;
    const tone = x.state === 'failed' ? 'bad' : !target ? 'ok' : fps >= target * 0.98 ? 'ok' : x.state === 'running' ? 'warn' : '';
    const badge = x.state === 'failed' ? ['Failed', 'bad'] : x.state === 'running' ? (target ? (fps >= target * 0.98 ? ['Keeping up', 'ok'] : ['Behind', 'warn']) : ['Running', 'ok']) : [x.state || 'starting', ''];
    return `<div class="kpi ai-card">
        <div class=ai-head><span>${esc(title)}</span><em class="badge ${badge[1]}">${esc(badge[0])}</em></div>
        <b>${fnum(fps)}${target ? ` <small class=inline>/ ${fnum(target)} FPS ${esc(targetLabel)}</small>` : ' FPS'}</b>
        <div class=ai-track><div class="ai-fill ${tone || 'ok'}" style="width:${Math.min(100, fps / scale * 100).toFixed(1)}%"></div>${target ? `<i style="left:${(target / scale * 100).toFixed(1)}%"></i>` : ''}</div>
        <small>${fnum(x.fps)} FPS avg · ${x.operations || 0} ops · ${fnum(x.latency_mean_ms)} ms mean${x.latency_p95_ms == null ? '' : ' · ' + fnum(x.latency_p95_ms) + ' ms p95'} · max ${fnum(x.latency_max_ms)} ms · ${fnum(x.throughput_mb_per_s)} MB/s${x.capacity_fps ? ' · ' + fnum(x.capacity_fps, 0) + ' FPS capacity' : ''}</small>
        <small>${extra}${x.error ? ' · ' + esc(x.error) : ''}</small>
    </div>`;
}

function obsCard(o) {
    const st = o.obs_version_stats || {};
    const badge = o.error ? ['Error', 'bad'] : o.started ? (o.profiler_available === false ? ['No profiler', 'warn'] : ['Receiving', 'ok']) : ['Starting', ''];
    return `<div class="kpi ai-card">
        <div class=ai-head><span>OBS · scene "${esc(o.scene || '')}" · ${o.ndi_streams ?? o.sources ?? 0} NDI + ${o.synthetic_streams || 0} synthetic stream(s)</span><em class="badge ${badge[1]}">${badge[0]}</em></div>
        <small>OBS render ${fnum(st.activeFps)} FPS · CPU ${fnum(st.cpuUsage, 1)}% · RAM ${fnum(st.memoryUsage, 0)} MB · frame render ${fnum(st.averageFrameRenderTime)} ms · skipped render frames ${st.renderSkippedFrames ?? 'N/A'} / ${st.renderTotalFrames ?? 'N/A'}</small>
        ${o.error ? `<small class=ai-reason><b class=bad-text>${esc(o.error)}</b></small>` : ''}
        ${o.profiler_error ? `<small class=ai-reason><b class=warn-text>Source Profiler: ${esc(o.profiler_error)}</b></small>` : ''}
        ${(o.notes || []).map(n => `<small class=ai-reason>${esc(n)}</small>`).join('')}
    </div>`;
}

function encodeCard(e) {
    const streams = e.streams || [], target = Number(e.target_fps || 30), n = Number(e.stream_count || streams.length || 1);
    const total = streams.reduce((a, x) => a + Number(x.instant_fps || 0), 0), goal = target * n;
    const scale = Math.max(goal, total, 1) * 1.1, ok = total >= goal * 0.98;
    const failed = streams.some(x => x.state === 'failed') || e.error;
    const badge = failed ? ['Error', 'bad'] : streams.every(x => x.state === 'running') ? (ok ? ['Keeping up', 'ok'] : ['Behind', 'warn']) : ['Starting', ''];
    const rows = streams.map((x, i) => {
        const f = Number(x.instant_fps || 0), sc = Math.max(target, f, 1) * 1.15;
        const tone = x.state === 'failed' ? 'bad' : x.state !== 'running' ? 'warn' : f >= target * 0.98 ? 'ok' : 'warn';
        return `<div class=ai-inst>
            <span class=ai-inst-name>#${i + 1}</span>
            <div class="ai-track small"><div class="ai-fill ${tone}" style="width:${Math.min(100, f / sc * 100).toFixed(1)}%"></div><i style="left:${(target / sc * 100).toFixed(1)}%"></i></div>
            <span class=ai-inst-val>${fnum(f)} / ${target} FPS</span>
            <small>${esc(x.state || '')} · ${esc(x.encoder || '')}${x.bitrate_mbps ? ' · ' + fnum(x.bitrate_mbps, 1) + ' Mbps' : ''}${x.keep_up_ratio ? ' · ' + (x.keep_up_ratio * 100).toFixed(1) + '% kept up' : ''}${x.frames_dropped ? ' · <b class=bad-text>' + x.frames_dropped + ' dropped</b>' : ''}${x.write_p95_ms != null ? ' · hand-off p95 ' + fnum(x.write_p95_ms) + ' ms' : ''}${x.fallback_reason ? ' · <b class=bad-text>GPU fallback: ' + esc(x.fallback_reason) + '</b>' : ''}${x.error ? ' · <b class=bad-text>' + esc(x.error) + '</b>' : ''}</small>
        </div>`;
    }).join('');
    return `<div class="kpi ai-card">
        <div class=ai-head><span>Encode test · ${n} × ${esc(e.resolution || '')}</span><em class="badge ${badge[1]}">${esc(badge[0])}</em></div>
        <b>${fnum(total)} <small class=inline>/ ${fnum(goal)} FPS total (${n} × ${target})</small></b>
        <div class=ai-track><div class="ai-fill ${ok ? 'ok' : 'warn'}" style="width:${Math.min(100, total / scale * 100).toFixed(1)}%"></div><i style="left:${(goal / scale * 100).toFixed(1)}%"></i></div>
        <small>Encoder: ${esc(e.gpu_encoder ? 'GPU — ' + e.gpu_encoder : 'CPU (libx264) — no working GPU encoder found')}${e.error ? ' · <b class=bad-text>' + esc(e.error) + '</b>' : ''}</small>
        <div class=ai-instances>${rows}</div>
    </div>`;
}

function renderAux(aux) {
    const el = $('auxWorkloads');
    if (!el) return;
    const cards = [];
    const e = aux?.encode;
    if (aux?.obs) cards.unshift(obsCard(aux.obs));
    if (e) cards.push(encodeCard(e));
    (aux?.persistence || []).forEach(p => cards.push(auxCard(`Persistence · ${p.model}`, p, Number(p.requested_fps || 0), 'model FPS',
        `${p.width}×${p.height} ${esc((p.format || '').toUpperCase())} · ${p.image_bytes ? (p.image_bytes / 1024).toFixed(0) + ' KiB' : ''} · fsync ${p.fsync ? 'on' : 'off'} · ${esc(p.image_path || '')}${p.keep_up_ratio ? ' · ' + (p.keep_up_ratio * 100).toFixed(1) + '% of requested saves' : ''}`)));
    el.innerHTML = cards.join('') || '<p class="muted">The encode test and image persistence are disabled.</p>';
}

function renderAiAndProcesses(ai, telemetryData) {
    const models = ai?.models || [];
    $('aiModels').innerHTML = models.length ? models.map(m => {
        const total = Number(m.instant_fps || 0), target = Number(m.target_fps || 0);
        const scale = Math.max(target, total, 1) * 1.1;
        const [label, tone] = AI_STATES[m.state] || [m.state || '', ''];
        const instances = m.instances || [], retired = m.retired_instances || [];
        return `<div class="kpi ai-card">
            <div class=ai-head><span>${esc(m.name)}</span><em class="badge ${tone}">${esc(label)}</em></div>
            <b>${fnum(total)}${target ? ` <small class=inline>/ ${fnum(target)} FPS target</small>` : ' FPS'}</b>
            <div class=ai-track><div class="ai-fill ${tone || 'ok'}" style="width:${Math.min(100, total / scale * 100).toFixed(1)}%"></div>${target ? `<i style="left:${(target / scale * 100).toFixed(1)}%" title="Target ${fnum(target)} FPS"></i>` : ''}</div>
            <small>${target ? `${fnum(m.target_fps_per_stream, 1)} FPS × ${m.stream_count} stream(s) · ` : ''}${m.instance_count || 0} instance(s) ${m.worker_mode === 'process' ? 'as separate processes' : 'as threads in 1 shared process'}${m.required_instances ? ` · <b class=ok-text>${m.required_instances} required</b>` : ''} · ${fnum(m.fps)} FPS steady avg · ${fnum(m.latency_mean_ms)} ms mean${m.latency_p95_ms == null ? '' : ' · ' + fnum(m.latency_p95_ms) + ' ms worst p95'}${m.provider ? ' · ' + esc(m.provider) : ''}</small>
            ${m.scaling_reason ? `<small class=ai-reason>${esc(m.scaling_reason)}</small>` : ''}
            ${m.device_fallback ? `<small class=ai-reason><b class=warn-text>Device fallback: ${esc(m.device_fallback)}</b></small>` : ''}
            <div class=ai-instances>${instances.map(x => aiInstanceRow(x, false)).join('')}${retired.map(x => aiInstanceRow(x, true)).join('')}</div>
        </div>`;
    }).join('') : '<p class="muted">No enabled AI models.</p>';
    const samples = telemetryData.samples || [];
    const processNames = [...new Set(samples.flatMap(s => (s.processes || []).map(p => p.name)))];
    const total = chart('Python Process Utilization (total)', '%', samples, [['All benchmark processes', ['tool_cpu_percent'], 1]]);
    $('processCharts').innerHTML = total + processNames.map(name => {
        const points = samples.map(s => {
            const process = (s.processes || []).find(p => p.name === name);
            return process ? [s.elapsed_seconds, process.cpu_percent] : null;
        }).filter(Boolean);
        if (!points.length) return '';
        const maxX = Math.max(...points.map(p => p[0]), 1), maxY = Math.max(...points.map(p => p[1]), 1);
        const d = points.map((p, i) => `${i ? 'L' : 'M'} ${35 + p[0] / maxX * 365} ${8 + (1 - p[1] / maxY) * 120}`).join(' ');
        return `<div class=chart><b>${esc(name)} CPU</b><svg viewBox="0 0 410 150"><line class=axis x1=35 y1=8 x2=35 y2=128/><line class=axis x1=35 y1=128 x2=400 y2=128/><text x=2 y=12>${maxY.toFixed(1)}%</text>${timeAxis(maxX)}<path d="${d}" fill=none stroke="#ffc95c" stroke-width=2.5/></svg></div>`;
    }).join('');
}

// -- Static system information (CPU, RAM, every GPU, disks, USB devices) ---
function renderSystemInfo(inv) {
    if (!inv) return;
    let top = [
        ['Host', inv.hostname],
        ['OS', inv.os],
        ['CPU', inv.cpu_model],
        ['Cores', `${inv.physical_cpu_count ?? '?'} phys / ${inv.logical_cpu_count ?? '?'} logical`],
        ['RAM', inv.ram_total_bytes != null ? `${(inv.ram_total_bytes / GiB).toFixed(1)} GiB` : '-']
        ,['RAM Bus', inv.memory_bus_speed_mhz != null ? `${Number(inv.memory_bus_speed_mhz).toFixed(0)} MT/s` : '-']
    ].map(x => `<div class="datum"><span>${x[0]}</span><b>${x[1] ?? '-'}</b></div>`).join('');

    let gpus = (inv.gpus || []).map(g => `<div class="datum">
            <span>GPU ${g.index}</span><b>${esc(g.name || 'Unknown')}</b>
            <small>${g.vram_total_bytes != null ? (g.vram_total_bytes / GiB).toFixed(1) : '?'} GiB VRAM &middot; driver ${esc(g.driver_version || '?')}</small>
        </div>`).join('') || `<div class="datum wide"><span>GPU</span><b>None detected</b></div>`;

    let disks = (inv.disks || []).map(d => `<div class="datum">
            <span>${esc(d.mountpoint)}</span><b>${(d.used_bytes / GiB).toFixed(1)} / ${(d.total_bytes / GiB).toFixed(1)} GiB</b>
            <small>${esc(d.filesystem || '')} &middot; ${(d.percent ?? 0).toFixed(1)}% used</small>
        </div>`).join('') || `<div class="datum wide"><span>Disk</span><b>None detected</b></div>`;

    let usb = (inv.usb_devices || []).map(u => `<li>${esc(u.description || '')}</li>`).join('')
        || `<li class="muted">No USB devices detected${inv.usb_inventory_warning ? ' (' + esc(inv.usb_inventory_warning) + ')' : ''}</li>`;

    $('sysinfo').innerHTML = `
        <div class="info">${top}</div>
        <h3>GPUs</h3><div class="info">${gpus}</div>
        <h3>Storage</h3><div class="info">${disks}</div>
        <h3>USB Devices</h3><ul class="usb-list">${usb}</ul>
    `;
}

async function loadSystemInfo() {
    try {
        renderSystemInfo(await api('/api/inventory'));
    } catch (e) { /* system info is best-effort; ignore transient failures */ }
}

function clock(x) {
    x = Math.max(0, Math.floor(x));
    return [Math.floor(x / 3600), Math.floor(x % 3600 / 60), x % 60].map(x => String(x).padStart(2, '0')).join(':')
}

// The live log grows every second. If the operator has scrolled up to read
// something, leave their view alone; only snap to the bottom when they were
// already there, so new lines don't yank them away mid-read.
function updateLog(text) {
    const el = $('log');
    const atBottomThreshold = 32;
    const wasAtBottom = el.scrollHeight - el.scrollTop - el.clientHeight <= atBottomThreshold;
    el.textContent = text;
    if (wasAtBottom) el.scrollTop = el.scrollHeight;
}

const reportHref = p => '/reports/' + p.split(/[/\\]reports[/\\]/).pop().replaceAll('\\', '/');

async function status() {
    let s = await api('/api/session');
    if (typeof renderStreamFps === 'function') {
        renderStreamFps(s.stream_metrics || {
            streams: []
        });
    }
    $('state').textContent = s.status;
    setFormLocked(s.status === 'running' || s.status === 'stopping');

    let st = s.started_at ? new Date(s.started_at) : null,
        fin = s.finished_at ? new Date(s.finished_at) : new Date();

        $('elapsed').textContent = clock(st ? (fin - st) / 1000 : 0);

    let fs = [
        ['Session', s.id],
        ['Scenario', s.scenario],
        ['Started', s.started_at],
        ['Finished', s.finished_at],
        ['Output', s.output_dir]
    ];

    $('session').innerHTML = fs.map((x, i) => `<div class="datum ${i == 4 ? 'wide' : ''}"><span>${x[0]}</span><b>${x[1] || '-'}</b></div>`).join('') + (s.report_path ? `<div class="datum wide"><a target=_blank href="${reportHref(s.report_path)}">Open report</a>${s.report_pdf_path ? ` &middot; <a target=_blank href="${reportHref(s.report_pdf_path)}">Open PDF report</a>` : ''}</div>` : '');

    telemetry(s.telemetry || {});
    renderAiAndProcesses(s.ai_metrics || { models: [] }, s.telemetry || {});
    renderAux(s.aux_metrics || {});

    let l = s.log_tail || '';
    updateLog(l);
    $('count').textContent = l.split('\n').filter(Boolean).length + ' lines'
}

load().then(status);
loadSystemInfo();

setInterval(status, 1000);
setInterval(loadSystemInfo, 15000);
