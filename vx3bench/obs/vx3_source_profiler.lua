--[[
VX3 Benchmark - OBS Source Profiler exporter.

Loaded automatically into the OBS instance the benchmark launches (it is
registered in the benchmark's own scene collection). Once a second it reads
OBS's Source Profiler -- the same numbers as View > Source Profiler -- for
every NDI source (DistroAV, kind "ndi_source") and every synthetic stream
(OBS Media Source playing the benchmark's SpeedHQ clip, kind
"ffmpeg_source") and writes them to a JSON
file the benchmark reads.

The Source Profiler is not exposed through obs-websocket, and not through
OBS's scripting bindings, but its functions are exported from libobs itself,
so they are called here through LuaJIT's FFI (OBS's Lua is LuaJIT).

async_input   = FPS of frames the NDI plugin submitted to OBS (each frame
                received from the network), averaged by OBS over ~5 s.
async_rendered= FPS of those frames OBS actually rendered.
]]

obs = obslua
-- OBS normally embeds LuaJIT (which has the FFI). A build with plain Lua has
-- no FFI: then the script still runs and reports the sources, with a clear
-- "profiler unavailable" reason, instead of failing to load.
local has_ffi, ffi = pcall(require, "ffi")
local CDEFS = [[
typedef struct obs_source obs_source_t;
typedef struct profiler_result {
    uint64_t tick_avg;
    uint64_t tick_max;
    uint64_t render_avg;
    uint64_t render_max;
    uint64_t render_gpu_avg;
    uint64_t render_gpu_max;
    uint64_t render_sum;
    uint64_t render_gpu_sum;
    double async_input;
    double async_rendered;
    uint64_t async_input_best;
    uint64_t async_input_worst;
    uint64_t async_rendered_best;
    uint64_t async_rendered_worst;
} profiler_result_t;
void source_profiler_enable(bool enable);
bool source_profiler_fill_result(obs_source_t *source, profiler_result_t *result);
obs_source_t *obs_get_source_by_name(const char *name);
void obs_source_release(obs_source_t *source);
]]
if has_ffi then
    pcall(ffi.cdef, CDEFS)
end

local libobs = nil
local load_error = ""
local output_path = ""
local source_kinds = {ndi_source = true}     -- set from the "source_kind" setting (comma-separated)
local result = has_ffi and ffi.new("profiler_result_t") or nil
local started = false

local function has_profiler(lib)
    return pcall(function() return lib.source_profiler_fill_result end)
end

local function try_load()
    if not has_ffi then
        load_error = "this OBS build's Lua has no FFI (LuaJIT), so the Source Profiler cannot be read"
        return nil
    end
    -- Linux / macOS: libobs is already loaded in this process, so resolve the
    -- symbols from the running program itself (no file name needed). This
    -- matters because Linux builds name the library after OBS's major
    -- version since OBS 30 (libobs.so.30, .31, ...), and Flatpak/other
    -- packages install it in different places.
    if ffi.os ~= "Windows" and has_profiler(ffi.C) then
        return ffi.C
    end
    local candidates
    if ffi.os == "Windows" then
        candidates = {"obs", "obs.dll"}
    elseif ffi.os == "OSX" then
        candidates = {"@rpath/libobs.framework/Versions/A/libobs", "libobs.framework/libobs", "libobs"}
    else
        candidates = {"libobs.so.32", "libobs.so.31", "libobs.so.30", "libobs.so.0", "libobs.so",
                      "/app/lib/libobs.so.32", "/app/lib/libobs.so.31", "/app/lib/libobs.so.30"}
    end
    local loaded_any = false
    for _, name in ipairs(candidates) do
        local ok, lib = pcall(ffi.load, name)
        if ok then
            loaded_any = true
            if has_profiler(lib) then return lib end
        end
    end
    if loaded_any or ffi.os ~= "Windows" then
        load_error = "Source Profiler functions not found in this OBS (OBS 30.1 or newer is required)"
    else
        load_error = "could not load obs.dll from the OBS process"
    end
    return nil
end

local function esc(s)
    s = tostring(s or "")
    s = s:gsub('\\', '\\\\'):gsub('"', '\\"'):gsub('\n', '\\n'):gsub('\r', '\\r'):gsub('\t', '\\t')
    return s
end

local function num(v)
    v = tonumber(v) or 0
    if v ~= v or v == math.huge or v == -math.huge then return "0" end
    return string.format("%.6f", v)
end

local function ns_to_ms(v) return tonumber(v) / 1e6 end

local function write_json(text)
    if output_path == "" then return end
    local tmp = output_path .. ".tmp"
    local f = io.open(tmp, "w")
    if not f then return end
    f:write(text)
    f:close()
    os.remove(output_path)          -- Windows rename does not overwrite
    os.rename(tmp, output_path)
end

local function collect()
    local parts = {}
    local sources = obs.obs_enum_sources()
    if sources ~= nil then
        for _, src in ipairs(sources) do
            local kind = obs.obs_source_get_unversioned_id(src)
            if source_kinds[kind] then
                local name = obs.obs_source_get_name(src)
                local entry = '"' .. esc(name) .. '":{'
                    .. '"kind":"' .. esc(kind) .. '"'
                    .. ',"width":' .. obs.obs_source_get_width(src)
                    .. ',"height":' .. obs.obs_source_get_height(src)
                    .. ',"active":' .. tostring(obs.obs_source_active(src))
                    .. ',"showing":' .. tostring(obs.obs_source_showing(src))
                local have = false
                if libobs ~= nil then
                    local ptr = libobs.obs_get_source_by_name(name)
                    if ptr ~= nil then
                        have = libobs.source_profiler_fill_result(ptr, result)
                        libobs.obs_source_release(ptr)
                    end
                end
                if have then
                    entry = entry
                        .. ',"async_input_fps":' .. num(result.async_input)
                        .. ',"async_rendered_fps":' .. num(result.async_rendered)
                        .. ',"async_input_best_ms":' .. num(ns_to_ms(result.async_input_best))
                        .. ',"async_input_worst_ms":' .. num(ns_to_ms(result.async_input_worst))
                        .. ',"async_rendered_worst_ms":' .. num(ns_to_ms(result.async_rendered_worst))
                        .. ',"tick_avg_ms":' .. num(ns_to_ms(result.tick_avg))
                        .. ',"render_avg_ms":' .. num(ns_to_ms(result.render_avg))
                        .. ',"render_max_ms":' .. num(ns_to_ms(result.render_max))
                        .. ',"render_gpu_avg_ms":' .. num(ns_to_ms(result.render_gpu_avg))
                        .. ',"profiled":true'
                else
                    entry = entry .. ',"profiled":false'
                end
                parts[#parts + 1] = entry .. '}'
            end
        end
        obs.source_list_release(sources)
    end
    local text = '{"script_version":1'
        .. ',"written_at_ms":' .. string.format("%.0f", os.time() * 1000)
        .. ',"profiler_available":' .. tostring(libobs ~= nil)
        .. ',"profiler_error":"' .. esc(load_error) .. '"'
        .. ',"obs":{"active_fps":' .. num(obs.obs_get_active_fps())
        .. ',"average_frame_time_ms":' .. num(ns_to_ms(obs.obs_get_average_frame_time_ns()))
        .. ',"lagged_frames":' .. tostring(obs.obs_get_lagged_frames())
        .. ',"total_frames":' .. tostring(obs.obs_get_total_frames()) .. '}'
        .. ',"sources":{' .. table.concat(parts, ",") .. '}}'
    write_json(text)
end

function script_description()
    return "VX3 Benchmark: exports OBS Source Profiler FPS for NDI sources to a JSON file once a second."
end

function script_properties()
    local props = obs.obs_properties_create()
    obs.obs_properties_add_text(props, "output_path", "Output JSON file", obs.OBS_TEXT_DEFAULT)
    obs.obs_properties_add_text(props, "source_kind", "Source kind", obs.OBS_TEXT_DEFAULT)
    return props
end

function script_defaults(settings)
    obs.obs_data_set_default_string(settings, "source_kind", "ndi_source")
end

function script_update(settings)
    output_path = obs.obs_data_get_string(settings, "output_path")
    local kind = obs.obs_data_get_string(settings, "source_kind")
    if kind ~= nil and kind ~= "" then
        source_kinds = {}
        for k in string.gmatch(kind, "[^,%s]+") do source_kinds[k] = true end
    end
end

function script_load(settings)
    script_update(settings)
    libobs = try_load()
    if libobs ~= nil then
        libobs.source_profiler_enable(true)
    end
    if not started then
        obs.timer_add(collect, 1000)
        started = true
    end
    collect()
end

function script_unload()
    obs.timer_remove(collect)
end
