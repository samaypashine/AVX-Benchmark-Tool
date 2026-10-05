def analyze(p):
    t=p.get('telemetry_summary',{});cfg=p.get('config',{});targets=cfg.get('targets',{});streams=p.get('streams',[]);obs=[]
    def mx(k):
        v=t.get(k,{});return v.get('max') if isinstance(v,dict) else None
    def gmx(k):
        v=[g.get(k,{}).get('max') for g in t.get('gpus',[]) if isinstance(g.get(k),dict) and g[k].get('max') is not None];return max(v) if v else None
    def add(s,e,r,score):obs.append({'subsystem':s,'severity':'high' if score>=85 else 'medium','evidence':e,'recommendation':r,'score':score})
    cpu=mx('cpu_percent');gpu=gmx('gpu_percent');vram=gmx('vram_percent');ram=mx('ram_percent');ct=mx('cpu_temperature_c');gt=gmx('temperature_c')
    if cpu is not None and cpu>=targets.get('cpu_percent_max',90):add('CPU',f'Peak CPU was {cpu:.1f}%.','Reduce the number of concurrent streams or check for contention from other processes.',90)
    if gpu is not None and gpu>=targets.get('gpu_percent_max',95):add('GPU Compute',f'Peak GPU compute was {gpu:.1f}%.','Check whether anything else on the machine is using the GPU concurrently.',92)
    if vram is not None and vram>=targets.get('vram_percent_max',90):add('GPU Memory',f'Peak VRAM was {vram:.1f}%.','Check for other processes holding VRAM.',91)
    if ram is not None and ram>=targets.get('ram_percent_max',90):add('System Memory',f'Peak RAM was {ram:.1f}%.','Reduce the number of concurrent streams or inspect RSS growth.',88)
    if (ct is not None and ct>=targets.get('temperature_c_max',90)) or (gt is not None and gt>=targets.get('temperature_c_max',90)):add('Thermal',f'Peak CPU/GPU temperatures were {ct}/{gt} °C.','Improve cooling and repeat a soak test.',95)
    dropped=sum(s.get('dropped',0) or 0 for s in streams);frames=sum(s.get('frames',0) or 0 for s in streams);dp=100*dropped/max(1,frames+dropped)
    if dp>targets.get('dropped_frame_percent_max',1):add('Input/Network',f'Dropped-frame rate was {dp:.2f}%.','Check NIC, switch, NDI buffers, and source stability.',87)
    fps_target=targets.get('stream_fps_min',0)
    if fps_target:
        slow=[s['stream_id'] for s in streams if (s.get('fps',0) or 0)<fps_target]
        if slow:add('NDI Stream Rate',f'{len(slow)} of {len(streams)} stream(s) averaged below the {fps_target} FPS target: {", ".join(slow[:5])}{"..." if len(slow)>5 else ""}.','Check network bandwidth, NDI source health, and CPU headroom for the affected streams.',89)
    for m in p.get('ai_models',[]) or []:
        target=m.get('target_fps') or 0
        if target and not m.get('target_met') and m.get('state') in ('saturated','instance_failed','max_instances','failed','scaling'):
            achieved=m.get('fps') or m.get('instant_fps') or 0
            add('AI Throughput',f"{m.get('name')} reached {achieved:.1f} of {target:.1f} FPS ({m.get('target_fps_per_stream')} FPS x {m.get('stream_count')} streams) with {m.get('instance_count')} instance(s): {m.get('scaling_reason') or m.get('state')}.",
                'The AI accelerator (or CPU for CPU models) did not sustain the per-stream AI target at this stream count. Check the per-instance-count totals in the AI section: if they stopped rising, the hardware is saturated -- use a faster device/provider (e.g. TensorRT), a smaller model, fewer streams per box, or a lower per-stream target; if they were still rising, raise the max processes per model.',93)
    for m in p.get('ai_models',[]) or []:
        if m.get('device_fallback') and str(m.get('requested_device','auto')) != 'auto':
            add('AI Device',f"{m.get('name')}: {m['device_fallback']}.",
                'The requested accelerator is not usable on this machine, so the model ran on a slower device. Install the matching GPU driver/runtime (CUDA+cuDNN, TensorRT, DirectML or ROCm) and ONNX Runtime package, or choose the device that is available.',80)
    enc=p.get('encode_test') or {}
    behind=[(i+1,s) for i,s in enumerate(enc.get('streams') or []) if s.get('keep_up_ratio') is not None and s['keep_up_ratio']<0.98]
    if behind:
        worst=min(behind,key=lambda x:x[1]['keep_up_ratio'])
        add('Encode',f"{len(behind)} of {enc.get('stream_count')} {enc.get('resolution')} encode stream(s) fell below 30 FPS (worst: stream #{worst[0]} at {worst[1].get('fps') or 0:.1f} FPS, {worst[1]['keep_up_ratio']*100:.1f}% kept up, encoder {worst[1].get('encoder')}).",
            ('The GPU encoder is at its session or throughput limit; reduce encode streams or resolution, or use a GPU with more encoder capacity.' if enc.get('gpu_encoder') else
             'CPU encoding (no working GPU encoder was found) cannot sustain this many streams; install an FFmpeg with NVENC/Quick Sync/AMF and a supported GPU driver, or reduce streams/resolution.'),88)
    for x in p.get('image_persistence') or []:
        r=x.get('keep_up_ratio')
        if r is not None and r<0.98:
            add('Storage / Image Persistence',f"Persistence for {x.get('model')} saved only {r*100:.1f}% of the images requested by the model's FPS (mean {x.get('latency_mean_ms') or 0:.2f} ms, max {x.get('latency_max_ms') or 0:.2f} ms per save, fsync {'on' if x.get('fsync') else 'off'}).",
                'The storage path cannot sustain one saved image per inference; check disk telemetry and consider faster storage (NVMe), a different directory/volume, or smaller images.',89)
    growth=t.get('process_rss_growth_bytes',0) or 0
    if growth>targets.get('process_rss_growth_mb_max',256)*2**20:add('Memory Growth',f'Engine process RSS grew by {growth/2**20:.1f} MiB.','Run a longer soak test and watch for a steady climb.',78)
    obs.sort(key=lambda x:x['score'],reverse=True);primary=obs[0]['subsystem'] if obs else 'No dominant bottleneck detected'
    return {'primary_bottleneck':primary,'confidence':'high' if len(obs)>1 else ('medium' if obs else 'low'),'summary':f'The strongest measured bottleneck indicator is {primary}.' if obs else 'No configured threshold violation produced a dominant bottleneck indicator.','dropped_frame_percent':dp,'observations':obs,'methodology':'Threshold analysis across NDI stream FPS, AI throughput vs per-stream target, dropped frames, CPU, GPU, memory, storage, network, and thermal measurements.'}
