import { useEffect, useMemo, useState } from 'react'
import { api } from '../api'
import { Dropzone, EmptyState, Metric, Panel } from './Shared'

function Trend({ runs, field, label }) {
  if (!runs.length) return <EmptyState title="No measured runs" detail="Run an explicit benchmark to populate this chart." />
  const max = Math.max(...runs.map((run) => Number(run[field]) || 0), 1)
  return <div className="trend"><div className="trend-bars">{runs.map((run, index) => <div className="trend-bar-wrap" key={run.run_id}><div className="trend-bar" style={{ height: `${Math.max(4, (Number(run[field]) || 0) / max * 100)}%` }} title={`${run[field]} ${label}`} /><small>{index + 1}</small></div>)}</div><p>{label} · measured benchmark order</p></div>
}

export default function Benchmarks() {
  const [data, setData] = useState(null)
  const [file, setFile] = useState(null)
  const [runs, setRuns] = useState(3)
  const [enhancement, setEnhancement] = useState(true)
  const [confidence, setConfidence] = useState(.35)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const load = async () => { try { setData(await api.benchmarks()) } catch (err) { setError(err.message || 'Benchmark service unavailable.') } }
  useEffect(() => { load() }, [])

  const summary = data?.summary
  const measured = data?.runs || []
  const modelConfigs = useMemo(() => [...new Set(measured.map((run) => run.model_config))], [measured])

  const runBenchmark = async () => {
    if (!file) return
    setLoading(true); setError('')
    try { await api.runImageBenchmark({ file, runs, enhancement, confidence }); await load() } catch (err) { setError(err.message || 'Benchmark runtime unavailable.') }
    finally { setLoading(false) }
  }

  return <div className="page">
    <div className="section-intro"><span className="eyebrow">EVALUATION / MEASURED PERFORMANCE</span><h1>Performance,<br /><span>measured honestly.</span></h1><p>Benchmark runs are explicit actions and are stored separately from inference history. These numbers describe pipeline performance only—not model accuracy without ground-truth evaluation.</p></div>
    <Panel eyebrow="01 / EXPLICIT BENCHMARK" title="Run a measured benchmark">
      <div className="benchmark-control"><Dropzone file={file} onFile={setFile} accept="image/*" label="Drop the benchmark image here" hint="The same image is used for each measured iteration." /><div className="benchmark-options"><label>Iterations <select value={runs} onChange={(e) => setRuns(Number(e.target.value))}><option value="1">1 run</option><option value="3">3 runs</option><option value="5">5 runs</option><option value="10">10 runs</option></select></label><label>Confidence <input type="range" min=".05" max=".95" step=".01" value={confidence} onChange={(e) => setConfidence(Number(e.target.value))} /><b>{Math.round(confidence * 100)}%</b></label><label className="inline-toggle"><span>NAFNet</span><button className={`toggle ${enhancement ? 'active' : ''}`} onClick={() => setEnhancement(!enhancement)}><i /></button></label><button className="primary" disabled={!file || loading} onClick={runBenchmark}>{loading ? 'Measuring…' : 'Run benchmark →'}</button></div></div>
      {error && <div className="error-box"><b>Benchmark unavailable</b><span>{error}</span></div>}
    </Panel>

    <div className="metric-grid benchmark-metrics"><Metric label="Benchmark actions" value={summary?.runs} tone="accent" /><Metric label="Average latency" value={summary ? summary.average_latency_ms.toFixed(2) : undefined} unit="ms" /><Metric label="Min / max" value={summary ? `${summary.min_latency_ms.toFixed(1)} / ${summary.max_latency_ms.toFixed(1)}` : undefined} unit="ms" /><Metric label="Average FPS" value={summary ? summary.average_fps.toFixed(1) : undefined} unit="fps" /><Metric label="Average detections" value={summary ? summary.average_detections.toFixed(2) : undefined} /></div>

    <div className="two-col"><Panel eyebrow="02 / LATENCY TREND" title="Latency over benchmark runs"><Trend runs={measured} field="latency_ms" label="Latency (ms)" /></Panel><Panel eyebrow="03 / THROUGHPUT TREND" title="FPS over benchmark runs"><Trend runs={measured} field="fps" label="FPS" /></Panel></div>

    <Panel eyebrow="04 / RUN LEDGER" title="Measured benchmark history" action={modelConfigs.length > 1 && <span className="muted">{modelConfigs.length} configurations</span>}>
      {measured.length ? <div className="history-table benchmark-table"><div className="history-header"><span>Timestamp</span><span>Type</span><span>Latency</span><span>FPS</span><span>Detections</span><span>Model config</span></div>{measured.map((run) => <div className="history-row static" key={run.run_id}><span>{new Date(run.timestamp).toLocaleString()}</span><span>{run.media_type}</span><span>{run.latency_ms} ms</span><span>{run.fps}</span><span>{run.detections}</span><span>{run.model_config}</span></div>)}</div> : <EmptyState title="No measured benchmark runs" detail="No benchmark numbers are fabricated. Run the explicit action above after model weights are configured." />}
    </Panel>
  </div>
}
