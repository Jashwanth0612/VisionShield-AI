import { useMemo, useState } from 'react'
import { api } from '../api'
import { ArtifactLinks, Dropzone, EmptyState, Metric, ModelHealth, Panel } from './Shared'

const WEATHER_OPTIONS = [
  ['auto', 'Auto detect'],
  ['fog_its', 'Fog · ITS'],
  ['fog_ots', 'Fog · OTS'],
  ['rain', 'Rain'],
  ['snow', 'Snow'],
  ['low_light', 'Low-Light'],
]

function Comparison({ result, preview, view }) {
  if (!result) return <EmptyState title="Awaiting a model response" detail="Run inference after the configured runtime is available." />
  const src = view === 'original' ? preview : view === 'enhanced' ? result.enhanced_image : result.annotated_image
  if (view !== 'annotated') return <img className="result-image" src={src} alt={`${view} view`} />
  const width = result.image_size?.width || 1
  const height = result.image_size?.height || 1
  return <div className="bbox-stage"><img className="result-image" src={result.enhanced_image || preview} alt="Detection overlay" />{result.detections.map((detection, index) => {
    const [x1, y1, x2, y2] = detection.bbox
    return <div className="bbox" key={`${detection.label}-${index}`} style={{ left: `${x1 / width * 100}%`, top: `${y1 / height * 100}%`, width: `${(x2 - x1) / width * 100}%`, height: `${(y2 - y1) / height * 100}%` }}><b>{detection.label}</b><span>{Math.round(detection.confidence * 100)}%</span></div>
  })}</div>
}

export default function Workspace({ health, onRefresh }) {
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState('')
  const [enhancement, setEnhancement] = useState(true)
  const [weather, setWeather] = useState('auto')
  const [confidence, setConfidence] = useState(.35)
  const [view, setView] = useState('annotated')
  const [result, setResult] = useState(null)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)

  const chooseFile = (next) => {
    if (!next) return
    if (!next.type.startsWith('image/')) { setError('Please choose a supported image file.'); return }
    setFile(next); setResult(null); setError(''); setPreview(URL.createObjectURL(next))
  }

  const run = async () => {
    if (!file) return
    setLoading(true); setError('')
    try { setResult(await api.processImage({ file, enhancement, confidence, weather })) }
    catch (err) { setResult(null); setError(err.message || 'Inference failed.') }
    finally { setLoading(false) }
  }

  const averageConfidence = useMemo(() => {
    if (!result?.detections?.length) return null
    return `${Math.round(result.detections.reduce((sum, item) => sum + item.confidence, 0) / result.detections.length * 100)}%`
  }, [result])

  return <div className="page">
    <div className="hero">
      <div><span className="eyebrow">ALL-WEATHER PERCEPTION / IMAGE</span><h1>See clearly.<br /><span>Detect confidently.</span></h1><p>Weather-routed NAFNet restoration followed by local RT-DETRv2 detection. Choose the known condition or let the rule-based router select the restoration model.</p></div>
      <div className="pipeline-card"><span>PIPELINE</span><strong>Weather <i>→</i> NAFNet <i>→</i> RT-DETR</strong><small>Restore · Detect · Explain</small></div>
    </div>

    <div className="two-col">
      <Panel eyebrow="01 / INPUT + CONTROL" title="Inference workspace">
        <Dropzone file={file} onFile={chooseFile} accept="image/*" label="Drop an image here" hint="or click to browse · JPG, PNG, WEBP · max 20 MB" />
        <div className="control-list">
          <div className="control-row"><div><b>Weather route</b><small>Routes to one of the five project NAFNet conditions</small></div><select className="weather-select" value={weather} onChange={(e) => setWeather(e.target.value)}>{WEATHER_OPTIONS.map(([value, label]) => <option key={value} value={value}>{label}</option>)}</select></div>
          <div className="control-row"><div><b>NAFNet enhancement</b><small>Restore degraded imagery before detection</small></div><button className={`toggle ${enhancement ? 'active' : ''}`} onClick={() => setEnhancement(!enhancement)} aria-label="Toggle NAFNet"><i /></button></div>
          <div className="range-row"><div><b>RT-DETR confidence</b><strong>{Math.round(confidence * 100)}%</strong></div><input type="range" min=".05" max=".95" step=".01" value={confidence} onChange={(e) => setConfidence(Number(e.target.value))} /></div>
        </div>
        <button className="primary" disabled={!file || loading} onClick={run}>{loading ? 'Processing measured inference…' : 'Run VisionShield pipeline →'}</button>
        {!file && <p className="hint">Select an image before running the configured models.</p>}
        {error && <div className="error-box"><b>Inference unavailable</b><span>{error}</span></div>}
      </Panel>

      <Panel eyebrow="02 / OUTPUT + OVERLAY" title="Detection result" action={result && <span className="success">● MEASURED</span>}>
        <div className="tabs">{['original', 'enhanced', 'annotated'].map((item) => <button className={view === item ? 'active' : ''} key={item} onClick={() => setView(item)}>{item}</button>)}</div>
        <div className="result-stage"><Comparison result={result} preview={preview} view={view} /></div>
        {result && <><div className="metric-grid"><Metric label="Objects" value={result.detections_count} tone="accent" /><Metric label="Confidence" value={averageConfidence} /><Metric label="Latency" value={result.metrics.total_latency_ms} unit="ms" /><Metric label="FPS equivalent" value={result.metrics.fps_equivalent} /></div><div className="route-readout"><span>WEATHER ROUTE</span><b>{result.weather?.label || 'Not reported'}</b><small>{result.weather?.source === 'rule-based-auto' ? 'Rule-based auto selection' : 'Operator selected'}</small></div><ArtifactLinks artifacts={result.artifacts} /></>}
      </Panel>
    </div>

    {result && <div className="two-col lower-grid">
      <Panel eyebrow="03 / DETECTIONS" title="Objects found"><div className="detection-list">{result.detections.length ? result.detections.map((item, index) => <div className="detection-row" key={`${item.label}-${index}`}><span className="detection-dot" /><b>{item.label}</b><div className="confidence-track"><i style={{ width: `${item.confidence * 100}%` }} /></div><strong>{Math.round(item.confidence * 100)}%</strong></div>) : <EmptyState title="No objects above threshold" detail="The model returned a valid result with zero detections." />}</div></Panel>
      <Panel eyebrow="04 / RUNTIME" title="Model health" action={<button className="ghost" onClick={onRefresh}>Refresh</button>}><ModelHealth health={health} /></Panel>
    </div>}
  </div>
}
