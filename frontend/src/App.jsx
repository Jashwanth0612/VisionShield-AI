import { useMemo, useState } from 'react'
import VideoAnalyzer from './components/VideoAnalyzer'

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

function Metric({ label, value, unit = '' }) {
  return <div className="metric"><span>{label}</span><strong>{value}{unit}</strong></div>
}

export default function App() {
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState('')
  const [result, setResult] = useState(null)
  const [enhance, setEnhance] = useState(true)
  const [confidence, setConfidence] = useState(0.35)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const detectionSummary = useMemo(() => {
    if (!result?.detections) return []
    const counts = {}
    result.detections.forEach((item) => { counts[item.label] = (counts[item.label] || 0) + 1 })
    return Object.entries(counts).sort((a, b) => b[1] - a[1])
  }, [result])

  function chooseFile(next) {
    if (!next) return
    setFile(next); setResult(null); setError(''); setPreview(URL.createObjectURL(next))
  }

  async function runPipeline() {
    if (!file) return
    setLoading(true); setError('')
    try {
      const body = new FormData(); body.append('file', file)
      const response = await fetch(`${API_URL}/pipeline/process?enable_enhancement=${enhance}&confidence=${confidence}`, { method: 'POST', body })
      const data = await response.json()
      if (!response.ok) throw new Error(data.detail || 'Inference failed')
      setResult(data)
    } catch (err) { setError(`${err.message}. Make sure the FastAPI backend is running on ${API_URL}.`) }
    finally { setLoading(false) }
  }

  return (
    <div className="app-shell">
      <header className="topbar"><div className="brand"><div className="brand-mark">V</div><div><b>VisionShield</b><small>AI PERCEPTION PLATFORM</small></div></div><div className="status-pill"><i /> SYSTEM READY</div></header>
      <main>
        <section className="hero"><div><p className="eyebrow">ALL-WEATHER COMPUTER VISION</p><h1>See clearly.<br /><span>Detect confidently.</span></h1><p className="hero-copy">Enhance degraded imagery with NAFNet and run transformer-based RT-DETR detection across images and sampled video frames.</p></div><div className="hero-card"><div className="pulse" /><span>PIPELINE</span><strong>NAFNet → RT-DETR</strong><small>Enhancement + detection</small></div></section>
        <section className="workspace">
          <div className="panel upload-panel"><div className="panel-head"><div><span className="section-kicker">01 / INPUT</span><h2>Inference workspace</h2></div><span className="model-chip">RT-DETR</span></div>
            <label className="dropzone" onDragOver={(e) => e.preventDefault()} onDrop={(e) => { e.preventDefault(); chooseFile(e.dataTransfer.files?.[0]) }}><input type="file" accept="image/*" onChange={(e) => chooseFile(e.target.files?.[0])} />{preview ? <img src={preview} alt="Input preview" /> : <><div className="upload-icon">↑</div><strong>Drop an image here</strong><span>or click to browse · JPG, PNG, WEBP · max 20 MB</span></>}</label>
            <div className="controls"><label className="toggle-row"><span><b>NAFNet enhancement</b><small>Restore degraded visual conditions before detection</small></span><button className={enhance ? 'toggle active' : 'toggle'} onClick={() => setEnhance(!enhance)}><i /></button></label><label className="range-row"><span><b>Detection confidence</b><strong>{Math.round(confidence * 100)}%</strong></span><input type="range" min="0.05" max="0.95" step="0.05" value={confidence} onChange={(e) => setConfidence(Number(e.target.value))} /></label></div>
            <button className="run-btn" disabled={!file || loading} onClick={runPipeline}>{loading ? 'Running inference…' : 'Run VisionShield Pipeline →'}</button>{error && <div className="error">{error}</div>}
          </div>
          <div className="panel result-panel"><div className="panel-head"><div><span className="section-kicker">02 / OUTPUT</span><h2>Detection result</h2></div>{result && <span className="success-chip">● COMPLETE</span>}</div><div className="result-stage">{result ? <img src={result.annotated_image} alt="Annotated detection result" /> : <div className="empty-result"><div className="crosshair">＋</div><strong>Awaiting inference</strong><span>Your annotated detection output will appear here.</span></div>}</div>{result && <div className="metrics"><Metric label="Objects" value={result.detections_count} /><Metric label="Total latency" value={result.metrics.total_latency_ms} unit=" ms" /><Metric label="FPS equiv." value={result.metrics.fps_equivalent} /><Metric label="Confidence" value={`${Math.round(confidence * 100)}%`} /></div>}</div>
        </section>
        {result && <section className="analysis-grid"><div className="panel compare"><div className="panel-head"><div><span className="section-kicker">03 / ANALYSIS</span><h2>Before / enhanced</h2></div></div><div className="image-grid"><div><span>INPUT</span><img src={preview} alt="Input" /></div><div><span>ENHANCED</span><img src={result.enhanced_image} alt="Enhanced" /></div></div></div><div className="panel detections"><div className="panel-head"><div><span className="section-kicker">DETECTIONS</span><h2>Objects found</h2></div></div>{detectionSummary.length ? detectionSummary.map(([label, count]) => <div className="det-row" key={label}><span>{label}</span><b>{count}</b></div>) : <p className="muted">No objects above the selected confidence threshold.</p>}</div></section>}
        <VideoAnalyzer />
      </main>
      <footer><span>VisionShield AI</span><span>NAFNet × RT-DETR</span><span>Computer Vision Research Platform</span></footer>
    </div>
  )
}
