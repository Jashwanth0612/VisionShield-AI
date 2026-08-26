import { useState } from 'react'

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

export default function VideoAnalyzer() {
  const [file, setFile] = useState(null)
  const [result, setResult] = useState(null)
  const [fps, setFps] = useState(2)
  const [confidence, setConfidence] = useState(0.35)
  const [enhance, setEnhance] = useState(true)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  async function analyze() {
    if (!file) return
    setLoading(true)
    setError('')
    try {
      const body = new FormData()
      body.append('file', file)
      const url = `${API_URL}/video/analyze?enable_enhancement=${enhance}&sample_fps=${fps}&confidence=${confidence}`
      const response = await fetch(url, { method: 'POST', body })
      const data = await response.json()
      if (!response.ok) throw new Error(data.detail || 'Video analysis failed')
      setResult(data)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <section className="video-panel">
      <div className="panel-head"><div><span className="section-kicker">04 / VIDEO ANALYSIS</span><h2>Temporal perception</h2></div><span className="model-chip">RT-DETR</span></div>
      <label className="video-dropzone">
        <input type="file" accept="video/mp4,video/quicktime,video/webm,video/x-msvideo,video/mpeg" onChange={(e) => { setFile(e.target.files?.[0] || null); setResult(null) }} />
        <strong>{file ? file.name : 'Choose a video for frame-sampled analysis'}</strong>
        <span>MP4, MOV, WebM, AVI, MPEG · max 100 MB · max 120 seconds</span>
      </label>
      <div className="video-controls">
        <label>Sampling FPS <strong>{fps}</strong><input type="range" min="0.5" max="10" step="0.5" value={fps} onChange={(e) => setFps(Number(e.target.value))} /></label>
        <label>Confidence <strong>{Math.round(confidence * 100)}%</strong><input type="range" min="0.05" max="0.95" step="0.05" value={confidence} onChange={(e) => setConfidence(Number(e.target.value))} /></label>
        <label className="check"><input type="checkbox" checked={enhance} onChange={(e) => setEnhance(e.target.checked)} /> NAFNet enhancement</label>
      </div>
      <button className="run-btn" disabled={!file || loading} onClick={analyze}>{loading ? 'Analyzing video…' : 'Analyze Video →'}</button>
      {error && <div className="error">{error}</div>}
      {result && <div className="video-results">
        <div className="metrics"><div className="metric"><span>Frames sampled</span><strong>{result.frames_analyzed}</strong></div><div className="metric"><span>Objects</span><strong>{result.total_detections}</strong></div><div className="metric"><span>Latency</span><strong>{result.latency_ms} ms</strong></div><div className="metric"><span>Analysis FPS</span><strong>{result.analysis_fps}</strong></div></div>
        <div className="class-list">{Object.entries(result.class_frequency || {}).map(([label, count]) => <div className="det-row" key={label}><span>{label}</span><b>{count}</b></div>)}</div>
        <small className="muted">Frames are sampled independently; this result does not claim persistent object tracking.</small>
      </div>}
    </section>
  )
}
