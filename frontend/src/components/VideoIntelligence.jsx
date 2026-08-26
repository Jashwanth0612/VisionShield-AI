import { useEffect, useState } from 'react'
import { api } from '../api'
import { ArtifactLinks, Dropzone, EmptyState, Metric, Panel } from './Shared'

const WEATHER_OPTIONS = [
  ['auto', 'Auto detect'],
  ['fog_its', 'Fog · ITS'],
  ['fog_ots', 'Fog · OTS'],
  ['rain', 'Rain'],
  ['snow', 'Snow'],
  ['low_light', 'Low-Light'],
]

export default function VideoIntelligence() {
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState('')
  const [enhancement, setEnhancement] = useState(true)
  const [weather, setWeather] = useState('auto')
  const [sampleFps, setSampleFps] = useState(2)
  const [confidence, setConfidence] = useState(.35)
  const [result, setResult] = useState(null)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)

  useEffect(() => () => { if (preview) URL.revokeObjectURL(preview) }, [preview])

  const chooseFile = (next) => {
    if (!next) return
    if (!next.type.startsWith('video/')) { setError('Please choose a supported video file.'); return }
    if (preview) URL.revokeObjectURL(preview)
    setFile(next); setPreview(URL.createObjectURL(next)); setResult(null); setError('')
  }

  const run = async () => {
    if (!file) return
    setLoading(true); setError('')
    try { const response = await api.analyzeVideo({ file, enhancement, sampleFps, confidence, weather }); setResult(response.result) }
    catch (err) { setResult(null); setError(err.message || 'Video analysis failed.') }
    finally { setLoading(false) }
  }

  const annotatedUrl = result?.artifacts?.annotated ? `${api.baseUrl}${result.artifacts.annotated.url}` : ''
  const enhancedUrl = result?.artifacts?.enhanced ? `${api.baseUrl}${result.artifacts.enhanced.url}` : ''

  return <div className="page">
    <div className="section-intro"><span className="eyebrow">VIDEO INTELLIGENCE / SAMPLED ANALYSIS</span><h1>Read the scene<br /><span>frame by frame.</span></h1><p>Every sampled frame is processed by the real configured pipeline. This is sampled analysis, not persistent object tracking.</p></div>
    <div className="two-col">
      <Panel eyebrow="01 / SEQUENCE INPUT" title="Video source">
        <Dropzone file={file} onFile={chooseFile} accept="video/mp4,video/quicktime,video/webm,video/x-msvideo,video/mpeg" label="Drop a video here" hint="MP4, MOV, WebM, AVI, MPEG · max 100 MB · max 120 sec" />
        {preview && <video className="source-video" src={preview} controls muted playsInline />}
        <div className="control-list">
          <div className="control-row"><div><b>Weather route</b><small>Use Auto or select the matching NAFNet condition</small></div><select className="weather-select" value={weather} onChange={(e) => setWeather(e.target.value)}>{WEATHER_OPTIONS.map(([value, label]) => <option key={value} value={value}>{label}</option>)}</select></div>
          <div className="control-row"><div><b>NAFNet enhancement</b><small>Apply restoration before sampled detection</small></div><button className={`toggle ${enhancement ? 'active' : ''}`} onClick={() => setEnhancement(!enhancement)}><i /></button></div>
          <div className="range-row"><div><b>Sampling rate</b><strong>{sampleFps} fps</strong></div><input type="range" min=".5" max="10" step=".5" value={sampleFps} onChange={(e) => setSampleFps(Number(e.target.value))} /></div>
          <div className="range-row"><div><b>Confidence</b><strong>{Math.round(confidence * 100)}%</strong></div><input type="range" min=".05" max=".95" step=".01" value={confidence} onChange={(e) => setConfidence(Number(e.target.value))} /></div>
        </div>
        <button className="primary" disabled={!file || loading} onClick={run}>{loading ? 'Analyzing sampled frames…' : 'Run video analysis →'}</button>
        {error && <div className="error-box"><b>Video analysis unavailable</b><span>{error}</span></div>}
      </Panel>
      <Panel eyebrow="02 / MEASURED OUTPUT" title="Sequence report">
        {result ? <><div className="metric-grid"><Metric label="Frames analyzed" value={result.frames_analyzed} tone="accent" /><Metric label="Analysis FPS" value={result.analysis_fps} /><Metric label="Mean inference" value={result.inference_latency_ms} unit="ms" /><Metric label="Detections" value={result.detections_total} /></div><div className="video-meta"><span>Source {result.source_fps} fps</span><span>Sampled {result.sample_fps} fps</span><span>Duration {result.duration_seconds}s</span><span>{result.weather?.label || 'Weather route unavailable'}</span></div><div className="class-bars">{Object.entries(result.detected_classes || {}).map(([label, count]) => <div key={label}><span>{label}</span><div><i style={{ width: `${Math.min(100, count / Math.max(result.detections_total, 1) * 100)}%` }} /></div><b>{count}</b></div>)}</div><div className="artifact-viewer">{annotatedUrl && <div><span>ANNOTATED SAMPLED RESULT</span><video src={annotatedUrl} controls playsInline /></div>}{enhancedUrl && <div><span>ENHANCED SAMPLED RESULT</span><video src={enhancedUrl} controls playsInline /></div>}</div><ArtifactLinks artifacts={result.artifacts} /></> : <EmptyState title="No measured sequence report" detail="Upload a video and run analysis after the model runtime is available." />}
      </Panel>
    </div>
    <Panel eyebrow="IMPORTANT" title="Interpretation"><div className="notice"><span>TRACKING</span><p>The current workflow samples independent frames. Counts and detections are measured per sampled frame; persistent object identity/tracking is not claimed.</p></div></Panel>
  </div>
}
