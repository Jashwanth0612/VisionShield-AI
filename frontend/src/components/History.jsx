import { useEffect, useState } from 'react'
import { api } from '../api'
import { ArtifactLinks, EmptyState, Panel } from './Shared'

export default function History() {
  const [items, setItems] = useState([])
  const [search, setSearch] = useState('')
  const [mediaType, setMediaType] = useState('all')
  const [selected, setSelected] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')

  const load = async () => {
    setLoading(true); setError('')
    try { setItems(await api.history({ search, mediaType })) } catch (err) { setError(err.message || 'History service unavailable.') }
    finally { setLoading(false) }
  }

  useEffect(() => { load() }, [mediaType])

  return <div className="page">
    <div className="section-intro"><span className="eyebrow">OPERATIONS / PERSISTED RUNS</span><h1>Every run,<br /><span>traceable.</span></h1><p>Successful image and video inference runs are persisted separately from benchmark actions. Failed or unavailable runs are never fabricated into history.</p></div>
    <Panel eyebrow="IMAGE + VIDEO / HISTORY" title="Inference history" action={<span className="muted">{items.length} visible</span>}>
      <div className="toolbar"><div className="search"><span>⌕</span><input value={search} onChange={(e) => setSearch(e.target.value)} onKeyDown={(e) => e.key === 'Enter' && load()} placeholder="Search filename" /></div><div className="filter-tabs">{['all', 'image', 'video'].map((type) => <button className={mediaType === type ? 'active' : ''} onClick={() => setMediaType(type)} key={type}>{type}</button>)}</div><button className="secondary" onClick={load}>Search</button></div>
      {error && <div className="error-box"><b>History unavailable</b><span>{error}</span></div>}
      {loading ? <EmptyState title="Loading persisted runs" detail="Reading the configured metadata store." /> : items.length ? <div className="history-table"><div className="history-header"><span>Run</span><span>Type / time</span><span>NAFNet</span><span>Objects</span><span>Latency</span><span>FPS</span></div>{items.map((item) => <button className="history-row" key={item.run_id} onClick={() => api.historyItem(item.run_id).then(setSelected)}><span><b>{item.filename}</b><small>{item.run_id}</small></span><span><b>{item.media_type}</b><small>{new Date(item.timestamp).toLocaleString()}</small></span><span>{item.nafnet_enabled ? 'ON' : 'OFF'}</span><span>{item.detections}</span><span>{item.latency_ms} ms</span><span>{item.fps}</span></button>)}</div> : <EmptyState title="No inference history" detail="Successful measured runs will appear here after the real model pipeline completes." />}
    </Panel>

    {selected && <Panel eyebrow="RUN DETAIL" title={selected.run_id} action={<button className="ghost" onClick={() => setSelected(null)}>Close</button>}>
      <div className="detail-grid">{Object.entries(selected).filter(([key]) => !['artifacts', 'details'].includes(key)).map(([key, value]) => <div key={key}><span>{key.replaceAll('_', ' ')}</span><b>{String(value)}</b></div>)}</div>
      <ArtifactLinks artifacts={selected.artifacts} />
      {selected.details?.detections_detail && <div className="detail-detections"><h3>Detection detail</h3>{selected.details.detections_detail.map((item, index) => <span key={index}>{item.label} · {(item.confidence * 100).toFixed(1)}%</span>)}</div>}
    </Panel>}
  </div>
}
