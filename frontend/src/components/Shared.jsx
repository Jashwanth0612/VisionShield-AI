import { useRef, useState } from 'react'
import { api } from '../api'

export const icon = (symbol) => <span className="icon-glyph" aria-hidden="true">{symbol}</span>

export function Panel({ eyebrow, title, action, children, className = '' }) {
  return <section className={`panel ${className}`}>
    {(eyebrow || title || action) && <div className="panel-head">
      <div><span className="eyebrow">{eyebrow}</span>{title && <h2>{title}</h2>}</div>
      {action}
    </div>}
    {children}
  </section>
}

export function Metric({ label, value, unit = '', tone = '' }) {
  return <div className={`metric ${tone}`}><span>{label}</span><strong>{value ?? '—'}{value !== undefined && value !== null && unit ? <em>{unit}</em> : null}</strong></div>
}

export function Dropzone({ file, onFile, accept, label, hint }) {
  const ref = useRef(null)
  const [dragging, setDragging] = useState(false)
  return <div className={`dropzone ${dragging ? 'dragging' : ''}`} onClick={() => ref.current?.click()} onDragOver={(event) => { event.preventDefault(); setDragging(true) }} onDragLeave={() => setDragging(false)} onDrop={(event) => { event.preventDefault(); setDragging(false); onFile(event.dataTransfer.files?.[0]) }}>
    <input ref={ref} hidden type="file" accept={accept} onChange={(event) => onFile(event.target.files?.[0])} />
    <div className="drop-icon">↥</div>
    <strong>{file?.name || label}</strong>
    <span>{file ? `${(file.size / 1024 / 1024).toFixed(2)} MB` : hint}</span>
  </div>
}

export function StatusPill({ good, children }) {
  return <span className={`status-pill ${good ? 'good' : 'warn'}`}><i />{children}</span>
}

export function ModelHealth({ health }) {
  const models = health?.models || {}
  return <div className="model-health">
    {['nafnet', 'rt_detr'].map((name) => {
      const model = models[name]
      const label = name === 'rt_detr' ? 'RT-DETR' : 'NAFNet'
      return <div className="health-item" key={name}>
        <span>{label}</span>
        <strong className={model?.loaded ? 'ready' : 'unavailable'}>{model?.loaded ? 'READY' : 'UNAVAILABLE'}</strong>
        <small>{model?.error || (model?.checkpoint ? model.checkpoint : `Configure ${label} weights`)}</small>
      </div>
    })}
  </div>
}

export function ArtifactLinks({ artifacts }) {
  if (!artifacts) return null
  return <div className="artifact-links">{Object.entries(artifacts).filter(([, value]) => value).map(([name, value]) => <a key={name} href={`${api.baseUrl}${value.url}`} target="_blank" rel="noreferrer">Open {name} ↗</a>)}</div>
}

export function EmptyState({ title, detail }) {
  return <div className="empty-state"><div className="empty-mark">＋</div><strong>{title}</strong><span>{detail}</span></div>
}
