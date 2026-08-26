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

const NAFNET_CONDITIONS = [
  ['fog_its', 'Fog · ITS'],
  ['fog_ots', 'Fog · OTS'],
  ['rain', 'Rain'],
  ['snow', 'Snow'],
  ['low_light', 'Low-Light'],
]

export function ModelHealth({ health }) {
  const models = health?.models || {}
  const nafnet = models.nafnet || {}
  const rtDetr = models.rt_detr || {}
  const conditionStates = nafnet.conditions || {}
  const readyConditions = NAFNET_CONDITIONS.filter(([id]) => conditionStates[id]?.checkpoint_exists).length

  return <div className="model-health">
    <div className="health-item health-item-wide">
      <span>NAFNet / condition checkpoints</span>
      <strong className={readyConditions === NAFNET_CONDITIONS.length ? 'ready' : 'unavailable'}>{readyConditions}/{NAFNET_CONDITIONS.length} CONFIGURED</strong>
      <div className="condition-grid">{NAFNET_CONDITIONS.map(([id, label]) => {
        const condition = conditionStates[id]
        const ready = Boolean(condition?.checkpoint_exists)
        return <div className={`condition-chip ${ready ? 'ready' : 'unavailable'}`} key={id} title={condition?.error || condition?.checkpoint || `Configure ${label} checkpoint`}><i />{label}</div>
      })}</div>
    </div>
    <div className="health-item">
      <span>RT-DETR</span>
      <strong className={rtDetr.loaded ? 'ready' : 'unavailable'}>{rtDetr.loaded ? 'READY' : 'UNAVAILABLE'}</strong>
      <small>{rtDetr.error || (rtDetr.checkpoint ? rtDetr.checkpoint : 'Configure RTDETR_WEIGHTS_PATH')}</small>
    </div>
    <div className="health-item">
      <span>Inference device</span>
      <strong>{rtDetr.device || nafnet.device || '—'}</strong>
      <small>NAFNet and RT-DETR execute from the configured backend runtime.</small>
    </div>
  </div>
}

export function ArtifactLinks({ artifacts }) {
  if (!artifacts) return null
  return <div className="artifact-links">{Object.entries(artifacts).filter(([, value]) => value).map(([name, value]) => <a key={name} href={`${api.baseUrl}${value.url}`} target="_blank" rel="noreferrer">Open {name} ↗</a>)}</div>
}

export function EmptyState({ title, detail }) {
  return <div className="empty-state"><div className="empty-mark">＋</div><strong>{title}</strong><span>{detail}</span></div>
}
