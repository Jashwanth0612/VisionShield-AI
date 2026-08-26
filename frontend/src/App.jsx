import { useEffect, useState } from 'react'
import { api } from './api'
import Workspace from './components/Workspace'
import VideoIntelligence from './components/VideoIntelligence'
import History from './components/History'
import Benchmarks from './components/Benchmarks'

const nav = [
  ['workspace', 'Workspace', '◎'],
  ['video', 'Video intelligence', '◫'],
  ['history', 'Inference history', '≡'],
  ['benchmarks', 'Benchmarks', '↗'],
]

export default function App() {
  const [active, setActive] = useState('workspace')
  const [health, setHealth] = useState(null)
  const [mobileOpen, setMobileOpen] = useState(false)

  const refreshHealth = () => api.health().then(setHealth).catch(() => setHealth(null))
  useEffect(() => { refreshHealth() }, [])

  const title = nav.find(([id]) => id === active)?.[1] || 'Workspace'
  const modelsReady = health?.status === 'operational'

  return <div className="app-shell">
    <aside className={mobileOpen ? 'open' : ''}>
      <div className="brand"><div className="brand-mark">V</div><div><b>VISION<span>SHIELD</span></b><small>AI PERCEPTION SYSTEMS</small></div></div>
      <div className="side-label">CONSOLE</div>
      <nav>{nav.map(([id, label, symbol]) => <button className={active === id ? 'selected' : ''} key={id} onClick={() => { setActive(id); setMobileOpen(false) }}><span>{symbol}</span>{label}<i>›</i></button>)}</nav>
      <div className="side-footer"><span className={`connection ${modelsReady ? 'ready' : ''}`}><i />{modelsReady ? 'Models ready' : 'Model runtime unavailable'}</span><small>v2.0.0 · production console</small></div>
    </aside>
    <main>
      <header><button className="mobile-menu" onClick={() => setMobileOpen(!mobileOpen)}>☰</button><div><span className="breadcrumb">CONSOLE <b>/</b> {title.toUpperCase()}</span><small>Perception operations center</small></div><div className="header-status"><span className={`connection ${health?.api_status === 'connected' ? 'ready' : ''}`}><i />{health?.api_status === 'connected' ? 'API connected' : 'API unavailable'}</span><button className="avatar" onClick={refreshHealth}>VS</button></div></header>
      <div className="content"><div className="runtime-strip"><span><b>RUNTIME</b> {modelsReady ? 'NAFNet + RT-DETR available' : 'Configure model weights to enable inference'}</span><span>{health ? `${health.total_inferences || 0} persisted inference runs` : 'Health unavailable'}</span></div>
        {active === 'workspace' && <Workspace health={health} onRefresh={refreshHealth} />}
        {active === 'video' && <VideoIntelligence />}
        {active === 'history' && <History />}
        {active === 'benchmarks' && <Benchmarks />}
      </div>
    </main>
  </div>
}
