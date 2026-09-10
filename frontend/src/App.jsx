import { useEffect, useMemo, useState } from 'react'
import { api } from './api'
import Workspace from './components/Workspace'
import VideoIntelligence from './components/VideoIntelligence'
import History from './components/History'
import Benchmarks from './components/Benchmarks'

const SITE_NAME = 'VisionShield AI'
const SITE_DESCRIPTION = 'All-weather visual perception using condition-aware NAFNet restoration and RT-DETRv2 object detection.'
const SITE_URL = (import.meta.env.VITE_SITE_URL || window.location.origin).replace(/\/$/, '')

const pages = {
  workspace: {
    path: '/',
    title: 'VisionShield AI | All-Weather Visual Perception',
    description: 'Run all-weather image perception with weather-routed NAFNet restoration and RT-DETRv2 detection.',
    label: 'Workspace',
  },
  video: {
    path: '/video',
    title: 'Video Intelligence | VisionShield AI',
    description: 'Analyze sampled video frames with the VisionShield all-weather restoration and detection pipeline.',
    label: 'Video intelligence',
  },
  history: {
    path: '/history',
    title: 'Inference History | VisionShield AI',
    description: 'Review persisted VisionShield inference runs, measured latency, detections and model routing.',
    label: 'Inference history',
  },
  benchmarks: {
    path: '/benchmarks',
    title: 'Benchmarks | VisionShield AI',
    description: 'Review VisionShield model benchmark measurements and runtime evaluation results.',
    label: 'Benchmarks',
  },
}

const nav = [
  ['workspace', 'Workspace', '◎'],
  ['video', 'Video intelligence', '◫'],
  ['history', 'Inference history', '≡'],
  ['benchmarks', 'Benchmarks', '↗'],
]

function routeFromPath(pathname) {
  const match = Object.entries(pages).find(([, page]) => page.path === pathname)
  return match?.[0] || null
}

function applyMetadata(page, active) {
  document.title = page.title
  const setMeta = (name, content) => {
    let node = document.querySelector(`meta[name="${name}"]`)
    if (!node) { node = document.createElement('meta'); node.name = name; document.head.appendChild(node) }
    node.content = content
  }
  setMeta('description', page.description)
  setMeta('theme-color', '#f4efe5')
  setMeta('application-name', SITE_NAME)
  setMeta('robots', 'index,follow,max-image-preview:large')

  const canonical = `${SITE_URL}${page.path === '/' ? '/' : page.path}`
  let canonicalNode = document.querySelector('link[rel="canonical"]')
  if (!canonicalNode) { canonicalNode = document.createElement('link'); canonicalNode.rel = 'canonical'; document.head.appendChild(canonicalNode) }
  canonicalNode.href = canonical

  const social = {
    'og:title': page.title,
    'og:description': page.description,
    'og:type': 'website',
    'og:url': canonical,
    'og:site_name': SITE_NAME,
    'og:image': `${SITE_URL}/social-card.svg`,
    'twitter:card': 'summary_large_image',
    'twitter:title': page.title,
    'twitter:description': page.description,
    'twitter:image': `${SITE_URL}/social-card.svg`,
  }
  Object.entries(social).forEach(([property, content]) => {
    const attr = property.startsWith('og:') ? 'property' : 'name'
    let node = document.querySelector(`meta[${attr}="${property}"]`)
    if (!node) { node = document.createElement('meta'); node.setAttribute(attr, property); document.head.appendChild(node) }
    node.content = content
  })

  const data = {
    '@context': 'https://schema.org',
    '@graph': [
      { '@type': 'Organization', '@id': `${SITE_URL}/#organization`, name: SITE_NAME, url: SITE_URL },
      { '@type': 'WebApplication', '@id': `${canonical}#application`, name: page.title, url: canonical, description: page.description, applicationCategory: 'ComputerVisionApplication', operatingSystem: 'Web' },
      { '@type': 'BreadcrumbList', itemListElement: [{ '@type': 'ListItem', position: 1, name: 'VisionShield AI', item: `${SITE_URL}/` }, ...(active !== 'workspace' ? [{ '@type': 'ListItem', position: 2, name: page.label, item: canonical }] : [])] },
    ],
  }
  let jsonLd = document.getElementById('visionshield-structured-data')
  if (!jsonLd) { jsonLd = document.createElement('script'); jsonLd.id = 'visionshield-structured-data'; jsonLd.type = 'application/ld+json'; document.head.appendChild(jsonLd) }
  jsonLd.textContent = JSON.stringify(data)
}

function Breadcrumbs({ page }) {
  return <nav className="breadcrumb-nav" aria-label="Breadcrumb">
    <a href="/" onClick={(e) => { if (window.location.origin === SITE_URL) { e.preventDefault(); window.history.pushState({}, '', '/'); window.dispatchEvent(new PopStateEvent('popstate')) } }}>VisionShield AI</a>
    {page.path !== '/' && <><span aria-hidden="true">/</span><span aria-current="page">{page.label}</span></>}
  </nav>
}

function NotFound() {
  useEffect(() => {
    const title = 'Page Not Found | VisionShield AI'
    document.title = title
    let description = document.querySelector('meta[name="description"]')
    if (!description) { description = document.createElement('meta'); description.name = 'description'; document.head.appendChild(description) }
    description.content = 'The requested VisionShield AI page could not be found.'
  }, [])
  return <div className="not-found">
    <span className="eyebrow">404 / ROUTE NOT FOUND</span>
    <h1>That page<br /><span>does not exist.</span></h1>
    <p>The requested VisionShield AI route is not available. Return to the perception workspace to continue.</p>
    <a className="primary" href="/">Return to workspace →</a>
  </div>
}

export default function App() {
  const [active, setActive] = useState(() => routeFromPath(window.location.pathname))
  const [health, setHealth] = useState(null)
  const [mobileOpen, setMobileOpen] = useState(false)

  const refreshHealth = () => api.health().then(setHealth).catch(() => setHealth(null))

  useEffect(() => {
    const onPopState = () => setActive(routeFromPath(window.location.pathname))
    window.addEventListener('popstate', onPopState)
    return () => window.removeEventListener('popstate', onPopState)
  }, [])

  useEffect(() => { refreshHealth() }, [])

  const page = active ? pages[active] : null
  useEffect(() => { if (page && active) applyMetadata(page, active) }, [page, active])

  const modelsReady = health?.status === 'operational'
  const title = page?.label || 'Not found'

  const navigate = (event, id) => {
    event.preventDefault()
    const next = pages[id]
    window.history.pushState({}, '', next.path)
    setActive(id)
    setMobileOpen(false)
  }

  if (!page) return <div className="app-shell"><main><div className="content"><NotFound /></div></main></div>

  return <div className="app-shell">
    <aside className={mobileOpen ? 'open' : ''}>
      <div className="brand"><div className="brand-mark" aria-hidden="true">V</div><div><b>VISION<span>SHIELD</span></b><small>AI PERCEPTION SYSTEMS</small></div></div>
      <div className="side-label">CONSOLE</div>
      <nav aria-label="Primary navigation">{nav.map(([id, label, symbol]) => <a className={active === id ? 'selected' : ''} href={pages[id].path} key={id} onClick={(event) => navigate(event, id)}><span aria-hidden="true">{symbol}</span>{label}<i aria-hidden="true">›</i></a>)}</nav>
      <div className="side-footer"><span className={`connection ${modelsReady ? 'ready' : ''}`}><i />{modelsReady ? 'Models ready' : 'Model runtime unavailable'}</span><small>Production perception console</small></div>
    </aside>
    <main>
      <header><button className="mobile-menu" aria-label="Open navigation" onClick={() => setMobileOpen(!mobileOpen)}>☰</button><div><span className="breadcrumb">CONSOLE <b>/</b> {title.toUpperCase()}</span><Breadcrumbs page={page} /></div><div className="header-status"><span className={`connection ${health?.api_status === 'connected' ? 'ready' : ''}`}><i />{health?.api_status === 'connected' ? 'API connected' : 'API unavailable'}</span><button className="avatar" aria-label="Refresh system status" onClick={refreshHealth}>VS</button></div></header>
      <div className="content">
        <div className="runtime-strip"><span><b>RUNTIME</b> {modelsReady ? 'NAFNet + RT-DETR available' : 'Configure model weights to enable inference'}</span><span>{health ? `${health.total_inferences || 0} persisted inference runs` : 'Health unavailable'}</span></div>
        {active === 'workspace' && <Workspace health={health} onRefresh={refreshHealth} />}
        {active === 'video' && <VideoIntelligence />}
        {active === 'history' && <History />}
        {active === 'benchmarks' && <Benchmarks />}
        <footer className="site-footer"><span>VisionShield AI · All-weather visual perception</span><nav aria-label="Footer navigation">{nav.map(([id, label]) => <a href={pages[id].path} key={id} onClick={(event) => navigate(event, id)}>{label}</a>)}</nav></footer>
      </div>
    </main>
  </div>
}
