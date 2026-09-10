import { lazy, Suspense, useEffect, useState } from 'react'
import { api } from './api'

const Workspace = lazy(() => import('./components/Workspace'))
const VideoIntelligence = lazy(() => import('./components/VideoIntelligence'))
const History = lazy(() => import('./components/History'))
const Benchmarks = lazy(() => import('./components/Benchmarks'))

const SITE_NAME = 'VisionShield AI'
const SITE_URL = (import.meta.env.VITE_SITE_URL || window.location.origin).replace(/\/$/, '')

const pages = {
  workspace: { path: '/', title: 'VisionShield AI | All-Weather Visual Perception', description: 'Run all-weather image perception with weather-routed NAFNet restoration and RT-DETRv2 detection.', label: 'Workspace' },
  video: { path: '/video', title: 'Video Intelligence | VisionShield AI', description: 'Analyze sampled video frames with the VisionShield all-weather restoration and detection pipeline.', label: 'Video intelligence' },
  history: { path: '/history', title: 'Inference History | VisionShield AI', description: 'Review persisted VisionShield inference runs, measured latency, detections and model routing.', label: 'Inference history' },
  benchmarks: { path: '/benchmarks', title: 'Benchmarks | VisionShield AI', description: 'Review VisionShield model benchmark measurements and runtime evaluation results.', label: 'Benchmarks' },
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

function upsertMeta(attribute, key, content) {
  let node = document.querySelector(`meta[${attribute}="${key}"]`)
  if (!node) { node = document.createElement('meta'); node.setAttribute(attribute, key); document.head.appendChild(node) }
  node.content = content
}

function applyMetadata(page, active) {
  document.title = page.title
  upsertMeta('name', 'description', page.description)
  upsertMeta('name', 'theme-color', '#f4efe5')
  upsertMeta('name', 'application-name', SITE_NAME)
  upsertMeta('name', 'robots', 'index,follow,max-image-preview:large')

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
  Object.entries(social).forEach(([key, content]) => upsertMeta(key.startsWith('og:') ? 'property' : 'name', key, content))

  const data = {
    '@context': 'https://schema.org',
    '@graph': [
      { '@type': 'Organization', '@id': `${SITE_URL}/#organization`, name: SITE_NAME, url: SITE_URL },
      { '@type': 'WebApplication', '@id': `${canonical}#application`, name: page.title, url: canonical, description: page.description, applicationCategory: 'ComputerVisionApplication', operatingSystem: 'Web' },
      { '@type': 'BreadcrumbList', itemListElement: [{ '@type': 'ListItem', position: 1, name: SITE_NAME, item: `${SITE_URL}/` }, ...(active !== 'workspace' ? [{ '@type': 'ListItem', position: 2, name: page.label, item: canonical }] : [])] },
    ],
  }
  let jsonLd = document.getElementById('visionshield-structured-data')
  if (!jsonLd) { jsonLd = document.createElement('script'); jsonLd.id = 'visionshield-structured-data'; jsonLd.type = 'application/ld+json'; document.head.appendChild(jsonLd) }
  jsonLd.textContent = JSON.stringify(data)
}

function Breadcrumbs({ page }) {
  return <nav className="breadcrumb-nav" aria-label="Breadcrumb">
    <a href="/" onClick={(e) => { e.preventDefault(); window.history.pushState({}, '', '/'); window.dispatchEvent(new PopStateEvent('popstate')) }}>VisionShield AI</a>
    {page.path !== '/' && <><span aria-hidden="true">/</span><span aria-current="page">{page.label}</span></>}
  </nav>
}

function LoadingPage() {
  return <div className="page"><div className="empty-state" role="status" aria-live="polite"><div className="empty-mark" aria-hidden="true">V</div><strong>Loading workspace</strong><span>Preparing the selected perception module.</span></div></div>
}

function NotFound() {
  useEffect(() => {
    document.title = 'Page Not Found | VisionShield AI'
    upsertMeta('name', 'description', 'The requested VisionShield AI page could not be found.')
    upsertMeta('name', 'robots', 'noindex,follow')
    let canonical = document.querySelector('link[rel="canonical"]')
    if (!canonical) { canonical = document.createElement('link'); canonical.rel = 'canonical'; document.head.appendChild(canonical) }
    canonical.href = `${SITE_URL}/`
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

  // The backend loads checkpoints lazily. A configured Drive-backed runtime is
  // therefore available for inference even before the first model has loaded.
  const modelsReady = Boolean(health?.models_loaded || health?.status === 'healthy' || health?.status === 'configured')
  const title = page?.label || 'Not found'

  const navigate = (event, id) => {
    event.preventDefault()
    window.history.pushState({}, '', pages[id].path)
    setActive(id)
    setMobileOpen(false)
  }

  if (!page) return <div className="app-shell"><main><div className="content"><NotFound /></div></main></div>

  return <div className="app-shell">
    <aside className={mobileOpen ? 'open' : ''}>
      <div className="brand"><div className="brand-mark" aria-hidden="true">V</div><div><b>VISION<span>SHIELD</span></b><small>AI PERCEPTION SYSTEMS</small></div></div>
      <div className="side-label">CONSOLE</div>
      <nav aria-label="Primary navigation">{nav.map(([id, label, symbol]) => <a className={active === id ? 'selected' : ''} href={pages[id].path} key={id} onClick={(event) => navigate(event, id)}><span aria-hidden="true">{symbol}</span>{label}<i aria-hidden="true">›</i></a>)}</nav>
      <div className="side-footer"><span className={`connection ${modelsReady ? 'ready' : ''}`}><i />{modelsReady ? 'Model runtime available' : 'Model runtime unavailable'}</span><small>Production perception console</small></div>
    </aside>
    <main>
      <header><button className="mobile-menu" aria-label="Open navigation" onClick={() => setMobileOpen(!mobileOpen)}>☰</button><div><span className="breadcrumb">CONSOLE <b>/</b> {title.toUpperCase()}</span><Breadcrumbs page={page} /></div><div className="header-status"><span className={`connection ${health?.api_status === 'connected' ? 'ready' : ''}`}><i />{health?.api_status === 'connected' ? 'API connected' : 'API unavailable'}</span><button className="avatar" aria-label="Refresh system status" onClick={refreshHealth}>VS</button></div></header>
      <div className="content">
        <div className="runtime-strip"><span><b>RUNTIME</b> {modelsReady ? 'NAFNet + RT-DETR available' : 'Configure model weights to enable inference'}</span><span>{health ? `${health.total_inferences || 0} persisted inference runs` : 'Health unavailable'}</span></div>
        <Suspense fallback={<LoadingPage />}>
          {active === 'workspace' && <Workspace health={health} onRefresh={refreshHealth} />}
          {active === 'video' && <VideoIntelligence />}
          {active === 'history' && <History />}
          {active === 'benchmarks' && <Benchmarks />}
        </Suspense>
        <footer className="site-footer"><span>VisionShield AI · All-weather visual perception</span><nav aria-label="Footer navigation">{nav.map(([id, label]) => <a href={pages[id].path} key={id} onClick={(event) => navigate(event, id)}>{label}</a>)}</nav></footer>
      </div>
    </main>
  </div>
}
