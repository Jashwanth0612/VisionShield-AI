const API_URL = (import.meta.env.VITE_API_URL || 'http://localhost:8000').replace(/\/$/, '')

async function request(path, options = {}) {
  const response = await fetch(`${API_URL}${path}`, options)
  const contentType = response.headers.get('content-type') || ''
  const payload = contentType.includes('application/json') ? await response.json() : await response.text()
  if (!response.ok) {
    const message = typeof payload === 'object' ? payload?.detail?.message || payload?.detail || 'Request failed' : payload
    const error = new Error(typeof message === 'string' ? message : JSON.stringify(message))
    error.status = response.status
    error.payload = payload
    throw error
  }
  return payload
}

function form(fields) {
  const body = new FormData()
  Object.entries(fields).forEach(([key, value]) => {
    if (value !== undefined && value !== null) body.append(key, value)
  })
  return body
}

export const api = {
  baseUrl: API_URL,
  // Backend mounts the pipeline router without a /pipeline prefix.
  health: () => request('/health'),
  processImage: ({ file, enhancement, confidence, weather = 'auto' }) => request(`/pipeline/process?enable_enhancement=${enhancement}&confidence=${confidence}&weather=${encodeURIComponent(weather)}`, {
    method: 'POST',
    body: form({ file }),
  }),
  analyzeVideo: ({ file, enhancement, sampleFps, confidence, weather = 'auto' }) => request(`/video/analyze?enable_enhancement=${enhancement}&sample_fps=${sampleFps}&confidence=${confidence}&weather=${encodeURIComponent(weather)}`, {
    method: 'POST',
    body: form({ file }),
  }),
  history: ({ search = '', mediaType = 'all' } = {}) => request(`/history?search=${encodeURIComponent(search)}&media_type=${mediaType}`),
  historyItem: (runId) => request(`/history/${encodeURIComponent(runId)}`),
  benchmarks: () => request('/benchmarks/summary'),
  runImageBenchmark: ({ file, runs, enhancement, confidence, weather = 'auto' }) => request('/benchmarks/run', {
    method: 'POST',
    body: form({ file, iterations: runs, enable_enhancement: enhancement, confidence, weather }),
  }),
}
