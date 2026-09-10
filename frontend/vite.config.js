import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'
import { writeFileSync } from 'node:fs'
import { resolve } from 'node:path'

const routes = ['/', '/video', '/history', '/benchmarks']

function seoFiles(siteUrl, mode) {
  return {
    name: 'visionshield-seo-files',
    apply: 'build',
    closeBundle() {
      if (!siteUrl) {
        if (mode === 'production') throw new Error('VITE_SITE_URL is required for a production SEO build. Set it to the real deployed site origin.')
        return
      }
      const normalized = siteUrl.replace(/\/$/, '')
      const sitemap = `<?xml version="1.0" encoding="UTF-8"?>\n<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n${routes.map((route) => `  <url><loc>${normalized}${route === '/' ? '/' : route}</loc></url>`).join('\n')}\n</urlset>\n`
      const robots = `User-agent: *\nAllow: /\nSitemap: ${normalized}/sitemap.xml\n`
      const outDir = resolve(process.cwd(), 'dist')
      writeFileSync(resolve(outDir, 'sitemap.xml'), sitemap)
      writeFileSync(resolve(outDir, 'robots.txt'), robots)
    },
  }
}

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '')
  const siteUrl = env.VITE_SITE_URL || (mode === 'development' ? 'http://localhost:5173' : '')

  return {
    plugins: [react(), seoFiles(siteUrl, mode)],
    server: { port: 5173, host: '0.0.0.0' },
    build: {
      sourcemap: false,
      cssCodeSplit: true,
      rollupOptions: {
        output: {
          manualChunks: { react: ['react', 'react-dom'] },
        },
      },
    },
  }
})
