import React, { Suspense, lazy } from 'react'
import ReactDOM from 'react-dom/client'
import { HashRouter, Route, Routes } from 'react-router-dom'
import Home from './pages/Home.jsx'
import Explore from './pages/Explore.jsx'
import ErrorBoundary from './components/ErrorBoundary.jsx'
import './index.css'

// The rating app exists only where it can actually save a rating -- i.e. where a
// backend is configured (the Hugging Face Space sets VITE_API_BASE). On the
// static GitHub Pages build there is nowhere for a rating to go, and a page that
// looks like it works while quietly dropping an hour of someone's annotation is
// worse than no page.
//
// The import is lazy and gated on a compile-time literal (see vite.config.js),
// so the public build drops the rating app's chunk entirely rather than shipping
// unreachable code. A static import, or a gate on
// `Boolean(import.meta.env.VITE_API_BASE)`, both leave the code in the bundle.
const HAS_BACKEND = __HAS_BACKEND__
const Rate = HAS_BACKEND ? lazy(() => import('./pages/Rate.jsx')) : null

// HashRouter, not BrowserRouter: GitHub Pages serves no SPA rewrite, so a deep
// link like /explore would 404 on refresh. The hash keeps routing client-side.
ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <ErrorBoundary label="ReMIX">
      <HashRouter>
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/explore" element={<Explore />} />
          {HAS_BACKEND && Rate && (
            <Route
              path="/rate"
              element={
                <Suspense fallback={null}>
                  <Rate />
                </Suspense>
              }
            />
          )}
        </Routes>
      </HashRouter>
    </ErrorBoundary>
  </React.StrictMode>,
)
