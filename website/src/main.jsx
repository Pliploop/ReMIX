import React from 'react'
import ReactDOM from 'react-dom/client'
import { HashRouter, Route, Routes } from 'react-router-dom'
import Home from './pages/Home.jsx'
import Explore from './pages/Explore.jsx'
import ErrorBoundary from './components/ErrorBoundary.jsx'
import './index.css'

// HashRouter, not BrowserRouter: GitHub Pages serves no SPA rewrite, so a deep
// link like /explore would 404 on refresh. The hash keeps routing client-side.
ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <ErrorBoundary label="ReMIX">
      <HashRouter>
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/explore" element={<Explore />} />
        </Routes>
      </HashRouter>
    </ErrorBoundary>
  </React.StrictMode>,
)
