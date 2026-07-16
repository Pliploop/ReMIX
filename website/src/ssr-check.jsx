// Dev-only smoke test: render each page to a string in Node. Effects do not run
// under SSR, so this cannot catch browser-only failures, but it does execute
// every component's render path -- which is where a blank page comes from.
import React from 'react'
import { renderToString } from 'react-dom/server'
import { MemoryRouter } from 'react-router-dom'
import Home from './pages/Home.jsx'
import Explore from './pages/Explore.jsx'
import Rate from './pages/Rate.jsx'

const pages = { Home, Explore, Rate }
let failed = false
for (const [name, Page] of Object.entries(pages)) {
  try {
    const html = renderToString(
      React.createElement(MemoryRouter, null, React.createElement(Page)),
    )
    console.log(`${name.padEnd(8)} OK   ${html.length} chars`)
  } catch (e) {
    failed = true
    console.log(`${name.padEnd(8)} THREW: ${e.message}`)
    console.log(String(e.stack).split('\n').slice(1, 4).join('\n'))
  }
}
process.exit(failed ? 1 : 0)
