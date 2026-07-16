# ReMIX companion website

Static companion site for the ReMIX dataset. React + Vite + Tailwind, deployed to
GitHub Pages. Additive to this repo: it reads pipeline outputs but never writes to
them, and nothing here affects the Streamlit apps or the pipeline stages.

```bash
npm install
npm run dev      # http://localhost:5173
npm run build    # -> dist/
```

## Audio: we host none of it

This is a hard constraint, not a preference.

| Catalog | Playback | Why |
| --- | --- | --- |
| MTG-Jamendo | Jamendo CDN | Tracks are Creative Commons. `track_0000214` -> Jamendo track `214`, per the dataset's own `audio_licenses.txt`, which also supplies the title, artist and per-track licence used for attribution. The CDN sends `access-control-allow-origin: *`, so wavesurfer can draw a real waveform. |
| Music4All | Spotify embed | Commercial masters, obtained under a signed agreement whose clause 1 states the database "will **NOT be shared**". Every track carries a `spotify_id`, so Spotify serves the audio under its own licences and we serve none. |

Do not add self-hosted audio for Music4All. Do not strip the Jamendo attribution:
the CC licences require it.

## Data

`public/data/chains.json` is generated, not hand-edited:

```bash
python scripts/export_website_data.py --per-dataset 6
```

The exporter keeps only chains that **both** LLM judges scored >= 4 on at **every**
turn, and pins the judge files to the exact pair the paper reports so the site and
the paper cannot silently diverge.

## Deploying

The workflow (`.github/workflows/deploy-website.yml`) publishes from the `website`
branch, or on demand via the Actions tab. `vite.config.js` sets `base: '/ReMIX/'`,
which assumes the repo is named `ReMIX`; change both together if it is not.

Routing uses `HashRouter` because GitHub Pages has no SPA rewrite — a deep link to
`/explore` would otherwise 404 on refresh.
