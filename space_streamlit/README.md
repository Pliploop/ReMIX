---
title: ReMIX Ratings
emoji: 🎧
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# ReMIX human validation (Streamlit)

The same Streamlit app the team runs on the cluster behind cloudflared
(`src/jamendo_instruct/demo/human_validation_app.py`), packaged for external raters.

This Space repo holds **code only**. The audio clips, the frozen validation sidecars and the
ratings live in the private dataset repo `DATASET_REPO`:

```
audio/<catalogue>/<clip>.mp3        30-second clips (Music4All is not redistributable: keep the repo private)
sidecars/<catalogue>.sidecar.json   the frozen rating slices
ratings/<catalogue>/*.jsonl         written by this Space (CommitScheduler, every EVERY_MINUTES)
```

On start the app downloads `audio/`, `sidecars/` and `ratings/`, restores the earlier ratings,
and only then starts committing, so a restart never overwrites saved ratings. If the download
fails, the app shows an error and takes no ratings.

## Settings → Variables and secrets

| Name | Kind | Value |
| --- | --- | --- |
| `DATASET_REPO` | variable | `Pliploop/Remix-Human-Ratings` |
| `HF_TOKEN` | secret | fine-grained token with write access to that dataset only |
| `RATER_CODE` | secret | access code raters type in (or pass as `?code=` in their link) |
| `ADMIN_PASSWORD` | secret | password for the Admin / LLM Ratings pages |
| `EVERY_MINUTES` | variable (optional) | commit interval, default `5` |

## Rater links

Send `https://pliploop-remix-ratings.hf.space/?code=<RATER_CODE>` (or the bare URL; the app then
asks for the code). Each session drafts a random rater id (`rater_<hash>`) and writes it into the
URL as `?annotator=...`; raters who bookmark that URL keep their id and progress. No names are used.

Rebuild the upload folders with `scripts/prepare_rating_space.py`; pull ratings back with
`scripts/sync_human_ratings.py --repo Pliploop/Remix-Human-Ratings`.
