# ReMIX video

Manim animation in the style of the paper's main figure: white ground, the five
stage colours, tinted cards with thin coloured borders, sans type.

Additive to the repo. It reads the exported chain data and writes only into
`video/media/`.

## Environment

`pip install manim` **does not work on this cluster**. `pycairo` ships no binary
wheel, so pip compiles it, and that needs cairo's C headers:

```
../cairo/meson.build:31:12: ERROR: Dependency "cairo" not found (tried pkg-config)
```

Installing those needs root. conda-forge ships them as env-local packages, so:

```bash
conda create -y -p /data/home/acw749/conda-envs/manim -c conda-forge manim
```

Installed and verified: manim 0.20.1, ffmpeg 8.1.2, cairo, pango.

**No LaTeX, by choice.** `Tex`/`MathTex` need a TeX install *and* render in
Computer Modern serif, which fights the sans identity of the paper figure and the
website. Everything uses `Text` (Pango); the one formula sets in Unicode. If you
ever need real LaTeX: `conda install -c conda-forge texlive-core` into that env.

## Rendering

```bash
cd video
PYTHONPATH=$PWD /data/home/acw749/conda-envs/manim/bin/manim -ql scenes/s00_cold_open.py ColdOpen   # draft
PYTHONPATH=$PWD /data/home/acw749/conda-envs/manim/bin/manim -qh scenes/s00_cold_open.py ColdOpen   # 1080p
bash render_all.sh                        # every scene, concatenated
```

Rendering is CPU-bound — **submit it, do not run `-qh` on a login node**:

```bash
sbatch -p compute -c 16 --mem 32G -t 02:00:00 --wrap "cd $PWD && bash render_all.sh -qh"
```

Preview a frame without a video player:

```bash
ffmpeg -y -ss 8.5 -i media/videos/s00_cold_open/480p15/ColdOpen.mp4 -vframes 1 frame.png
```

## Structure

| Path | What |
| --- | --- |
| `remix_video/theme.py` | palette, type scale, card/arrow primitives |
| `remix_video/components.py` | `Waveform`, `TrackCard`, `InstructionBubble`, `StageRail`, `Logo` |
| `remix_video/chain.py` | the one real chain the video follows |
| `scenes/s00_cold_open.py` | the use case, before any pipeline talk |

## The chain

The video follows **one real chain** end to end — `chain_00000496`, MTG-Jamendo,
scored 5.0 by both judges — so nothing on screen is invented:

> Conway Hambone → *swap guitars for industrial percussion and spoken word* →
> The Hate Eighties → **Keep vocals, make them robotic and metal.** →
> After Many Days → *Shout vocals, fast punk, heavy aggressive* →
> Countdown → *Ditch the heavy metal for energetic pop-punk hooks.* → Crazed Outlook

That second turn is the thesis in one line: keep one thing, change another.

It comes from `website/public/data/chains.json`; regenerate with
`python scripts/export_website_data.py`. `chain.py` falls back to a baked-in copy
so a render never dies on data plumbing.

The video is **silent** by design. Captions carry the words, and adding audio
would make the video a derivative of CC BY-NC-SA tracks (share-alike would then
bind the whole video) — and Music4All audio could never be included at all.
