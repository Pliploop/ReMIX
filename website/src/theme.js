// Single source of truth for the pipeline's visual language, matching the
// paper's main figure (paper/figures/Remix Pipeline.png) and the genre plots in
// scripts/paper_data_stats.py. Keep these in sync with that script.

export const STAGES = [
  {
    id: 'enrich',
    n: 1,
    name: 'Dataset Enrichment',
    color: '#E23B34',
    blurb: 'Open catalogs go in. Captions come from AFNext, lyrics and transcripts from Whisper, and everything lands in one structured manifest.',
    detail: 'Music4All and MTG-Jamendo give us audio plus metadata. We caption every clip and transcribe its lyrics, so each track carries a rich text description alongside its tags.',
  },
  {
    id: 'neighbour',
    n: 2,
    name: 'Neighbourhood Building',
    color: '#2E6FD6',
    blurb: 'Every clip is embedded twice — audio with MuQ-MuLan, text with EmbeddingGemma — and the two similarities are averaged.',
    detail: 'A composite similarity s(A,B) blends what a track sounds like with what it is described as. Low-similarity edges are pruned, leaving a directed graph of plausible transitions.',
  },
  {
    id: 'chain',
    n: 3,
    name: 'Chain Sampling',
    color: '#1FA347',
    blurb: 'A stochastic weighted walk over that graph draws multi-turn chains of 1–6 steps.',
    detail: 'Because transitions are sampled in proportion to similarity, chains stay musically plausible while still surprising — each hop is a change someone could actually ask for.',
  },
  {
    id: 'instruct',
    n: 4,
    name: 'Instruction Generation',
    color: '#FB8B24',
    blurb: 'We diff each pair into a semantic delta, then ask an LLM to write the instruction that turns one into the other.',
    detail: 'Every step gets both a standalone instruction and a contextual one that can refer back to earlier turns ("bring back the piano from before"). Five variants are drafted per step.',
  },
  {
    id: 'validate',
    n: 5,
    name: 'Validation & Benchmark',
    color: '#7B3FF2',
    blurb: 'Two LLM judges score every variant against a rubric, and a gate keeps only what passes.',
    detail: 'Qwen3.6-27B and Gemma-4-31B independently rate each instruction. The surviving variants form ReMIX; a graded relevance pool over held-out chains forms the ReMIX-B benchmark.',
  },
]

export const STAGE_BY_ID = Object.fromEntries(STAGES.map((s) => [s.id, s]))

// Fixed genre -> colour, shared with the paper's donut charts so the site and
// the figures never disagree about what "Rock" looks like.
export const GENRE_COLORS = {
  Rock: '#2E6FD6',
  Pop: '#FB8B24',
  Metal: '#E23B34',
  Electronic: '#7B3FF2',
  Folk: '#1FA347',
  Punk: '#EC4F9E',
  'Hip-hop': '#F5B301',
  'Soul/R&B': '#17B2C3',
  Classical: '#C81D6B',
  Ambient: '#00B4D8',
  Jazz: '#8A5A2B',
  Reggae: '#86C232',
  Blues: '#3A6EA5',
  Country: '#CC7A00',
  Latin: '#F72585',
  Funk: '#9B5DE5',
  World: '#00BFA5',
  Experimental: '#6C757D',
  Other: '#C3C7CD',
}

// Placeholders — fill in once the paper and dataset are public.
export const LINKS = {
  paper: '#',
  huggingface: '#',
  github: 'https://github.com/Pliploop/ReMIX',
  video: '#',
}

export const hexToRgba = (hex, a) => {
  const h = hex.replace('#', '')
  const r = parseInt(h.slice(0, 2), 16)
  const g = parseInt(h.slice(2, 4), 16)
  const b = parseInt(h.slice(4, 6), 16)
  return `rgba(${r}, ${g}, ${b}, ${a})`
}
