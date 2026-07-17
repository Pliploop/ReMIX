import { motion } from 'framer-motion'
import Formula from './Formula.jsx'
import { hexToRgba } from '../theme.js'

/**
 * One small animated diagram per pipeline stage, drawn in the same grammar as the
 * paper's main figure: white ground, tinted cards, thin strokes, stage colour.
 * Purely decorative — every stage's meaning is also stated in prose beside it.
 */
const box = (color) => ({
  fill: hexToRgba(color, 0.1),
  stroke: color,
  strokeWidth: 1.4,
  rx: 4,
})

const draw = {
  hidden: { pathLength: 0, opacity: 0 },
  show: (i = 0) => ({
    pathLength: 1,
    opacity: 1,
    transition: { pathLength: { duration: 0.7, delay: 0.15 * i, ease: 'easeInOut' }, opacity: { duration: 0.2, delay: 0.15 * i } },
  }),
}

const pop = {
  hidden: { scale: 0.8, opacity: 0 },
  show: (i = 0) => ({ scale: 1, opacity: 1, transition: { delay: 0.1 * i, duration: 0.35, ease: 'backOut' } }),
}

const fade = {
  hidden: { opacity: 0 },
  show: (i = 0) => ({ opacity: 1, transition: { delay: 0.08 * i, duration: 0.3 } }),
}

function Frame({ children, viewBox = '0 0 240 150', className = '' }) {
  return (
    <motion.svg
      viewBox={viewBox}
      className={`h-auto w-full max-w-[340px] ${className}`}
      initial="hidden"
      animate="show"
      aria-hidden
    >
      {children}
    </motion.svg>
  )
}

/** Deterministic pseudo-waveform: same shape every render, no useMemo needed. */
const WAVE = Array.from({ length: 22 }, (_, i) => {
  const v = Math.abs(Math.sin(i * 1.7) * 0.6 + Math.sin(i * 0.6) * 0.4)
  return Math.max(0.12, v)
})

function Waveform({ x, y, width, height, color, opacity = 1 }) {
  const bw = width / (WAVE.length * 1.6)
  return (
    <g opacity={opacity}>
      {WAVE.map((v, i) => {
        const h = Math.max(1.2, v * height)
        return (
          <rect
            key={i}
            x={x + i * bw * 1.6}
            y={y + (height - h) / 2}
            width={bw}
            height={h}
            rx={bw / 2}
            fill={color}
          />
        )
      })}
    </g>
  )
}

/**
 * Stage 1. The point is that a clip arrives as audio plus a handful of thin tags,
 * and leaves carrying prose: a caption of how it sounds and a transcript of what
 * is sung. So the diagram shows one track gaining those two text fields, rather
 * than an abstract catalog-to-manifest flow.
 */
function Enrich({ color }) {
  return (
    <Frame viewBox="0 0 240 156">
      {/* Raw clip: audio + the few tags the catalog gave us. */}
      <motion.g variants={pop} custom={0}>
        <rect x="4" y="42" width="56" height="54" {...box(color)} rx="6" />
        <Waveform x={10} y={52} width={44} height={20} color={color} />
        <rect x="10" y="78" width="20" height="5" rx="2.5" fill={hexToRgba(color, 0.35)} />
        <rect x="33" y="78" width="15" height="5" rx="2.5" fill={hexToRgba(color, 0.35)} />
        <text x="32" y="108" textAnchor="middle" fontSize="7.5" fill="currentColor" opacity="0.7">
          30s clip + tags
        </text>
      </motion.g>

      {/* Fan out to the two models... */}
      <motion.path d="M62 62h12a4 4 0 0 1 4 4v-8" fill="none" stroke={color} strokeWidth="1.3" variants={draw} custom={1} />
      <motion.path d="M62 76h12a4 4 0 0 0 4-4v8" fill="none" stroke={color} strokeWidth="1.3" variants={draw} custom={1} />

      <motion.g variants={pop} custom={2}>
        <rect x="78" y="30" width="62" height="30" {...box(color)} rx="5" />
        <text x="109" y="43" textAnchor="middle" fontSize="8.5" fill={color} fontWeight="700">AFNext</text>
        <text x="109" y="53" textAnchor="middle" fontSize="6.5" fill="currentColor" opacity="0.65">captioning</text>
      </motion.g>
      <motion.g variants={pop} custom={3}>
        <rect x="78" y="78" width="62" height="30" {...box(color)} rx="5" />
        <text x="109" y="91" textAnchor="middle" fontSize="8.5" fill={color} fontWeight="700">Whisper</text>
        <text x="109" y="101" textAnchor="middle" fontSize="6.5" fill="currentColor" opacity="0.65">transcription</text>
      </motion.g>

      {/* ...and back into one record. */}
      <motion.path d="M142 45h10a4 4 0 0 1 4 4v14" fill="none" stroke={color} strokeWidth="1.3" variants={draw} custom={4} />
      <motion.path d="M142 93h10a4 4 0 0 0 4-4V75" fill="none" stroke={color} strokeWidth="1.3" variants={draw} custom={4} />
      <motion.path d="M156 69h12" fill="none" stroke={color} strokeWidth="1.3" variants={draw} custom={5} />

      {/* The enriched record: tags, then prose. */}
      <motion.g variants={pop} custom={6}>
        <rect x="168" y="24" width="68" height="90" rx="6" {...box(color)} />
        {[
          { label: 'tags', y: 34, lines: [26, 16] },
          { label: 'caption', y: 58, lines: [54, 48, 38] },
          { label: 'lyrics', y: 88, lines: [44, 30] },
        ].map((row) => (
          <g key={row.label}>
            <text x="176" y={row.y} fontSize="6" fill={color} fontWeight="700" opacity="0.9">
              {row.label}
            </text>
            {row.lines.map((w, i) => (
              <motion.rect
                key={i}
                x="176"
                y={row.y + 4 + i * 6}
                width={w}
                height="3"
                rx="1.5"
                fill={hexToRgba(color, 0.3)}
                variants={fade}
                custom={7 + i}
              />
            ))}
          </g>
        ))}
        <text x="202" y="126" textAnchor="middle" fontSize="7.5" fill="currentColor" opacity="0.7">
          enriched track
        </text>
      </motion.g>
    </Frame>
  )
}

/**
 * Stage 2. The latent grid sits behind because that is literally what happens:
 * the graph is built *in* the embedding space. The formula is the stage's whole
 * claim -- sound and description, weighted equally -- so it is typeset, not drawn.
 */
function Neighbour({ color }) {
  const nodes = [
    [40, 40], [96, 26], [150, 44], [58, 86], [116, 78], [176, 88], [206, 50],
  ]
  const kept = [[0, 1], [1, 2], [0, 3], [3, 4], [4, 5], [2, 6]]
  const pruned = [[1, 4], [4, 6], [3, 1]]

  return (
    <div className="w-full max-w-[340px]">
      <div className="relative">
        <Frame viewBox="0 0 240 120">
          {/* The latent space itself. */}
          <motion.g variants={fade} custom={0}>
            <defs>
              <pattern id="latent-grid" width="12" height="12" patternUnits="userSpaceOnUse">
                <path d="M12 0H0V12" fill="none" stroke={color} strokeOpacity="0.16" strokeWidth="0.5" />
              </pattern>
              <radialGradient id="latent-fade">
                <stop offset="55%" stopColor="white" stopOpacity="1" />
                <stop offset="100%" stopColor="white" stopOpacity="0" />
              </radialGradient>
              <mask id="latent-mask">
                <rect width="240" height="120" fill="url(#latent-fade)" />
              </mask>
            </defs>
            <g mask="url(#latent-mask)">
              <rect width="240" height="120" fill="url(#latent-grid)" />
              {/* Other clips populating the space. */}
              {[[22, 64], [70, 18], [134, 100], [190, 22], [214, 96], [88, 106], [162, 66], [46, 108], [122, 50]].map(
                ([x, y], i) => (
                  <circle key={i} cx={x} cy={y} r="1.6" fill={color} opacity="0.3" />
                ),
              )}
            </g>
          </motion.g>

          {pruned.map(([a, b], i) => (
            <motion.line
              key={`p${i}`}
              x1={nodes[a][0]} y1={nodes[a][1]} x2={nodes[b][0]} y2={nodes[b][1]}
              stroke="currentColor" strokeOpacity="0.22" strokeWidth="1" strokeDasharray="3 3"
              variants={draw} custom={i}
            />
          ))}
          {kept.map(([a, b], i) => (
            <motion.line
              key={`k${i}`}
              x1={nodes[a][0]} y1={nodes[a][1]} x2={nodes[b][0]} y2={nodes[b][1]}
              stroke={color} strokeWidth="1.6"
              variants={draw} custom={i}
            />
          ))}
          {nodes.map(([x, y], i) => (
            <motion.circle
              key={i} cx={x} cy={y} r="7"
              fill={hexToRgba(color, 0.18)} stroke={color} strokeWidth="1.6"
              variants={pop} custom={i}
            />
          ))}
        </Frame>
      </div>

      <motion.div
        initial={{ opacity: 0, y: 6 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.5, duration: 0.35 }}
        className="mt-2 flex justify-center"
      >
        <div
          className="rounded-xl border px-3 py-2 text-center"
          style={{ borderColor: hexToRgba(color, 0.35), backgroundColor: hexToRgba(color, 0.06) }}
        >
          <div className="text-neutral-900 dark:text-neutral-100">
            <Formula
              tex="s(A,B)=\tfrac{1}{2}\bigl[\,s_{\mathrm{audio}}(A,B)+s_{\mathrm{text}}(A,B)\,\bigr]"
              fallback="s(A,B) = ½ [ s_audio(A,B) + s_text(A,B) ]"
            />
          </div>
          <p className="mt-1 text-[10px] text-neutral-500 dark:text-neutral-400">
            MuQ-MuLan · EmbeddingGemma
          </p>
        </div>
      </motion.div>
    </div>
  )
}

function Chain({ color }) {
  const path = [[30, 100], [80, 62], [138, 88], [196, 44]]
  const d = `M${path.map(([x, y]) => `${x} ${y}`).join(' L')}`
  return (
    <Frame>
      {[[70, 26], [160, 118], [110, 34], [46, 52]].map(([x, y], i) => (
        <motion.circle key={i} cx={x} cy={y} r="6" fill="none" stroke="currentColor" strokeOpacity="0.25" strokeWidth="1.3" variants={pop} custom={i} />
      ))}
      <motion.path
        d={d}
        fill="none"
        stroke={color}
        strokeWidth="2.4"
        strokeLinecap="round"
        strokeLinejoin="round"
        variants={draw}
        custom={1}
      />
      {path.map(([x, y], i) => (
        <motion.g key={i} variants={pop} custom={i + 1}>
          <circle cx={x} cy={y} r="9" fill={hexToRgba(color, 0.18)} stroke={color} strokeWidth="1.8" />
          <text x={x} y={y + 3.2} textAnchor="middle" fontSize="8" fontWeight="700" fill={color}>
            {String.fromCharCode(65 + i)}
          </text>
        </motion.g>
      ))}
      <motion.g variants={pop} custom={5}>
        <rect x="76" y="122" width="88" height="20" rx="10" {...box(color)} />
        <text x="120" y="136" textAnchor="middle" fontSize="8.5" fill={color} fontWeight="600">1–6 steps</text>
      </motion.g>
    </Frame>
  )
}

function Instruct({ color }) {
  return (
    <Frame>
      <motion.g variants={pop} custom={0}>
        <rect x="10" y="22" width="92" height="70" rx="5" {...box(color)} />
        {['lost: …', 'new: …', 'preserved: …'].map((t, i) => (
          <text key={t} x="20" y={42 + i * 16} fontSize="7.5" fill={color} fontFamily="ui-monospace, monospace">
            {t}
          </text>
        ))}
        <text x="56" y="106" textAnchor="middle" fontSize="8" fill="currentColor" opacity="0.7">semantic delta</text>
      </motion.g>

      <motion.path d="M104 57h22" stroke={color} strokeWidth="1.4" variants={draw} custom={1} />

      <motion.g variants={pop} custom={2}>
        <rect x="128" y="40" width="46" height="34" rx="5" {...box(color)} />
        <text x="151" y="61" textAnchor="middle" fontSize="8.5" fontWeight="600" fill={color}>LLM</text>
      </motion.g>

      <motion.path d="M176 57h14" stroke={color} strokeWidth="1.4" variants={draw} custom={3} />

      {[0, 1].map((i) => (
        <motion.g key={i} variants={pop} custom={4 + i}>
          <rect x="190" y={36 + i * 24} width="42" height="18" rx="9" fill={hexToRgba(color, 0.16)} stroke={color} strokeWidth="1.2" />
          <text x="211" y={48 + i * 24} textAnchor="middle" fontSize="6.5" fill={color} fontWeight="600">
            {i === 0 ? 'standalone' : 'contextual'}
          </text>
        </motion.g>
      ))}
    </Frame>
  )
}

const STAR = 'M0-4.2 1.2-1.3 4.3-1 2-1.1 2.7 2 0 0.4-2.7 2-2-1.1-4.3-1-1.2-1.3Z'

function Stars({ x, y, n, color, delay }) {
  return (
    <g transform={`translate(${x} ${y})`}>
      {[0, 1, 2, 3, 4].map((i) => (
        <motion.path
          key={i}
          d={STAR}
          transform={`translate(${i * 10} 0) scale(1.05)`}
          fill={i < n ? color : 'none'}
          stroke={i < n ? color : 'currentColor'}
          strokeOpacity={i < n ? 1 : 0.3}
          strokeWidth="0.9"
          initial={{ opacity: 0, scale: 0.4 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: delay + i * 0.07, duration: 0.25, ease: 'backOut' }}
        />
      ))}
    </g>
  )
}

/**
 * Stage 5. Two LLM judges gate the dataset; humans rate a frozen sidecar of the
 * same items against the same rubric, which is what lets us report agreement
 * rather than just assert the judges are right. Both belong in the picture.
 */
function Validate({ color }) {
  const judges = [
    { name: 'Qwen3.6', kind: 'llm', stars: 5 },
    { name: 'Gemma-4', kind: 'llm', stars: 4 },
    { name: 'Human', kind: 'person', stars: 5 },
  ]

  return (
    <Frame viewBox="0 0 240 152">
      {/* The thing being judged. */}
      <motion.g variants={pop} custom={0}>
        <rect x="4" y="52" width="58" height="44" rx="6" {...box(color)} />
        <text x="33" y="70" textAnchor="middle" fontSize="7" fill={color} fontWeight="700">“make it</text>
        <text x="33" y="79" textAnchor="middle" fontSize="7" fill={color} fontWeight="700">punchier”</text>
        <text x="33" y="106" textAnchor="middle" fontSize="7" fill="currentColor" opacity="0.7">variant</text>
      </motion.g>

      {judges.map((j, i) => (
        <motion.path
          key={j.name}
          d={`M64 74C74 74 74 ${28 + i * 44} 84 ${28 + i * 44}`}
          fill="none"
          stroke={color}
          strokeWidth="1.2"
          strokeOpacity={j.kind === 'person' ? 0.9 : 0.55}
          strokeDasharray={j.kind === 'person' ? '3 2' : undefined}
          variants={draw}
          custom={i + 1}
        />
      ))}

      {judges.map((j, i) => {
        const y = 28 + i * 44
        return (
          <motion.g key={j.name} variants={pop} custom={i + 2}>
            <rect x="86" y={y - 16} width="94" height="32" rx="6" {...box(color)} />
            {j.kind === 'llm' ? (
              /* a chip: the machine judges */
              <g stroke={color} strokeWidth="1.1" fill="none">
                <rect x="94" y={y - 8} width="12" height="12" rx="2.5" />
                <path d={`M97 ${y - 11}v3M103 ${y - 11}v3M97 ${y + 4}v3M103 ${y + 4}v3M91 ${y - 5}h3M91 ${y + 1}h3M106 ${y - 5}h3M106 ${y + 1}h3`} />
              </g>
            ) : (
              /* a person: the human sidecar */
              <g stroke={color} strokeWidth="1.2" fill="none">
                <circle cx="100" cy={y - 5} r="3.6" />
                <path d={`M94 ${y + 6}a6 6 0 0 1 12 0`} />
              </g>
            )}
            <text x="114" y={y - 3} fontSize="7.5" fill={color} fontWeight="700">{j.name}</text>
            <Stars x={116} y={y + 8} n={j.stars} color={color} delay={0.3 + i * 0.12} />
          </motion.g>
        )
      })}

      <motion.path d="M182 72h12" stroke={color} strokeWidth="1.3" variants={draw} custom={5} />

      {/* The gate. */}
      <motion.g variants={pop} custom={6}>
        <circle cx="212" cy="56" r="13" fill={hexToRgba('#1FA347', 0.14)} stroke="#1FA347" strokeWidth="1.6" />
        <path d="M206 56l4 4 8-8" fill="none" stroke="#1FA347" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
      </motion.g>
      <motion.g variants={pop} custom={7}>
        <circle cx="212" cy="92" r="13" fill={hexToRgba('#E23B34', 0.1)} stroke="#E23B34" strokeWidth="1.6" strokeDasharray="3 2" />
        <path d="M207 87l10 10M217 87l-10 10" stroke="#E23B34" strokeWidth="1.8" strokeLinecap="round" />
      </motion.g>
      <motion.text x="212" y="120" textAnchor="middle" fontSize="7.5" fill="currentColor" opacity="0.7" variants={pop} custom={8}>
        gate
      </motion.text>
    </Frame>
  )
}

const ART = { enrich: Enrich, neighbour: Neighbour, chain: Chain, instruct: Instruct, validate: Validate }

export default function StageArt({ id, color }) {
  const Art = ART[id]
  return (
    <div className="flex w-full justify-center text-neutral-900 dark:text-neutral-100">
      <Art color={color} />
    </div>
  )
}
