import { motion } from 'framer-motion'
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

function Frame({ children }) {
  return (
    <motion.svg
      viewBox="0 0 240 150"
      className="h-auto w-full max-w-[320px]"
      initial="hidden"
      animate="show"
      aria-hidden
    >
      {children}
    </motion.svg>
  )
}

function Enrich({ color }) {
  return (
    <Frame>
      <motion.g variants={pop} custom={0}>
        <ellipse cx="38" cy="38" rx="22" ry="7" {...box(color)} />
        <path d="M16 38v34c0 3.9 9.8 7 22 7s22-3.1 22-7V38" {...box(color)} fill={hexToRgba(color, 0.1)} />
        <ellipse cx="38" cy="72" rx="22" ry="7" fill="none" stroke={color} strokeWidth="1.4" />
        <text x="38" y="96" textAnchor="middle" fontSize="8" fill="currentColor" opacity="0.7">catalogs</text>
      </motion.g>

      {[0, 1].map((i) => (
        <motion.path
          key={i}
          d={`M62 ${48 + i * 20}h28`}
          stroke={color}
          strokeWidth="1.4"
          fill="none"
          markerEnd=""
          variants={draw}
          custom={i + 1}
        />
      ))}

      <motion.g variants={pop} custom={2}>
        <rect x="92" y="34" width="58" height="22" {...box(color)} />
        <text x="121" y="48" textAnchor="middle" fontSize="8.5" fill={color} fontWeight="600">AFNext</text>
      </motion.g>
      <motion.g variants={pop} custom={3}>
        <rect x="92" y="66" width="58" height="22" {...box(color)} />
        <text x="121" y="80" textAnchor="middle" fontSize="8.5" fill={color} fontWeight="600">Whisper</text>
      </motion.g>

      <motion.path d="M152 45h20v16h-20" stroke={color} strokeWidth="1.4" fill="none" variants={draw} custom={4} />
      <motion.path d="M152 77h20V61" stroke={color} strokeWidth="1.4" fill="none" variants={draw} custom={4} />

      <motion.g variants={pop} custom={5}>
        <rect x="176" y="30" width="52" height="62" rx="5" fill="none" stroke={color} strokeWidth="1.4" strokeDasharray="3 3" />
        {[0, 1, 2, 3].map((i) => (
          <rect key={i} x="183" y={39 + i * 14} width="38" height="8" rx="2" fill={hexToRgba(color, 0.28)} />
        ))}
        <text x="202" y="106" textAnchor="middle" fontSize="8" fill="currentColor" opacity="0.7">manifest</text>
      </motion.g>
    </Frame>
  )
}

function Neighbour({ color }) {
  const nodes = [
    [40, 40], [96, 26], [150, 44], [58, 86], [116, 78], [176, 88], [206, 50],
  ]
  const kept = [[0, 1], [1, 2], [0, 3], [3, 4], [4, 5], [2, 6]]
  const pruned = [[1, 4], [4, 6], [3, 1]]

  return (
    <Frame>
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
      <motion.g variants={pop} custom={7}>
        <rect x="52" y="112" width="136" height="26" rx="6" {...box(color)} />
        <text x="120" y="129" textAnchor="middle" fontSize="10" fill={color} fontFamily="ui-monospace, monospace">
          s(A,B) = ½(s_A + s_T)
        </text>
      </motion.g>
    </Frame>
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

function Validate({ color }) {
  const rows = [
    ['meaningful', 0.92],
    ['grounded', 0.84],
    ['specific', 0.76],
    ['clear', 0.88],
  ]
  return (
    <Frame>
      <motion.g variants={pop} custom={0}>
        <rect x="12" y="18" width="130" height="84" rx="6" {...box(color)} />
      </motion.g>
      {rows.map(([label, v], i) => (
        <g key={label}>
          <motion.text x="22" y={38 + i * 19} fontSize="7.5" fill="currentColor" opacity="0.75" variants={pop} custom={i}>
            {label}
          </motion.text>
          <rect x="72" y={31 + i * 19} width="58" height="6" rx="3" fill={hexToRgba(color, 0.14)} />
          <motion.rect
            x="72" y={31 + i * 19} height="6" rx="3" fill={color}
            initial={{ width: 0 }}
            animate={{ width: 58 * v }}
            transition={{ delay: 0.25 + i * 0.1, duration: 0.5, ease: 'easeOut' }}
          />
        </g>
      ))}

      <motion.path d="M144 60h18" stroke={color} strokeWidth="1.4" variants={draw} custom={4} />

      <motion.g variants={pop} custom={5}>
        <circle cx="182" cy="44" r="13" fill={hexToRgba('#1FA347', 0.14)} stroke="#1FA347" strokeWidth="1.6" />
        <path d="M176 44l4 4 8-8" fill="none" stroke="#1FA347" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
      </motion.g>
      <motion.g variants={pop} custom={6}>
        <circle cx="182" cy="80" r="13" fill={hexToRgba('#E23B34', 0.1)} stroke="#E23B34" strokeWidth="1.6" strokeDasharray="3 2" />
        <path d="M177 75l10 10M187 75l-10 10" stroke="#E23B34" strokeWidth="1.8" strokeLinecap="round" />
      </motion.g>
      <motion.text x="182" y="112" textAnchor="middle" fontSize="8" fill="currentColor" opacity="0.7" variants={pop} custom={7}>
        gate
      </motion.text>
    </Frame>
  )
}

const ART = { enrich: Enrich, neighbour: Neighbour, chain: Chain, instruct: Instruct, validate: Validate }

export default function StageArt({ id, color }) {
  const Art = ART[id]
  return (
    <div className="w-full text-neutral-900 dark:text-neutral-100">
      <Art color={color} />
    </div>
  )
}
