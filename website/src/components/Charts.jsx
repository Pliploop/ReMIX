import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { GENRE_COLORS, hexToRgba } from '../theme.js'

// Recharts needs concrete colours, so the two themes are resolved here rather
// than through CSS. `dark` is threaded down from the page's theme toggle.
export const axisTheme = (dark) => ({
  tick: { fill: dark ? '#a1a1aa' : '#8E8E96', fontSize: 11 },
  line: dark ? '#3f3f46' : '#E7E5E4',
  grid: dark ? '#27272a' : '#f4f4f5',
})

// Compact y-axis ticks (26000 -> 26k) so they fit their gutter instead of being clipped.
const kFormat = (v) => (Math.abs(v) >= 1000 ? `${Math.round(v / 1000)}k` : `${v}`)

function ChartTip({ active, payload, label, unit = '', dark }) {
  if (!active || !payload?.length) return null
  return (
    <div
      className="rounded-lg border px-3 py-2 text-xs shadow-lg"
      style={{
        backgroundColor: dark ? '#18181b' : '#ffffff',
        borderColor: dark ? '#3f3f46' : '#e4e4e7',
        color: dark ? '#e4e4e7' : '#27272a',
      }}
    >
      {label !== undefined && <p className="mb-1 font-semibold">{label}</p>}
      {payload.map((p) => (
        <p key={p.dataKey ?? p.name} className="flex items-center gap-2">
          <span className="inline-block h-2 w-2 rounded-full" style={{ backgroundColor: p.color ?? p.payload?.fill }} />
          <span className="text-neutral-500 dark:text-neutral-400">{p.name}</span>
          <span className="font-semibold">
            {typeof p.value === 'number' ? p.value.toLocaleString() : p.value}
            {unit}
          </span>
        </p>
      ))}
    </div>
  )
}

function Panel({ title, note, children, height = 260 }) {
  return (
    <figure className="surface p-6">
      <figcaption className="mb-5">
        <h4 className="text-[15px] font-medium text-ink dark:text-neutral-100">{title}</h4>
        {note && <p className="mt-1 text-xs text-ink-2 dark:text-neutral-400">{note}</p>}
      </figcaption>
      <div style={{ height }}>
        <ResponsiveContainer width="100%" height="100%">
          {children}
        </ResponsiveContainer>
      </div>
    </figure>
  )
}

export function GenreDonut({ data, dark, total }) {
  return (
    <Panel title="What music is in here?" note="Clips by genre, from the catalog tags.">
      <PieChart>
        <Pie
          data={data}
          dataKey="value"
          nameKey="name"
          innerRadius="58%"
          outerRadius="82%"
          paddingAngle={1.5}
          stroke={dark ? '#0a0a0a' : '#ffffff'}
          strokeWidth={2}
        >
          {data.map((d) => (
            <Cell key={d.name} fill={GENRE_COLORS[d.name] ?? GENRE_COLORS.Other} />
          ))}
        </Pie>
        <Tooltip content={<ChartTip dark={dark} />} />
        <Legend
          verticalAlign="bottom"
          height={44}
          iconType="circle"
          iconSize={7}
          formatter={(v) => <span className="text-xs text-neutral-600 dark:text-neutral-400">{v}</span>}
        />
      </PieChart>
    </Panel>
  )
}

export function ChainLengthBar({ data, dark, color }) {
  const t = axisTheme(dark)
  return (
    <Panel title="How long are the conversations?" note="Chains by number of turns.">
      <BarChart data={data} margin={{ top: 6, right: 8, left: 0, bottom: 0 }}>
        <CartesianGrid stroke={t.grid} vertical={false} />
        <XAxis dataKey="steps" tick={t.tick} axisLine={{ stroke: t.line }} tickLine={false} />
        <YAxis tick={t.tick} axisLine={false} tickLine={false} width={40} tickFormatter={kFormat} />
        <Tooltip cursor={{ fill: hexToRgba(color, 0.06) }} content={<ChartTip dark={dark} />} />
        <Bar dataKey="chains" name="Chains" fill={color} radius={[4, 4, 0, 0]} />
      </BarChart>
    </Panel>
  )
}

export function AxesBar({ data, dark }) {
  const t = axisTheme(dark)
  return (
    <Panel
      title="What do instructions ask to change?"
      note="Steps per musical axis — changed versus explicitly preserved."
      height={300}
    >
      <BarChart data={data} layout="vertical" margin={{ top: 4, right: 12, left: 40, bottom: 0 }}>
        <CartesianGrid stroke={t.grid} horizontal={false} />
        <XAxis type="number" tick={t.tick} axisLine={{ stroke: t.line }} tickLine={false} />
        <YAxis type="category" dataKey="axis" tick={t.tick} axisLine={false} tickLine={false} width={132} />
        <Tooltip cursor={{ fill: hexToRgba('#F76B15', 0.06) }} content={<ChartTip dark={dark} />} />
        <Legend
          iconType="circle"
          iconSize={7}
          formatter={(v) => <span className="text-xs text-neutral-600 dark:text-neutral-400">{v}</span>}
        />
        <Bar dataKey="changed" name="Changed" fill="#F76B15" radius={[0, 3, 3, 0]} />
        <Bar dataKey="preserved" name="Preserved" fill="#3E63DD" radius={[0, 3, 3, 0]} />
      </BarChart>
    </Panel>
  )
}

export function TransitionArea({ data, dark, color }) {
  const t = axisTheme(dark)
  return (
    <Panel title="How big is each jump?" note="Similarity between the two tracks of a step.">
      <AreaChart data={data} margin={{ top: 6, right: 8, left: 0, bottom: 0 }}>
        <defs>
          <linearGradient id="transGrad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor={color} stopOpacity={0.45} />
            <stop offset="100%" stopColor={color} stopOpacity={0.02} />
          </linearGradient>
        </defs>
        <CartesianGrid stroke={t.grid} vertical={false} />
        <XAxis
          dataKey="x"
          tick={t.tick}
          axisLine={{ stroke: t.line }}
          tickLine={false}
          tickFormatter={(v) => v.toFixed(1)}
        />
        <YAxis tick={t.tick} axisLine={false} tickLine={false} width={40} tickFormatter={kFormat} />
        <Tooltip
          content={<ChartTip dark={dark} />}
          labelFormatter={(v) => `similarity ≈ ${Number(v).toFixed(2)}`}
        />
        <Area
          type="monotone"
          dataKey="count"
          name="Steps"
          stroke={color}
          strokeWidth={2}
          fill="url(#transGrad)"
        />
      </AreaChart>
    </Panel>
  )
}

const JUDGE_COLORS = ['#8E4EC6', '#30A46C']

export function AcceptByQuestion({ data, judges, dark }) {
  const t = axisTheme(dark)
  return (
    <Panel
      title="How often does each judge accept?"
      note="Share of instructions scoring 4 or 5, per rubric question."
      height={320}
    >
      <BarChart data={data} layout="vertical" margin={{ top: 4, right: 16, left: 50, bottom: 0 }}>
        <CartesianGrid stroke={t.grid} horizontal={false} />
        <XAxis type="number" domain={[0, 100]} unit="%" tick={t.tick} axisLine={{ stroke: t.line }} tickLine={false} />
        <YAxis type="category" dataKey="question" tick={t.tick} axisLine={false} tickLine={false} width={124} />
        <Tooltip cursor={{ fill: hexToRgba('#8E4EC6', 0.06) }} content={<ChartTip dark={dark} unit="%" />} />
        <Legend
          iconType="circle"
          iconSize={7}
          formatter={(v) => <span className="text-xs text-neutral-600 dark:text-neutral-400">{v}</span>}
        />
        {judges.map((j, i) => (
          <Bar key={j} dataKey={j} name={j} fill={JUDGE_COLORS[i % 2]} radius={[0, 3, 3, 0]} />
        ))}
      </BarChart>
    </Panel>
  )
}

export function AgreementBar({ data, dark }) {
  const t = axisTheme(dark)
  return (
    <Panel
      title="Do the two judges agree?"
      note="Gwet's AC1 on the accept decision — chance-corrected, and unlike κ it does not collapse when judges mostly agree."
      height={320}
    >
      <BarChart data={data} layout="vertical" margin={{ top: 4, right: 16, left: 50, bottom: 0 }}>
        <CartesianGrid stroke={t.grid} horizontal={false} />
        <XAxis type="number" domain={[0, 1]} tick={t.tick} axisLine={{ stroke: t.line }} tickLine={false} />
        <YAxis type="category" dataKey="question" tick={t.tick} axisLine={false} tickLine={false} width={124} />
        <Tooltip cursor={{ fill: hexToRgba('#3E63DD', 0.06) }} content={<ChartTip dark={dark} />} />
        <Bar dataKey="ac1" name="AC1" radius={[0, 3, 3, 0]}>
          {data.map((d) => (
            <Cell key={d.question} fill={d.ac1 >= 0.9 ? '#30A46C' : d.ac1 >= 0.8 ? '#3E63DD' : '#F76B15'} />
          ))}
        </Bar>
      </BarChart>
    </Panel>
  )
}
