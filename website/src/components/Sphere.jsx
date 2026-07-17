import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Canvas, useFrame, useThree } from '@react-three/fiber'
import { Html, OrbitControls } from '@react-three/drei'
import * as THREE from 'three'
import { SPHERE_RADIUS as RADIUS } from '../theme.js'

// How far a dense neighbourhood lifts off the sphere, as a fraction of RADIUS.
const RELIEF = 0.14
// The chain path floats above the tallest relief so it never dips into terrain.
const PATH_LIFT = 1.0 + RELIEF + 0.04

/** Great-circle (slerp) arc between two points, lifted clear of the relief. */
export function arcPoints(a, b, segments = 64, lift = PATH_LIFT) {
  const va = new THREE.Vector3(...a).normalize()
  const vb = new THREE.Vector3(...b).normalize()
  const out = []
  const dot = THREE.MathUtils.clamp(va.dot(vb), -1, 1)
  const omega = Math.acos(dot)
  for (let i = 0; i <= segments; i++) {
    const t = i / segments
    let v
    if (omega < 1e-6) {
      v = va.clone()
    } else {
      const s0 = Math.sin((1 - t) * omega) / Math.sin(omega)
      const s1 = Math.sin(t * omega) / Math.sin(omega)
      v = va.clone().multiplyScalar(s0).add(vb.clone().multiplyScalar(s1))
    }
    const bow = lift + 0.08 * Math.sin(Math.PI * t)
    out.push(v.normalize().multiplyScalar(RADIUS * bow))
  }
  return out
}

const lift = (p, k) => new THREE.Vector3(...p).normalize().multiplyScalar(RADIUS * k)

/**
 * Turn the raw cloud into a relief map: bin points onto a lat/long grid, and
 * push each point outward in proportion to how crowded its cell is. Dense
 * regions of the embedding rise into ridges; sparse space stays near the
 * surface. Returns the displaced positions, a matching density in [0,1], and a
 * per-point colour warming with density — so the landscape reads even before a
 * chain is drawn on it.
 */
function useRelief(positions, dark) {
  return useMemo(() => {
    const n = positions.length / 3
    const NLAT = 40
    const NLON = 80
    const bins = new Float32Array(NLAT * NLON)
    const binOf = new Int32Array(n)

    for (let i = 0; i < n; i++) {
      const x = positions[i * 3]
      const y = positions[i * 3 + 1]
      const z = positions[i * 3 + 2]
      const r = Math.hypot(x, y, z) || 1
      const lat = Math.acos(THREE.MathUtils.clamp(y / r, -1, 1)) / Math.PI
      const lon = (Math.atan2(z, x) + Math.PI) / (2 * Math.PI)
      const bi =
        Math.min(NLAT - 1, Math.floor(lat * NLAT)) * NLON + Math.min(NLON - 1, Math.floor(lon * NLON))
      bins[bi] += 1
      binOf[i] = bi
    }

    let max = 0
    for (const b of bins) if (b > max) max = b

    const displaced = new Float32Array(n * 3)
    const density = new Float32Array(n)
    const colors = new Float32Array(n * 3)
    const lo = new THREE.Color(dark ? '#3f3f46' : '#b6bcc6')
    const hi = new THREE.Color('#FB8B24')
    const c = new THREE.Color()

    for (let i = 0; i < n; i++) {
      const d = max > 0 ? bins[binOf[i]] / max : 0
      const eased = Math.pow(d, 0.6) // lift low-density areas a little off the floor
      density[i] = d
      const x = positions[i * 3]
      const y = positions[i * 3 + 1]
      const z = positions[i * 3 + 2]
      const r = Math.hypot(x, y, z) || 1
      const s = (RADIUS * (1 + RELIEF * eased)) / r
      displaced[i * 3] = x * s
      displaced[i * 3 + 1] = y * s
      displaced[i * 3 + 2] = z * s
      c.copy(lo).lerp(hi, d)
      colors[i * 3] = c.r
      colors[i * 3 + 1] = c.g
      colors[i * 3 + 2] = c.b
    }
    return { displaced, density, colors }
  }, [positions, dark])
}

/** All tracks as one instanced point cloud, coloured and lifted by local density. */
function Cloud({ displaced, colors, onHover, onPick }) {
  const dot = useDotTexture()
  const geom = useMemo(() => {
    const g = new THREE.BufferGeometry()
    g.setAttribute('position', new THREE.BufferAttribute(displaced, 3))
    g.setAttribute('color', new THREE.BufferAttribute(colors, 3))
    return g
  }, [displaced, colors])

  useEffect(() => () => geom.dispose(), [geom])

  const last = useRef(-1)

  return (
    <points
      geometry={geom}
      onPointerMove={(e) => {
        e.stopPropagation()
        const i = e.index ?? -1
        if (i !== last.current) {
          last.current = i
          onHover(i)
        }
      }}
      onPointerOut={() => {
        last.current = -1
        onHover(-1)
      }}
      onClick={(e) => {
        e.stopPropagation()
        if (e.index != null) onPick(e.index)
      }}
    >
      {/* Round sprites, not squares: a soft radial-alpha texture makes each point a dot. */}
      <pointsMaterial size={0.075} sizeAttenuation vertexColors map={dot} alphaTest={0.5} transparent depthWrite />
    </points>
  )
}

/** A small round alpha sprite so GL points render as dots rather than squares. */
let _dotTex = null
function useDotTexture() {
  return useMemo(() => {
    if (_dotTex) return _dotTex
    const s = 64
    const cv = document.createElement('canvas')
    cv.width = cv.height = s
    const ctx = cv.getContext('2d')
    const g = ctx.createRadialGradient(s / 2, s / 2, 0, s / 2, s / 2, s / 2)
    g.addColorStop(0, 'rgba(255,255,255,1)')
    g.addColorStop(0.7, 'rgba(255,255,255,1)')
    g.addColorStop(1, 'rgba(255,255,255,0)')
    ctx.fillStyle = g
    ctx.fillRect(0, 0, s, s)
    _dotTex = new THREE.CanvasTexture(cv)
    return _dotTex
  }, [])
}

/**
 * A faint solid shell plus a grey wireframe cage.
 *
 * The solid shell is nearly clear (opacity 0.1) but it still intercepts
 * raycasts, so you cannot select a point on the far side straight through the
 * globe — clicks land on what you can actually see. The wireframe gives the
 * sphere its form without hiding the cloud.
 */
function Globe({ dark }) {
  return (
    <>
      <mesh>
        <sphereGeometry args={[RADIUS * 0.992, 48, 32]} />
        <meshBasicMaterial
          color={dark ? '#0a0a0a' : '#ffffff'}
          transparent
          opacity={0.1}
          side={THREE.FrontSide}
          depthWrite={false}
        />
      </mesh>
      <mesh raycast={() => null}>
        <sphereGeometry args={[RADIUS * 0.994, 32, 20]} />
        <meshBasicMaterial
          color={dark ? '#a1a1aa' : '#9aa0aa'}
          wireframe
          transparent
          opacity={0.5}
          depthWrite={false}
        />
      </mesh>
    </>
  )
}

/** A bright dot that flows source→target along the active arc, showing direction. */
function FlowComet({ curve, color }) {
  const ref = useRef()
  useFrame(({ clock }) => {
    if (!ref.current || !curve) return
    const t = (clock.getElapsedTime() * 0.35) % 1
    curve.getPointAt(t, ref.current.position)
  })
  return (
    <mesh ref={ref} raycast={() => null}>
      <sphereGeometry args={[0.045, 16, 16]} />
      <meshBasicMaterial color={color} />
    </mesh>
  )
}

function ChainPath({ nodes, activeStep, dark }) {
  const arcs = useMemo(() => {
    const out = []
    for (let i = 0; i < nodes.length - 1; i++) out.push(arcPoints(nodes[i], nodes[i + 1]))
    return out
  }, [nodes])

  const curves = useMemo(() => arcs.map((pts) => new THREE.CatmullRomCurve3(pts)), [arcs])

  const geoms = useMemo(() => arcs.map((pts) => new THREE.BufferGeometry().setFromPoints(pts)), [arcs])
  const tubes = useMemo(
    () => curves.map((cv) => new THREE.TubeGeometry(cv, 48, 0.014, 8, false)),
    [curves],
  )
  useEffect(() => () => {
    geoms.forEach((g) => g.dispose())
    tubes.forEach((t) => t.dispose())
  }, [geoms, tubes])

  return (
    <>
      {arcs.map((_, i) => {
        const on = i === activeStep
        if (on) {
          // The live edge is a real tube so it reads with weight, plus a comet.
          return (
            <group key={i}>
              <mesh geometry={tubes[i]} raycast={() => null}>
                <meshBasicMaterial color="#FB8B24" />
              </mesh>
              <FlowComet curve={curves[i]} color="#FB8B24" />
            </group>
          )
        }
        return (
          <line key={i} geometry={geoms[i]}>
            <lineBasicMaterial color={dark ? '#6b6b74' : '#a2a8b2'} transparent opacity={0.65} />
          </line>
        )
      })}
      {nodes.map((p, i) => {
        const isEnd = i === nodes.length - 1
        const on = i === activeStep || i === activeStep + 1
        const color = i === 0 ? '#1FA347' : isEnd ? '#7B3FF2' : '#2E6FD6'
        return (
          <mesh key={i} position={lift(p, PATH_LIFT)}>
            <sphereGeometry args={[on ? 0.12 : 0.08, 24, 24]} />
            <meshBasicMaterial color={color} transparent opacity={on ? 1 : 0.75} />
          </mesh>
        )
      })}
    </>
  )
}

/** Grows the point under the cursor. */
function HoverMarker({ position }) {
  if (!position) return null
  return (
    <mesh position={position} raycast={() => null}>
      <sphereGeometry args={[0.06, 16, 16]} />
      <meshBasicMaterial color="#FB8B24" />
    </mesh>
  )
}

/**
 * Flies the camera to a target once, then hands control back — never per frame.
 * A fly is armed only when flyKey changes (new chain / Home) and any interaction
 * cancels it. The target is read through a ref so a freshly-allocated focus array
 * cannot re-arm the fly on every render. Distance comes from wherever the camera
 * already is, so a fly preserves the viewer's zoom.
 */
function CameraRig({ target, flyKey, controls }) {
  const desired = useRef(null)
  const flying = useRef(false)
  const targetRef = useRef(target)
  targetRef.current = target

  useEffect(() => {
    const t = targetRef.current
    if (!t) return
    const c = controls.current?.object
    const radius = c ? THREE.MathUtils.clamp(c.position.length(), 3.2, 9) : 6.2
    desired.current = new THREE.Vector3(...t).normalize().multiplyScalar(radius)
    flying.current = true
  }, [flyKey, controls])

  useEffect(() => {
    const c = controls.current
    if (!c) return undefined
    const cancel = () => {
      flying.current = false
    }
    c.addEventListener('start', cancel)
    return () => c.removeEventListener('start', cancel)
  }, [controls])

  useFrame(({ camera }) => {
    if (!flying.current || !desired.current) return
    camera.position.lerp(desired.current, 0.07)
    controls.current?.update?.()
    if (camera.position.distanceTo(desired.current) < 0.02) flying.current = false
  })

  return null
}

function Scene({ positions, pointIds, chainNodes, activeStep, focus, flyKey, dark, onPick, tooltipFor, cards }) {
  const controls = useRef()
  const { gl, raycaster } = useThree()
  const [hover, setHover] = useState(-1)

  const { displaced, colors } = useRelief(positions, dark)

  useEffect(() => {
    gl.setClearAlpha(0)
  }, [gl])

  // Default Points threshold is 1 (half the sphere radius) so every ray hits.
  // Scale it to roughly a point's on-screen size.
  useEffect(() => {
    raycaster.params.Points.threshold = 0.05
  }, [raycaster])

  // `hover` indexes the drawn cloud (a subset); pointIds maps it back to the
  // global track index the tooltip and pick callbacks speak in.
  const toGlobal = (i) => (pointIds ? pointIds[i] : i)
  const hoverPos = hover >= 0 ? [displaced[hover * 3], displaced[hover * 3 + 1], displaced[hover * 3 + 2]] : null
  const tip = hover >= 0 ? tooltipFor(toGlobal(hover)) : null

  return (
    <>
      <Globe dark={dark} />
      <Cloud displaced={displaced} colors={colors} onHover={setHover} onPick={(i) => onPick(toGlobal(i))} />
      {chainNodes?.length > 1 && <ChainPath nodes={chainNodes} activeStep={activeStep} dark={dark} />}
      <HoverMarker position={hoverPos} />

      {hoverPos && tip && (
        <Html position={hoverPos} center style={{ pointerEvents: 'none' }} zIndexRange={[40, 0]}>
          <div className="w-56 -translate-y-14 rounded-xl border border-white/60 bg-white/80 p-2.5 shadow-2xl backdrop-blur-xl dark:border-white/10 dark:bg-neutral-900/80">
            <p className="truncate text-xs font-semibold text-neutral-900 dark:text-neutral-100">{tip.title}</p>
            <p className="mt-0.5 truncate text-[11px] text-neutral-500 dark:text-neutral-400">{tip.artist}</p>
            <p className="mt-1.5 text-[10px] font-medium text-stage-instruct">{tip.hint}</p>
          </div>
        </Html>
      )}

      {cards.map((c) => (
        <Html
          key={c.key}
          position={lift(c.position, c.k ?? PATH_LIFT)}
          center
          zIndexRange={[30, 0]}
          style={{ pointerEvents: 'auto' }}
        >
          {c.node}
        </Html>
      ))}

      <CameraRig target={focus} flyKey={flyKey} controls={controls} />
      <OrbitControls
        ref={controls}
        enablePan={false}
        enableDamping
        dampingFactor={0.08}
        rotateSpeed={0.5}
        zoomSpeed={0.8}
        minDistance={2.6}
        maxDistance={9}
      />
    </>
  )
}

export default function SphereView(props) {
  return (
    <Canvas camera={{ position: [0, 0, 6.2], fov: 42 }} gl={{ antialias: true, alpha: true }} dpr={[1, 2]}>
      <Scene {...props} />
    </Canvas>
  )
}
