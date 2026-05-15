// The WebGL constellation. Mirrors the original viz.html scene: dark
// background + exponential fog, a faint starfield and floor grid, one glowing
// <points> object per topic level, additive knowledge-graph edges, a private
// raycaster driving the tooltip + rail sync, and OrbitControls whose target
// snaps to the search-focused topic.

import { OrbitControls } from "@react-three/drei";
import { Canvas, useThree } from "@react-three/fiber";
import { useCallback, useEffect, useMemo, useRef } from "react";
import {
	ACESFilmicToneMapping,
	type GridHelper,
	Raycaster,
	Vector2,
} from "three";
import type { VizData } from "../api/types";
import { useVizStore } from "../state/vizStore";
import { Edges } from "./Edges";
import { getGlowTexture } from "./glowTexture";
import { HighlightGlow } from "./HighlightGlow";
import { computeLayout, type Layout } from "./layout";
import { PointsLayer, type RegEntry } from "./PointsLayer";
import { SearchHighlight } from "./SearchHighlight";

interface OrbitControlsLike {
	target: { set: (x: number, y: number, z: number) => void };
	update: () => void;
}

function Starfield(): JSX.Element {
	const positions = useMemo(() => {
		const n = 800;
		const arr = new Float32Array(n * 3);
		for (let i = 0; i < n; i++) {
			const theta = Math.random() * Math.PI * 2;
			const phi = Math.acos(2 * Math.random() - 1);
			const r = 70 + Math.random() * 40;
			arr[i * 3] = r * Math.sin(phi) * Math.cos(theta);
			arr[i * 3 + 1] = r * Math.sin(phi) * Math.sin(theta);
			arr[i * 3 + 2] = r * Math.cos(phi);
		}
		return arr;
	}, []);
	return (
		<points>
			<bufferGeometry>
				<bufferAttribute attach="attributes-position" args={[positions, 3]} />
			</bufferGeometry>
			<pointsMaterial color={0x8899bb} size={0.04} sizeAttenuation />
		</points>
	);
}

function GridFloor(): JSX.Element {
	const ref = useRef<GridHelper>(null);
	useEffect(() => {
		const g = ref.current;
		if (!g || Array.isArray(g.material)) return;
		g.material.transparent = true;
		g.material.opacity = 0.3;
	}, []);
	return (
		<gridHelper
			ref={ref}
			args={[20, 24, 0x1e2d45, 0x1e2d45]}
			position={[0, -4.5, 0]}
		/>
	);
}

function FocusController({ layout }: { layout: Layout }): null {
	const controls = useThree((s) => s.controls) as OrbitControlsLike | null;
	const focusIndex = useVizStore((s) => s.focusIndex);
	useEffect(() => {
		if (focusIndex === null || !controls) return;
		const p = layout.worldPositions[focusIndex];
		if (!p) return;
		controls.target.set(p.x, p.y, p.z);
		controls.update();
	}, [focusIndex, controls, layout]);
	return null;
}

function HoverController({
	registryRef,
	data,
}: {
	registryRef: React.MutableRefObject<RegEntry[]>;
	data: VizData;
}): null {
	const camera = useThree((s) => s.camera);
	const gl = useThree((s) => s.gl);
	const setHoverIndex = useVizStore((s) => s.setHoverIndex);
	const setTooltip = useVizStore((s) => s.setTooltip);
	const clearTooltip = useVizStore((s) => s.clearTooltip);

	useEffect(() => {
		const el = gl.domElement;
		const ndc = new Vector2();
		const raycaster = new Raycaster();
		raycaster.params.Points.threshold = 0.18;
		function onMove(e: PointerEvent): void {
			const rect = el.getBoundingClientRect();
			ndc.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
			ndc.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
			raycaster.setFromCamera(ndc, camera);

			let bestDist = Number.POSITIVE_INFINITY;
			let bestGi = -1;
			for (const reg of registryRef.current) {
				const hits = raycaster.intersectObject(reg.points, false);
				const hit = hits[0];
				if (hit && (hit.distanceToRay ?? Number.POSITIVE_INFINITY) < bestDist) {
					bestDist = hit.distanceToRay ?? Number.POSITIVE_INFINITY;
					bestGi = reg.indices[hit.index ?? -1] ?? -1;
				}
			}

			if (bestGi >= 0) {
				setHoverIndex(bestGi);
				setTooltip({ point: data.points[bestGi], x: e.clientX, y: e.clientY });
			} else {
				setHoverIndex(null);
				clearTooltip();
			}
		}
		el.addEventListener("pointermove", onMove);
		return () => el.removeEventListener("pointermove", onMove);
	}, [camera, gl, registryRef, data, setHoverIndex, setTooltip, clearTooltip]);
	return null;
}

interface SceneProps {
	data: VizData;
}

export function Scene({ data }: SceneProps): JSX.Element {
	const colorMode = useVizStore((s) => s.colorMode);
	const edgesVisible = useVizStore((s) => s.edgesVisible);

	const layout = useMemo(() => computeLayout(data.points), [data.points]);
	const clipIndex = useMemo(
		() => Object.fromEntries(data.clip_ids.map((id, i) => [id, i])),
		[data.clip_ids],
	);
	const glow = useMemo(() => getGlowTexture(), []);
	const levels = useMemo(
		() => [...layout.byLevel.keys()].sort((a, b) => a - b),
		[layout],
	);

	const registryRef = useRef<RegEntry[]>([]);
	const register = useCallback((entry: RegEntry) => {
		registryRef.current.push(entry);
		return () => {
			registryRef.current = registryRef.current.filter((r) => r !== entry);
		};
	}, []);

	return (
		<Canvas
			style={{ position: "fixed", inset: 0 }}
			dpr={[1, 2]}
			camera={{ fov: 55, near: 0.01, far: 600, position: [0, 0, 9] }}
			gl={{ antialias: true }}
			onCreated={({ gl }) => {
				gl.toneMapping = ACESFilmicToneMapping;
				gl.toneMappingExposure = 1.3;
			}}
		>
			<color attach="background" args={[0x0d1117]} />
			<fogExp2 attach="fog" args={[0x0d1117, 0.018]} />
			<Starfield />
			<GridFloor />
			{levels.map((level) => (
				<PointsLayer
					key={level}
					level={level}
					indices={layout.byLevel.get(level) ?? []}
					points={data.points}
					layout={layout}
					colorMode={colorMode}
					clipIndex={clipIndex}
					glow={glow}
					register={register}
				/>
			))}
			<Edges edges={data.edges} layout={layout} visible={edgesVisible} />
			<HighlightGlow
				points={data.points}
				layout={layout}
				colorMode={colorMode}
				clipIndex={clipIndex}
				glow={glow}
			/>
			<SearchHighlight layout={layout} glow={glow} />
			<OrbitControls
				makeDefault
				enableDamping
				dampingFactor={0.06}
				minDistance={1}
				maxDistance={80}
			/>
			<FocusController layout={layout} />
			<HoverController registryRef={registryRef} data={data} />
		</Canvas>
	);
}
