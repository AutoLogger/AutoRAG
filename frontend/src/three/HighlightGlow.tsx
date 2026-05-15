// Pulsing glow marking the "active" topic — the focused one (search-hit click)
// takes precedence over the hovered one (rail row or 3D point). Mirrors the
// rail-hover highlight from the original viz.html, extended to also track the
// search-focus index so a clicked hit is visibly pinned in 3D.

import { useFrame } from "@react-three/fiber";
import { useRef } from "react";
import {
	AdditiveBlending,
	type Texture,
	type PointsMaterial as ThreePointsMaterial,
} from "three";
import type { TopicPoint } from "../api/types";
import type { ColorMode } from "../state/vizStore";
import { useVizStore } from "../state/vizStore";
import type { Layout } from "./layout";
import { clipColor, clusterColor } from "./palettes";

const SINGLE_VERTEX = new Float32Array([0, 0, 0]);

interface HighlightGlowProps {
	points: readonly TopicPoint[];
	layout: Layout;
	colorMode: ColorMode;
	clipIndex: Record<string, number>;
	glow: Texture;
}

export function HighlightGlow({
	points,
	layout,
	colorMode,
	clipIndex,
	glow,
}: HighlightGlowProps): JSX.Element | null {
	const hoverIndex = useVizStore((s) => s.hoverIndex);
	const focusIndex = useVizStore((s) => s.focusIndex);
	const matRef = useRef<ThreePointsMaterial>(null);
	const phase = useRef(0);

	const active = focusIndex ?? hoverIndex;

	useFrame(() => {
		const m = matRef.current;
		if (!m || active === null) return;
		phase.current += 0.07;
		m.size = 0.55 + 0.3 * Math.sin(phase.current);
		m.opacity = 0.75 + 0.2 * Math.sin(phase.current * 1.3);
	});

	if (active === null) return null;
	const p = layout.worldPositions[active];
	if (!p) return null;

	const pt = points[active];
	const color =
		colorMode === "clip"
			? clipColor(clipIndex[pt.clip_id] ?? 0)
			: clusterColor(pt.cluster_id);

	return (
		<points position={[p.x, p.y, p.z]}>
			<bufferGeometry>
				<bufferAttribute
					attach="attributes-position"
					args={[SINGLE_VERTEX, 3]}
				/>
			</bufferGeometry>
			<pointsMaterial
				ref={matRef}
				map={glow}
				color={color}
				size={0.65}
				opacity={0.95}
				transparent
				depthWrite={false}
				blending={AdditiveBlending}
				sizeAttenuation
			/>
		</points>
	);
}
