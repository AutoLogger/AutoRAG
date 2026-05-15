// One <points> object per topic level. Geometry is index-parallel to a slice
// of data.points; vertex colors are recomputed when the color mode toggles
// (clip ↔ cluster) without rebuilding positions. The THREE.Points object is
// registered with the Scene so the shared raycaster can hit-test it.

import { useEffect, useMemo, useRef } from "react";
import {
	AdditiveBlending,
	type Texture,
	type Points as ThreePoints,
} from "three";
import type { TopicPoint } from "../api/types";
import type { ColorMode } from "../state/vizStore";
import type { Layout } from "./layout";
import {
	clipColor,
	clusterColor,
	hexToRgb01,
	LEVEL_OPACITY,
	LEVEL_SIZE,
} from "./palettes";

export interface RegEntry {
	points: ThreePoints;
	indices: number[];
}

interface PointsLayerProps {
	level: 1 | 2 | 3;
	indices: number[];
	points: readonly TopicPoint[];
	layout: Layout;
	colorMode: ColorMode;
	clipIndex: Record<string, number>;
	glow: Texture;
	register: (entry: RegEntry) => () => void;
}

export function PointsLayer({
	level,
	indices,
	points,
	layout,
	colorMode,
	clipIndex,
	glow,
	register,
}: PointsLayerProps): JSX.Element {
	const ref = useRef<ThreePoints>(null);

	const positions = useMemo(() => {
		const arr = new Float32Array(indices.length * 3);
		indices.forEach((gi, i) => {
			const p = layout.worldPositions[gi];
			arr[i * 3] = p.x;
			arr[i * 3 + 1] = p.y;
			arr[i * 3 + 2] = p.z;
		});
		return arr;
	}, [indices, layout]);

	const colors = useMemo(() => {
		const arr = new Float32Array(indices.length * 3);
		indices.forEach((gi, i) => {
			const pt = points[gi];
			const hex =
				colorMode === "clip"
					? clipColor(clipIndex[pt.clip_id] ?? 0)
					: clusterColor(pt.cluster_id);
			const [r, g, b] = hexToRgb01(hex);
			arr[i * 3] = r;
			arr[i * 3 + 1] = g;
			arr[i * 3 + 2] = b;
		});
		return arr;
	}, [indices, points, colorMode, clipIndex]);

	useEffect(() => {
		const obj = ref.current;
		if (!obj) return;
		return register({ points: obj, indices });
	}, [register, indices]);

	return (
		<points ref={ref}>
			<bufferGeometry>
				<bufferAttribute attach="attributes-position" args={[positions, 3]} />
				<bufferAttribute
					key={colorMode}
					attach="attributes-color"
					args={[colors, 3]}
				/>
			</bufferGeometry>
			<pointsMaterial
				map={glow}
				size={LEVEL_SIZE[level]}
				opacity={LEVEL_OPACITY[level]}
				transparent
				depthWrite={false}
				blending={AdditiveBlending}
				sizeAttenuation
				vertexColors
			/>
		</points>
	);
}
