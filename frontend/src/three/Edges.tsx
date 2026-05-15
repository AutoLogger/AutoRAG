// Knowledge-graph edges: one additive blue line segment per Edge, between the
// normalized world positions of its two endpoint topics. Toggled by the rail's
// "Edges" button (store.edgesVisible).

import { useMemo } from "react";
import { AdditiveBlending } from "three";
import type { Edge } from "../api/types";
import type { Layout } from "./layout";

interface EdgesProps {
	edges: readonly Edge[];
	layout: Layout;
	visible: boolean;
}

export function Edges({
	edges,
	layout,
	visible,
}: EdgesProps): JSX.Element | null {
	const positions = useMemo(() => {
		const arr = new Float32Array(edges.length * 6);
		edges.forEach((e, i) => {
			const a = layout.worldPositions[e.a];
			const b = layout.worldPositions[e.b];
			if (!a || !b) return;
			arr[i * 6] = a.x;
			arr[i * 6 + 1] = a.y;
			arr[i * 6 + 2] = a.z;
			arr[i * 6 + 3] = b.x;
			arr[i * 6 + 4] = b.y;
			arr[i * 6 + 5] = b.z;
		});
		return arr;
	}, [edges, layout]);

	if (edges.length === 0) return null;

	return (
		<lineSegments visible={visible}>
			<bufferGeometry>
				<bufferAttribute attach="attributes-position" args={[positions, 3]} />
			</bufferGeometry>
			<lineBasicMaterial
				color={0x4488cc}
				opacity={0.25}
				transparent
				depthWrite={false}
				blending={AdditiveBlending}
			/>
		</lineSegments>
	);
}
