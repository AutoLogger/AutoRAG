// White glow over every current semantic-search hit (store.searchResults),
// so matches stand out in the cloud before the user clicks one to focus it.

import { useMemo } from "react";
import { AdditiveBlending, type Texture } from "three";
import { useVizStore } from "../state/vizStore";
import type { Layout } from "./layout";

interface SearchHighlightProps {
	layout: Layout;
	glow: Texture;
}

export function SearchHighlight({
	layout,
	glow,
}: SearchHighlightProps): JSX.Element | null {
	const results = useVizStore((s) => s.searchResults);

	const positions = useMemo(() => {
		const arr = new Float32Array(results.length * 3);
		results.forEach((r, i) => {
			const p = layout.worldPositions[r.point_index];
			if (!p) return;
			arr[i * 3] = p.x;
			arr[i * 3 + 1] = p.y;
			arr[i * 3 + 2] = p.z;
		});
		return arr;
	}, [results, layout]);

	if (results.length === 0) return null;

	return (
		<points key={results.length}>
			<bufferGeometry>
				<bufferAttribute attach="attributes-position" args={[positions, 3]} />
			</bufferGeometry>
			<pointsMaterial
				map={glow}
				color={0xffffff}
				size={0.55}
				opacity={0.9}
				transparent
				depthWrite={false}
				blending={AdditiveBlending}
				sizeAttenuation
			/>
		</points>
	);
}
