// Geometry layout for the topic constellation.
//
// UMAP coordinates from /viz/data are NOT origin-centered (a single clip can
// sit at e.g. x≈-10, y≈25, z≈1.5 with a spread of only ~1-2 units). Rendering
// them raw leaves the whole cloud off-camera — which reads as "no embeddings".
// Same normalization as the original viz.html buildScene(): translate the
// bounding-box centroid to the origin, then scale the largest axis span to
// SCENE_SPAN world units so the camera at z=9 always frames the cloud.

import type { TopicPoint } from "../api/types";

export interface Vec3 {
	x: number;
	y: number;
	z: number;
}

export interface Layout {
	/** index-parallel to data.points — normalized world position per point */
	worldPositions: Vec3[];
	/** global point indices grouped by topic level (1/2/3) */
	byLevel: Map<1 | 2 | 3, number[]>;
}

const SCENE_SPAN = 7.0;

export function computeLayout(points: readonly TopicPoint[]): Layout {
	const worldPositions: Vec3[] = new Array(points.length);
	const byLevel = new Map<1 | 2 | 3, number[]>();
	if (points.length === 0) return { worldPositions, byLevel };

	const xs = points.map((p) => p.x);
	const ys = points.map((p) => p.y);
	const zs = points.map((p) => p.z);

	const minX = Math.min(...xs);
	const minY = Math.min(...ys);
	const minZ = Math.min(...zs);
	const maxX = Math.max(...xs);
	const maxY = Math.max(...ys);
	const maxZ = Math.max(...zs);

	const maxRange = Math.max(maxX - minX, maxY - minY, maxZ - minZ, 1e-6);
	const scale = SCENE_SPAN / maxRange;
	const cx = (maxX + minX) / 2;
	const cy = (maxY + minY) / 2;
	const cz = (maxZ + minZ) / 2;

	points.forEach((pt, i) => {
		worldPositions[i] = {
			x: (pt.x - cx) * scale,
			y: (pt.y - cy) * scale,
			z: (pt.z - cz) * scale,
		};
		const bucket = byLevel.get(pt.level);
		if (bucket) bucket.push(i);
		else byLevel.set(pt.level, [i]);
	});

	return { worldPositions, byLevel };
}
