// Color palettes for the topic constellation viz.
// Same values as the original viz.html.

export const PALETTE: readonly number[] = [
	0x7b61ff, // electric violet
	0x00d9ff, // cyan
	0xff6b6b, // coral
	0x69ff47, // neon green
	0xffb800, // amber
	0xff47d4, // magenta
	0x47ffd4, // teal
	0xff8c00, // orange
];

export const CLUSTER_PALETTE: readonly number[] = [
	0xffe156, // yellow
	0xff7f50, // coral-orange
	0x98ff98, // mint
	0xffb6c1, // light pink
	0xadd8e6, // light blue
	0xdda0dd, // plum
	0x90ee90, // light green
	0xf0e68c, // khaki
	0xe0ffff, // light cyan
	0xffa07a, // light salmon
	0xb0c4de, // steel blue
	0x20b2aa, // light sea green
];

export const LEVEL_SIZE: Readonly<Record<1 | 2 | 3, number>> = {
	1: 0.35,
	2: 0.22,
	3: 0.15,
};

export const LEVEL_OPACITY: Readonly<Record<1 | 2 | 3, number>> = {
	1: 1.0,
	2: 0.9,
	3: 0.75,
};

export function hexToCss(n: number): string {
	return `#${n.toString(16).padStart(6, "0")}`;
}

export function clipColor(clipIndex: number): number {
	return PALETTE[clipIndex % PALETTE.length] ?? 0xffffff;
}

export function clusterColor(clusterId: number): number {
	return CLUSTER_PALETTE[clusterId % CLUSTER_PALETTE.length] ?? 0xffffff;
}
