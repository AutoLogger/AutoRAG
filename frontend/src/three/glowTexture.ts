// Soft radial-gradient sprite shared by every glowing point (level points,
// rail-hover highlight, search hits). Ported verbatim from viz.html
// makeGlowTexture() so the constellation keeps its bloom look.

import { CanvasTexture, type Texture } from "three";

let cached: Texture | null = null;

export function getGlowTexture(): Texture {
	if (cached) return cached;
	const size = 128;
	const c = document.createElement("canvas");
	c.width = size;
	c.height = size;
	const ctx = c.getContext("2d");
	if (!ctx) throw new Error("2D canvas context unavailable for glow sprite");
	const g = ctx.createRadialGradient(
		size / 2,
		size / 2,
		0,
		size / 2,
		size / 2,
		size / 2,
	);
	g.addColorStop(0, "rgba(255,255,255,1.0)");
	g.addColorStop(0.18, "rgba(255,255,255,0.85)");
	g.addColorStop(0.5, "rgba(255,255,255,0.25)");
	g.addColorStop(1, "rgba(255,255,255,0.0)");
	ctx.fillStyle = g;
	ctx.fillRect(0, 0, size, size);
	cached = new CanvasTexture(c);
	return cached;
}
