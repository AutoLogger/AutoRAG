// Pointer-following tooltip for the hovered 3D topic. Driven entirely by
// store.tooltip (set by the Scene's shared raycaster). The #tooltip / .tt-*
// styles already live in styles.css; visibility is the `visible` class.

import { useVizStore } from "../state/vizStore";

export function Tooltip(): JSX.Element {
	const { point, x, y } = useVizStore((s) => s.tooltip);

	if (!point) return <div id="tooltip" />;

	return (
		<div id="tooltip" className="visible" style={{ left: x + 18, top: y - 8 }}>
			<div className="tt-title">{point.topic_title}</div>
			<div className="tt-meta">
				{point.clip_title}
				<br />L{point.level} · cluster {point.cluster_id} · {point.number} ·{" "}
				{point.start_s.toFixed(1)}s
			</div>
			{point.summary && <div className="tt-summary">{point.summary}</div>}
		</div>
	);
}
