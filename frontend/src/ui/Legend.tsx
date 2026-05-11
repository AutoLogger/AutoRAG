import type { VizData } from "../api/types";
import type { ColorMode } from "../state/vizStore";
import { clipColor, clusterColor, hexToCss } from "../three/palettes";

interface LegendProps {
	data: VizData;
	mode: ColorMode;
}

const MAX_CLIP_ROWS = 8;
const MAX_CLUSTER_ROWS = 10;

export function Legend({ data, mode }: LegendProps): JSX.Element {
	if (mode === "cluster") {
		const count = Math.min(data.total_clusters, MAX_CLUSTER_ROWS);
		const rows = Array.from({ length: count }, (_, i) => {
			const color = hexToCss(clusterColor(i));
			return (
				// Cluster index IS the natural identifier — clusters are dense 0..N-1.
				// biome-ignore lint/suspicious/noArrayIndexKey: cluster id is the key
				<div className="legend-row" key={`cluster-${i}`}>
					<div
						className="legend-dot"
						style={{ background: color, color: color }}
					/>
					<div className="legend-label">Cluster {i}</div>
				</div>
			);
		});
		const overflow =
			data.total_clusters > MAX_CLUSTER_ROWS
				? data.total_clusters - MAX_CLUSTER_ROWS
				: 0;
		return (
			<div id="legend">
				{rows}
				{overflow > 0 && (
					<div
						className="legend-row"
						style={{ color: "var(--text-lo)", fontSize: "0.65rem" }}
					>
						+{overflow} more
					</div>
				)}
			</div>
		);
	}

	const ids = data.clip_ids.slice(0, MAX_CLIP_ROWS);
	const overflow =
		data.clip_ids.length > MAX_CLIP_ROWS
			? data.clip_ids.length - MAX_CLIP_ROWS
			: 0;
	return (
		<div id="legend">
			{ids.map((id, i) => {
				const color = hexToCss(clipColor(i));
				const fullLabel = data.clip_titles[id] ?? id;
				const label = fullLabel.slice(0, 28);
				return (
					<div className="legend-row" key={id}>
						<div
							className="legend-dot"
							style={{ background: color, color: color }}
						/>
						<div className="legend-label" title={fullLabel}>
							{label}
						</div>
					</div>
				);
			})}
			{overflow > 0 && (
				<div
					className="legend-row"
					style={{ color: "var(--text-lo)", fontSize: "0.65rem" }}
				>
					+{overflow} more
				</div>
			)}
		</div>
	);
}
