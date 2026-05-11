import type { VizData } from "../api/types";
import { useVizStore } from "../state/vizStore";
import { Legend } from "./Legend";
import { SearchBox } from "./SearchBox";
import { SizeLegend } from "./SizeLegend";
import { TopicList } from "./TopicList";

interface RailProps {
	data: VizData;
}

export function Rail({ data }: RailProps): JSX.Element {
	const colorMode = useVizStore((s) => s.colorMode);
	const toggleColorMode = useVizStore((s) => s.toggleColorMode);
	const edgesVisible = useVizStore((s) => s.edgesVisible);
	const toggleEdges = useVizStore((s) => s.toggleEdges);

	const stats = `${data.total_topics} topics · ${data.total_clips} clips · ${data.total_clusters} clusters`;

	return (
		<div id="rail">
			<div id="rail-header">
				<h1>Topic Constellation</h1>
				<div id="stats">{stats}</div>
			</div>
			<Legend data={data} mode={colorMode} />
			<SizeLegend />
			<div id="controls">
				<button
					type="button"
					className={`ctrl-btn${colorMode === "cluster" ? " active" : ""}`}
					id="btn-color-mode"
					onClick={toggleColorMode}
				>
					{colorMode === "clip" ? "Color: By Clip" : "Color: By Cluster"}
				</button>
				<button
					type="button"
					className={`ctrl-btn${edgesVisible ? " active" : ""}`}
					id="btn-edges"
					onClick={toggleEdges}
				>
					{edgesVisible ? "Edges: Visible" : "Edges: Hidden"}
				</button>
			</div>
			<SearchBox />
			<div id="topic-list-header">
				<span id="topic-list-label">Topics</span>
			</div>
			<TopicList data={data} />
		</div>
	);
}
