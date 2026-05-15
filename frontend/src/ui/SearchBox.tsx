import { useState } from "react";
import { useDebouncedSearch } from "../hooks/useDebouncedSearch";
import { useVizStore } from "../state/vizStore";

export function SearchBox(): JSX.Element {
	const [query, setQuery] = useState("");
	useDebouncedSearch(query);
	const results = useVizStore((s) => s.searchResults);
	const setFocusIndex = useVizStore((s) => s.setFocusIndex);

	return (
		<div id="search-wrap">
			<input
				id="search-input"
				type="text"
				placeholder="search topics…"
				autoComplete="off"
				value={query}
				onChange={(e) => setQuery(e.target.value)}
			/>
			<div id="search-results">
				{results.map((r) => (
					<button
						type="button"
						className="search-hit"
						key={`${r.clip_id}|${r.point_index}`}
						title={r.topic_title}
						onClick={() => setFocusIndex(r.point_index)}
					>
						<span className="search-hit-title">
							{r.topic_title.slice(0, 32)}
						</span>
						<span className="search-hit-score">
							{Math.round(r.similarity * 100)}%
						</span>
					</button>
				))}
			</div>
		</div>
	);
}
