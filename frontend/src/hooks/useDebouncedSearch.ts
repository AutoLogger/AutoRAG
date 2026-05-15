// Debounced semantic search: 350ms after the query settles, fetch /viz/search
// and push the hits into the store (consumed by SearchBox's result list and
// the 3D SearchHighlight). In-flight requests are aborted on the next keystroke.

import { useEffect } from "react";
import { fetchSearch } from "../api/client";
import { useVizStore } from "../state/vizStore";

export function useDebouncedSearch(query: string, delay = 350): void {
	const setSearchResults = useVizStore((s) => s.setSearchResults);

	useEffect(() => {
		const q = query.trim();
		if (!q) {
			setSearchResults([]);
			return;
		}
		const controller = new AbortController();
		const timer = setTimeout(() => {
			fetchSearch(q, 10, controller.signal)
				.then(setSearchResults)
				.catch((err: unknown) => {
					if (err instanceof DOMException && err.name === "AbortError") return;
					setSearchResults([]);
				});
		}, delay);
		return () => {
			clearTimeout(timer);
			controller.abort();
		};
	}, [query, delay, setSearchResults]);
}
