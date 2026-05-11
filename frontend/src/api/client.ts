import type { SearchResult, VizData } from "./types";

export async function fetchVizData(signal?: AbortSignal): Promise<VizData> {
	const resp = await fetch("/viz/data", { signal });
	if (!resp.ok) {
		throw new Error(`/viz/data returned ${resp.status}: ${await resp.text()}`);
	}
	return (await resp.json()) as VizData;
}

export async function fetchSearch(
	q: string,
	topK = 10,
	signal?: AbortSignal,
): Promise<SearchResult[]> {
	const url = `/viz/search?q=${encodeURIComponent(q)}&top_k=${topK}`;
	const resp = await fetch(url, { signal });
	if (!resp.ok) {
		throw new Error(`/viz/search returned ${resp.status}`);
	}
	return (await resp.json()) as SearchResult[];
}
