// Hand-typed mirror of src/autorag/viz.py response schemas
// (TopicPoint, Edge, VizData, SearchResult).
// Keep in sync when the backend schemas change.

export interface TopicPoint {
	topic_title: string;
	clip_id: string;
	clip_title: string;
	level: 1 | 2 | 3;
	start_s: number;
	duration_s: number;
	number: string;
	summary: string;
	x: number;
	y: number;
	z: number;
	cluster_id: number;
}

export interface Edge {
	a: number;
	b: number;
	similarity: number;
}

export interface VizData {
	points: TopicPoint[];
	clip_ids: string[];
	clip_titles: Record<string, string>;
	total_topics: number;
	total_clips: number;
	edges: Edge[];
	total_clusters: number;
}

export interface SearchResult {
	point_index: number;
	topic_title: string;
	clip_title: string;
	clip_id: string;
	similarity: number;
	summary: string;
}
