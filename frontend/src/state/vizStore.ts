import { create } from "zustand";
import type { SearchResult, TopicPoint } from "../api/types";

export type ColorMode = "clip" | "cluster";

export interface TooltipState {
	point: TopicPoint | null;
	x: number;
	y: number;
}

export interface VizState {
	colorMode: ColorMode;
	edgesVisible: boolean;
	hoverIndex: number | null;
	focusIndex: number | null;
	tooltip: TooltipState;
	searchResults: SearchResult[];
	setColorMode: (mode: ColorMode) => void;
	toggleColorMode: () => void;
	setEdgesVisible: (visible: boolean) => void;
	toggleEdges: () => void;
	setHoverIndex: (i: number | null) => void;
	setFocusIndex: (i: number | null) => void;
	setTooltip: (t: TooltipState) => void;
	clearTooltip: () => void;
	setSearchResults: (results: SearchResult[]) => void;
}

export const useVizStore = create<VizState>((set) => ({
	colorMode: "clip",
	edgesVisible: false,
	hoverIndex: null,
	focusIndex: null,
	tooltip: { point: null, x: 0, y: 0 },
	searchResults: [],
	setColorMode: (mode) => set({ colorMode: mode }),
	toggleColorMode: () =>
		set((s) => ({ colorMode: s.colorMode === "clip" ? "cluster" : "clip" })),
	setEdgesVisible: (visible) => set({ edgesVisible: visible }),
	toggleEdges: () => set((s) => ({ edgesVisible: !s.edgesVisible })),
	setHoverIndex: (i) => set({ hoverIndex: i }),
	setFocusIndex: (i) => set({ focusIndex: i }),
	setTooltip: (t) => set({ tooltip: t }),
	clearTooltip: () => set({ tooltip: { point: null, x: 0, y: 0 } }),
	setSearchResults: (results) => set({ searchResults: results }),
}));
