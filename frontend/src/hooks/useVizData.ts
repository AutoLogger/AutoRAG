import { useEffect, useState } from "react";
import { fetchVizData } from "../api/client";
import type { VizData } from "../api/types";

export interface VizDataState {
	data: VizData | null;
	loading: boolean;
	error: string | null;
	empty: boolean;
}

export function useVizData(): VizDataState {
	const [state, setState] = useState<VizDataState>({
		data: null,
		loading: true,
		error: null,
		empty: false,
	});

	useEffect(() => {
		const controller = new AbortController();
		fetchVizData(controller.signal)
			.then((data) => {
				setState({
					data,
					loading: false,
					error: null,
					empty: data.total_topics === 0,
				});
			})
			.catch((err: unknown) => {
				if (err instanceof DOMException && err.name === "AbortError") return;
				setState({
					data: null,
					loading: false,
					error: err instanceof Error ? err.message : String(err),
					empty: false,
				});
			});
		return () => {
			controller.abort();
		};
	}, []);

	return state;
}
