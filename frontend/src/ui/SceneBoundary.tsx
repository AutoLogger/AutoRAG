// r3f's <Canvas> throws synchronously if a WebGL context can't be created
// (headless/software GL, blocklisted GPU, lost context at init). Without a
// boundary that error unmounts the whole app — the user loses the rail and
// every overlay and just sees a blank page. This keeps the DOM UI alive and
// surfaces a legible reason instead.

import { Component, type ErrorInfo, type ReactNode } from "react";

interface Props {
	children: ReactNode;
}

interface State {
	failed: boolean;
}

export class SceneBoundary extends Component<Props, State> {
	state: State = { failed: false };

	static getDerivedStateFromError(): State {
		return { failed: true };
	}

	componentDidCatch(error: Error, info: ErrorInfo): void {
		console.error("3D scene failed to initialize:", error, info.componentStack);
	}

	render(): ReactNode {
		if (this.state.failed) {
			return (
				<div
					style={{
						position: "fixed",
						left: "calc(var(--rail-w) + 24px)",
						bottom: 20,
						zIndex: 15,
						fontFamily: "var(--mono)",
						fontSize: "0.7rem",
						color: "var(--text-lo)",
						border: "1px solid var(--border)",
						borderRadius: 6,
						padding: "8px 12px",
						background: "var(--panel-bg)",
						maxWidth: 360,
					}}
				>
					3D view unavailable — WebGL context could not be created. The topic
					list on the left still works.
				</div>
			);
		}
		return this.props.children;
	}
}
