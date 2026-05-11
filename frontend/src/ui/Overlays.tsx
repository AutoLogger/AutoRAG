interface OverlaysProps {
	loading: boolean;
	empty: boolean;
	error: string | null;
}

export function Overlays({
	loading,
	empty,
	error,
}: OverlaysProps): JSX.Element {
	return (
		<>
			<div
				id="loading-overlay"
				className={`overlay${loading ? " visible" : ""}`}
			>
				<div className="overlay-icon" />
				<div className="overlay-text">loading visualization…</div>
			</div>

			<div
				id="empty-overlay"
				className={`overlay${empty && !loading && !error ? " visible" : ""}`}
			>
				<div className="overlay-text pulse">
					no topic data — run{" "}
					<span style={{ color: "var(--accent)" }}>autorag transcribe</span>{" "}
					first
				</div>
			</div>

			<div id="error-overlay" className={`overlay${error ? " visible" : ""}`}>
				<div id="error-msg" className="overlay-text error">
					{error}
				</div>
			</div>
		</>
	);
}
