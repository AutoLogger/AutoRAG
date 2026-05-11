export function SizeLegend(): JSX.Element {
	return (
		<div id="size-legend">
			<div className="size-row">
				<div className="size-dot" style={{ width: 9, height: 9 }} />
				L1
			</div>
			<div className="size-row">
				<div className="size-dot" style={{ width: 6, height: 6 }} />
				L2
			</div>
			<div className="size-row">
				<div className="size-dot" style={{ width: 4, height: 4 }} />
				L3
			</div>
		</div>
	);
}
