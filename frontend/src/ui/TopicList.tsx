import type { TopicPoint, VizData } from "../api/types";
import { useVizStore } from "../state/vizStore";
import { clipColor, hexToCss } from "../three/palettes";

interface TopicListProps {
	data: VizData;
}

interface ClipGroup {
	id: string;
	title: string;
	color: string;
	colorWithAlpha: string;
	points: { point: TopicPoint; index: number }[];
}

const MAX_TITLE_CHARS = 36;

function groupByClip(data: VizData): ClipGroup[] {
	const byClip = new Map<string, { point: TopicPoint; index: number }[]>();
	for (const id of data.clip_ids) byClip.set(id, []);
	data.points.forEach((point, index) => {
		const bucket = byClip.get(point.clip_id);
		if (bucket) bucket.push({ point, index });
	});

	const groups: ClipGroup[] = [];
	data.clip_ids.forEach((id, i) => {
		const points = byClip.get(id) ?? [];
		if (points.length === 0) return;
		const colorHex = clipColor(i);
		const color = hexToCss(colorHex);
		groups.push({
			id,
			title: data.clip_titles[id] ?? id,
			color,
			colorWithAlpha: `${color}11`,
			points,
		});
	});
	return groups;
}

export function TopicList({ data }: TopicListProps): JSX.Element {
	const groups = groupByClip(data);
	const setHoverIndex = useVizStore((s) => s.setHoverIndex);
	const hoverIndex = useVizStore((s) => s.hoverIndex);

	return (
		<div id="topic-list">
			{groups.map((group) => {
				const truncatedTitle = group.title.slice(0, 34);
				return (
					<div className="clip-section" key={group.id}>
						<div
							className="clip-section-header"
							style={
								{
									"--clip-color": group.color,
								} as React.CSSProperties
							}
						>
							<div
								className="clip-section-dot"
								style={{
									background: group.color,
									boxShadow: `0 0 5px ${group.color}`,
								}}
							/>
							<span className="clip-section-name" title={group.title}>
								{truncatedTitle}
							</span>
						</div>
						{group.points.map(({ point, index }) => {
							const displayTitle =
								point.topic_title.length > MAX_TITLE_CHARS
									? `${point.topic_title.slice(0, MAX_TITLE_CHARS)}…`
									: point.topic_title;
							const isActive = hoverIndex === index;
							return (
								// biome-ignore lint/a11y/noStaticElementInteractions: hover-only sync with 3D point; click-to-focus added in Phase E
								<div
									className={`topic-row level-${point.level}${
										isActive ? " active" : ""
									}`}
									key={`${point.clip_id}|${point.topic_title}|${index}`}
									title={point.topic_title}
									style={
										{
											"--row-color": group.color,
											"--row-hover-bg": group.colorWithAlpha,
										} as React.CSSProperties
									}
									onMouseEnter={() => setHoverIndex(index)}
									onMouseLeave={() => setHoverIndex(null)}
								>
									<span className="level-badge">L{point.level}</span>
									<span className="topic-num">{point.number}</span>
									<span className="topic-name">{displayTitle}</span>
								</div>
							);
						})}
					</div>
				);
			})}
		</div>
	);
}
