import { useVizData } from "./hooks/useVizData";
import { Scene } from "./three/Scene";
import { Overlays } from "./ui/Overlays";
import { Rail } from "./ui/Rail";
import { SceneBoundary } from "./ui/SceneBoundary";
import { Tooltip } from "./ui/Tooltip";

export function App(): JSX.Element {
	const { data, loading, error, empty } = useVizData();
	const showScene = !!data && !empty && !error;

	return (
		<>
			{showScene && (
				<SceneBoundary>
					<Scene data={data} />
				</SceneBoundary>
			)}
			<Overlays loading={loading} empty={empty} error={error} />
			{showScene && <Rail data={data} />}
			<Tooltip />
		</>
	);
}
