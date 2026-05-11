import { useVizData } from "./hooks/useVizData";
import { Overlays } from "./ui/Overlays";
import { Rail } from "./ui/Rail";

export function App(): JSX.Element {
	const { data, loading, error, empty } = useVizData();
	const showRail = !!data && !empty && !error;

	return (
		<>
			<Overlays loading={loading} empty={empty} error={error} />
			{showRail && <Rail data={data} />}
		</>
	);
}
