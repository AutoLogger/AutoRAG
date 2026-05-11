import { useState } from "react";

interface SearchBoxProps {
	onQuery?: (q: string) => void;
}

export function SearchBox({ onQuery }: SearchBoxProps): JSX.Element {
	const [query, setQuery] = useState("");

	return (
		<div id="search-wrap">
			<input
				id="search-input"
				type="text"
				placeholder="search topics…"
				autoComplete="off"
				value={query}
				onChange={(e) => {
					const next = e.target.value;
					setQuery(next);
					onQuery?.(next);
				}}
			/>
			<div id="search-results" />
		</div>
	);
}
