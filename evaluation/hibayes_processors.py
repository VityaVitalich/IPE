"""Custom HiBayes processors for preference evaluation analysis."""

from __future__ import annotations

from typing import TYPE_CHECKING

from hibayes.process import DataProcessor, process

if TYPE_CHECKING:
    from hibayes.analysis import AnalysisState
    from hibayes.ui import ModellingDisplay


@process
def filter_decided() -> DataProcessor:
    """Remove rows where score == 0.5 (unknown/refusal)."""

    def processor(
        state: AnalysisState,
        display: ModellingDisplay | None = None,
    ) -> AnalysisState:
        before = len(state.processed_data)
        state.processed_data = state.processed_data[
            state.processed_data["score"] != 0.5
        ].reset_index(drop=True)
        if display:
            display.logger.info(
                f"Filtered unknowns: {before} -> {len(state.processed_data)} rows"
            )
        return state

    return processor
