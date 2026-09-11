"""Optional bridge for exporting Qlib's official Alpha158 feature handler."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def export_alpha158(
    provider_uri: str,
    instruments: str,
    start: str,
    end: str,
    output: str | Path,
    selected_columns_file: str | Path | None = None,
) -> Path:
    try:
        import qlib
        from qlib.config import REG_US
        from qlib.contrib.data.handler import Alpha158
    except ImportError as exc:  # pragma: no cover - optional heavy dependency
        raise RuntimeError("Install the optional dependency with pip install -e '.[qlib]'") from exc

    qlib.init(provider_uri=provider_uri, region=REG_US)
    handler = Alpha158(
        instruments=instruments,
        start_time=start,
        end_time=end,
        fit_start_time=start,
        fit_end_time=end,
        infer_processors=[],
        learn_processors=[],
    )
    frame = handler.fetch(col_set="feature")
    if not isinstance(frame, pd.DataFrame):
        frame = pd.DataFrame(frame)
    frame = frame.reset_index()
    aliases = {str(column).lower(): column for column in frame.columns}
    instrument_column = aliases.get("instrument") or aliases.get("ticker")
    datetime_column = aliases.get("datetime") or aliases.get("date")
    if instrument_column is None or datetime_column is None:
        raise RuntimeError("Unexpected Qlib Alpha158 index schema")
    frame = frame.rename(columns={instrument_column: "ticker", datetime_column: "date"})
    if selected_columns_file is not None:
        selected = [
            line.strip()
            for line in Path(selected_columns_file).read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        missing = set(selected) - set(frame.columns)
        if missing:
            raise ValueError(f"Selected Alpha158 columns are missing: {sorted(missing)}")
        frame = frame[["date", "ticker", *selected]]
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False, compression="gzip" if output.suffix == ".gz" else None)
    return output
