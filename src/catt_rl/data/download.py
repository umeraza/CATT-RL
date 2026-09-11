"""Optional Yahoo Finance research-data downloader."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = ("date", "ticker", "open", "high", "low", "close", "adj_close", "volume")


def load_ticker_file(path: str | Path) -> list[tuple[str, str]]:
    frame = pd.read_csv(path)
    if "ticker" not in frame.columns:
        raise ValueError("Ticker file must contain a 'ticker' column")
    yahoo = frame["yahoo_ticker"] if "yahoo_ticker" in frame else frame["ticker"]
    pairs = [
        (str(source).strip(), str(target).strip())
        for source, target in zip(yahoo, frame["ticker"], strict=True)
    ]
    pairs = [(source, target) for source, target in pairs if source and target]
    if not pairs:
        raise ValueError("Ticker file contains no usable symbols")
    return pairs


def _one_ticker_frame(data: pd.DataFrame, source: str, count: int) -> pd.DataFrame:
    if isinstance(data.columns, pd.MultiIndex):
        level0 = set(map(str, data.columns.get_level_values(0)))
        if source in level0:
            frame = data[source].copy()
        else:
            frame = data.xs(source, axis=1, level=1).copy()
    elif count == 1:
        frame = data.copy()
    else:
        return pd.DataFrame()
    frame.columns = [str(column).strip().lower().replace(" ", "_") for column in frame.columns]
    if "adj_close" not in frame and "close" in frame:
        frame["adj_close"] = frame["close"]
    return frame


def download_yahoo(
    tickers_path: str | Path,
    start: str,
    end: str,
    output: str | Path,
    threads: bool = True,
) -> Path:
    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover - exercised only without optional runtime
        raise RuntimeError("Install yfinance to use the downloader") from exc

    pairs = load_ticker_file(tickers_path)
    source_tickers = [source for source, _ in pairs]
    data = yf.download(
        source_tickers,
        start=start,
        end=end,
        interval="1d",
        group_by="ticker",
        auto_adjust=False,
        actions=False,
        threads=threads,
        progress=False,
    )
    rows: list[pd.DataFrame] = []
    missing: list[str] = []
    for source, canonical in pairs:
        frame = _one_ticker_frame(data, source, len(pairs))
        if frame.empty or "close" not in frame or frame["close"].dropna().empty:
            missing.append(canonical)
            continue
        frame = frame.reset_index()
        date_column = frame.columns[0]
        frame = frame.rename(columns={date_column: "date"})
        frame["ticker"] = canonical
        for column in REQUIRED_COLUMNS:
            if column not in frame and column not in {"date", "ticker"}:
                frame[column] = float("nan")
        rows.append(frame[list(REQUIRED_COLUMNS)])
    if not rows:
        raise RuntimeError("No usable Yahoo Finance rows were returned")
    output_frame = pd.concat(rows, ignore_index=True)
    output_frame["date"] = pd.to_datetime(output_frame["date"]).dt.tz_localize(None)
    output_frame = output_frame.sort_values(["date", "ticker"]).drop_duplicates(
        ["date", "ticker"], keep="last"
    )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    compression = "gzip" if output.suffix == ".gz" else None
    output_frame.to_csv(output, index=False, compression=compression)
    if missing:
        output.with_suffix(output.suffix + ".missing.txt").write_text(
            "\n".join(missing) + "\n", encoding="utf-8"
        )
    return output
