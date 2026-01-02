#!/usr/bin/env python3
"""Verify v3.5 cash-state refactor against a pre-refactor run."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _load_equity_curve(path: Path) -> pd.Series:
    df = pd.read_csv(path)
    if "portfolio_equity" not in df.columns or "date" not in df.columns:
        raise ValueError(f"Missing portfolio_equity/date columns in {path}")
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date")["portfolio_equity"].sort_index()


def _load_state_log(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "state" not in df.columns:
        raise ValueError(f"Missing state column in {path}")
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    return df


def _map_legacy_cash(state: str, invested_flag: bool) -> str:
    if state == "CASH":
        return "CASH_INVESTED" if invested_flag else "CASH_TRUE"
    return state


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify CASH state refactor consistency")
    parser.add_argument("--pre", required=True, type=Path, help="Pre-refactor output dir")
    parser.add_argument("--post", required=True, type=Path, help="Post-refactor output dir")
    parser.add_argument("--allow-legacy", action="store_true", help="Allow legacy CASH names in post state_log")
    args = parser.parse_args()

    pre_equity = _load_equity_curve(args.pre / "equity_curve.csv")
    post_equity = _load_equity_curve(args.post / "equity_curve.csv")

    common = pre_equity.index.intersection(post_equity.index)
    if len(common) == 0:
        raise ValueError("No overlapping dates in equity curves")

    equity_diff = (pre_equity.reindex(common) - post_equity.reindex(common)).abs().max()
    if equity_diff > 1e-10:
        raise AssertionError(f"Equity curve mismatch: max abs diff {equity_diff:.12f}")

    pre_state = _load_state_log(args.pre / "state_log.csv")
    post_state = _load_state_log(args.post / "state_log.csv")

    if "invested_flag" not in pre_state.columns:
        raise ValueError("Pre state_log.csv must include invested_flag")

    if "date" in pre_state.columns and "date" in post_state.columns:
        pre_state = pre_state.sort_values("date").reset_index(drop=True)
        post_state = post_state.sort_values("date").reset_index(drop=True)

    if len(pre_state) != len(post_state):
        raise AssertionError(f"State log length mismatch: {len(pre_state)} vs {len(post_state)}")

    if not args.allow_legacy and (post_state["state"] == "CASH").any():
        raise AssertionError("Legacy CASH state found in post state_log.csv")

    mapped_pre = [
        _map_legacy_cash(state, bool(inv))
        for state, inv in zip(pre_state["state"], pre_state["invested_flag"])
    ]

    mapped_post = []
    if (post_state["state"] == "CASH").any():
        if "invested_flag" not in post_state.columns:
            raise ValueError("Post state_log.csv has legacy CASH but no invested_flag")
        mapped_post = [
            _map_legacy_cash(state, bool(inv))
            for state, inv in zip(post_state["state"], post_state.get("invested_flag", [False] * len(post_state)))
        ]
    else:
        mapped_post = list(post_state["state"])

    mismatches = [
        (idx, pre_s, post_s)
        for idx, (pre_s, post_s) in enumerate(zip(mapped_pre, mapped_post))
        if pre_s != post_s
    ]

    if mismatches:
        idx, pre_s, post_s = mismatches[0]
        raise AssertionError(
            f"State mapping mismatch at row {idx}: pre={pre_s} post={post_s}"
        )

    print("OK: Equity curve and cash-state mapping are consistent.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
