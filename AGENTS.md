# MT5AutoTrader AI Handoff Rules

## Repository layout

- ALL Python code lives in `src/` (app.py, config.py, trading/, model_core/, web/, tests/, ...). The project root intentionally holds only user-facing files (bat, txt, md) plus runtime data (strategies/, checkpoints/, logs/, portfolio_state.json, trader_config.json).
- Two path constants matter: `SRC` = src/ (on sys.path, web dir), `ROOT` = project root (config/logs/strategies/STOP_SIGNAL). config.py exposes ROOT_DIR; app.py and trading/runner.py define their own — keep them consistent when moving files.
- Subprocess spawns from the dashboard: training/backtest run `src/train_file.py` etc. with cwd=project root (their relative writes land at root); the runner runs `python -m trading.runner` with cwd=src (its ROOT constant points back at the project root).
- Training timeframe is selectable (M5..D1); parquet filenames are `{symbol}_{TF}.parquet`. `training_history_{stem}.json` at root feeds `/api/training/curve`.
- Training is CPU-only by design (formula tensors are tiny; GPU/DirectML was measured slower and the feature was removed). Do not reintroduce device selection or install GPU packages without explicit user consent.
- Python interpreter lookup order (app.py `_venv_python()` and `start.bat` agree): project `.venv` → `fallback_python.txt` (machine-specific absolute path, gitignored) → `sys.executable`/PATH python. Never hardcode an external project's venv path in code.

## Windows launch rules

- The reliable user entry points are the ASCII files `start.bat` and `stop.bat`. `stop.bat` delegates to `scripts\stop_all.ps1`. (Chinese-named wrappers were removed on user request — do not recreate them.)
- **Write every `.bat` with CRLF line endings.** LF-only batch files silently corrupt `set "VAR=value"` on this machine: the dashboard fell back from the venv interpreter to the system Python because `if not exist "%PY%"` misread an LF-terminated variable. Verify with a byte check, not by eye.
- Do not put long PowerShell programs inside a `.bat`: Git Bash, CMD quoting, `%` expansion, and `$variable` expansion can corrupt them. Put them in an ASCII-content `.ps1` and call it with `powershell -File`.
- Do not rely on a Chinese-content `.ps1` without a UTF-8 BOM. Windows PowerShell 5.1 may decode a no-BOM UTF-8 script using the system code page and produce parser errors. ASCII content avoids the problem entirely.
- Do not use `$PID` as a PowerShell variable name: it is a built-in read-only automatic variable.
- Check port `8900` before starting. A second start must open the existing page and must not create another dashboard process.
- The venv `Scripts\python.exe` on this machine is a redirector: it spawns a child `Python311\python.exe` that actually binds the port. Two OS processes for one logical dashboard is expected. Stop logic must kill the port owner **and** its python parent, which `stop_all.ps1` does.
- Stop the runner with `STOP_SIGNAL`; use PowerShell `Stop-Process` only as a verified PID fallback. Do not use `wmic` or `os.kill(pid, 0)`.
- Do not verify batch files by running them through Git Bash `cmd.exe /c`. MSYS path conversion turns `/c` into a path and opens an interactive shell. Use `powershell.exe -NoProfile -Command "cmd.exe /c '<abs path>'"`.

## Logging rules

- Runner and dashboard loguru sinks must call `logger.remove()` first.
- File sinks must use `encoding="utf-8"`, `colorize=False`, and an explicit format without ANSI codes.
- The dashboard strips ANSI escape sequences before displaying legacy log lines.
- Do not append raw colored terminal output to user-facing logs.

## Server time

- MT5 timestamps (bars, deals, position open time) are "server wall clock" encoded as pseudo-Unix epochs — NOT UTC. This broker runs UTC+3.
- `MT5Client.server_time_offset()` estimates the offset from a fresh tick (`tick.time - time.time()`, rounded to 30 min). The dashboard exposes it as `/api/status.server_offset_sec` and `/api/mt5/history.server_offset_sec`; the frontend subtracts it before rendering (`tsToStr`).
- Never render a raw MT5 timestamp with `new Date(ts * 1000)` — it will be hours off for the user.

## Position management & history

- `/api/mt5/position/close|sl|tp` act on REAL MT5 positions by ticket (any position carrying the software's magic, manual or runner-opened). They set `client.dry_run = False` explicitly because they are user-confirmed actions, independent of the runner's dry-run config.
- The overview merges positions from the runner status file; in live mode the runner reports ALL magic positions from MT5 (not just its book), so manual positions are visible too.
- After a web close, a still-bound strategy may reopen the position on the next closed bar — the UI warns about this; suggest unbinding first.
- `trading/history.py::group_history_deals` merges `history_deals_get` output by `position_id` into trades (partial closes aggregated; profit includes commission + swap). Position ids with no exit deals are reported as "open" records.

## Trading safety

- `trader_config.json` defaults to `dry_run=true` and no bindings.
- Switching to real order mode requires an explicit UI confirmation.
- Manual orders require an explicit UI confirmation.
- `order_check` is read-only and must never be described as a filled order.
- Never send a real order for testing without the user specifying the exact symbol, direction, and lot size in the current request.
- A dry-run position in `portfolio_state.json` is not a real MT5 position. Always query MT5 read-only before claiming an order was filled or closed.

## MT5 execution rules

- Read `symbol_info(symbol).filling_mode` and try that fill mode first. On this broker `ETHUSD_` reports `filling_mode=1`, and only `ORDER_FILLING_FOK` (value `0`) is accepted; hardcoding IOC first returns retcode `10030 Unsupported filling mode`.
- Do not assume MetaTrader5 python constants map to a fixed order. Read them from the module.
- `order_check` retcode `0` with comment `Done` means the parameters are acceptable, not that anything traded.
- retcode `10027` (`TRADE_RETCODE_CLIENT_DISABLES_AT`) means the MT5 terminal's AutoTrading button is off. `terminal_info().trade_allowed` is `False` while `account_info().trade_allowed` can still be `True`. Software cannot enable it; surface it to the user. The dashboard exposes it as `/api/status.trade_allowed` and shows a header badge plus a warning.

## Support/Resistance (S/R) engine

- Vendored from github.com/rosemarycox5334-debug/Detect_support_and_resistance_levels (V3 fusion algorithm) into `src/trading/srlab/` — numpy-only, no scipy/sklearn. Keep `prob_models.json`/`score_calibration.json` next to it; they are optional (levels work without them, probabilities degrade to null).
- `src/trading/sr.py` is the adapter: `detect_levels(rates)` for display, `sl_tp_for_trade(...)` for open-time SL/TP. The V3 score has two axes: `edge` (ranking) and `p_stall` (where price stalls) — do not merge them back into one score.
- Open-time rule (runner `_open_position`): SL = the TIGHTER of the fixed initial stop (`stop_loss_pct`) and a valid S/R stop (zone edge ± `sr_buffer_atr`×ATR, distance clamped to `[sr_min_sl_pct, sr_max_sl_pct]`); fixed SL is never removed. TP: with partial-TP enabled and the model gate passing (`p_hold ≥ sr_partial_min_phold`), `partial` = first opposing zone front and full TP moves to the NEXT zone (or none); otherwise TP = first opposing zone front, clamped to `[sr_min_tp_pct, sr_max_tp_pct]`.
- Partial take-profit ("止盈一半"): the plan (trigger price, close_volume, remaining, done) is stored in the book entry as `sr_partial` at open time; `_check_partial_take_profits` runs every loop for BOTH modes, closes part of the position via `close_position(volume=...)` when bid/ask crosses the trigger, and marks `done`. Lots that cannot be split (e.g. 0.01 with step 0.01) skip the plan. In live mode, manual magic positions absent from the book are adopted (book entry with `manual: true`) and get a backfilled plan — the BTC/ETH style live positions therefore get partial TP without reopening.
- Manual override protocol: the dashboard's drag-confirm writes `logs/sr_partial_overrides.json` (ticket → {price, close_volume, ts}) via POST `/api/mt5/position/partial`; the runner loads it once per loop (mtime-cached), applies each ts only once to the matching book entry (price=0 cancels; invalid split marks done+skipped), and manual overrides apply even when `sr_enabled`/`sr_partial_enabled` are off.
- In dry-run the book emulates TP fills (`_check_dry_take_profits`); in live mode TP rides on the MT5 order. Always respect `stops_level` before attaching SL/TP to an order, or the whole order gets rejected.
- Manual SL protocol: SL setters (chart drag / positions-table 止损 button) hit POST `/api/mt5/position/sl`, which modifies MT5 directly for live positions (dry positions resolve via the runner status file) and always writes `logs/sl_overrides.json` (ticket → {sl, ts, applied_to_mt5}). The runner applies it to the book and sets `tickets[ticket].manual_sl = True`; `RiskParams.manual_sl_tickets` (rebuilt each loop from the book) exempts those tickets from the initial-stop safety net — a manually dragged SL is respected even when looser than `stop_loss_pct` (user explicitly removed the auto pull-back); ladder breakeven still applies.
- Config keys live under `risk` (`sr_enabled`, `sr_buffer_atr`, `sr_tp_buffer_atr`, `sr_min_sl_pct`, `sr_max_sl_pct`, `sr_min_tp_pct`, `sr_max_tp_pct`, `sr_partial_enabled`, `sr_partial_fraction`, `sr_partial_use_model`, `sr_partial_min_phold`) and hot-reload each runner loop.
- `_write_status` reports `positions` as a LIST (one entry per ticket, each with a `symbol` field) — the same symbol may hold several positions (manual + strategy); the frontend positions table and the chart's "操作仓位" selector both consume the list. Do not collapse it back to a symbol-keyed dict.
- `/api/sr/chart` (overview canvas chart) returns candles (closed + `forming`), zones, ATR/price. The chart JS (`drawSRChart`) draws zones, position lines from `runner_status.positions` (entry/SL/TP/`sr_partial`), supports wheel-zoom/drag-pan/crosshair. The vertical range is computed from VISIBLE CANDLES ONLY — position lines/levels must never feed the y-range (small timeframes flatten); off-range lines render as edge tags (↑/↓) and are not draggable until visible. Keep the zone list and the chart in ONE card on the overview — the user rejected a separate key-level list card as duplication.
- Drag-to-set on the chart goes through a confirm dialog (`askConfirm` supports an `inputs` array rendered as editable number fields in `#ov-inputs`; cb receives `{key: value}`): SL/TP dialogs allow typing an exact price; the partial dialog allows price + volume and posts to `/api/mt5/position/partial`. The chart symbol picker is an `<input>` backed by datalist `#mt5-symbol-list` populated from `/api/mt5/symbols` (ALL terminal symbols — the user explicitly wants every symbol, not just bindings).

## Stop-script & process model

- `scripts/stop_all.ps1` takes exactly ONE `Get-CimInstance` snapshot of python processes and does all parent/child matching in memory (cold WMI enumeration took the old 5-query script to ~30s; the new one is ~10s including the graceful-stop wait). Do not add more WMI queries.
- Graceful stop first: write STOP_SIGNAL, poll `runner_status.json` for `phase == "stopped"` up to 5s, then kill the computed process set (port owner tree + pid-file runner tree + root-path matches).
- venv `python.exe` is a redirector: `/api/runner/start` returns the redirector pid (stored in `logs/trading_runner.pid`) while `runner_status.pid` is the real Python311 child. Both must be handled (the tree-kill covers this).

## Training data

- The user-facing training flow may call `mt5_train.py`, which fetches H1 bars from the running MT5 terminal, saves an internal Parquet cache, then invokes `train_file.py`.
- The training core remains offline after the cache is written; this keeps training reproducible.
- The current machine's existing cache directory is `D:\K线数据`. MT5-fetched files are named `{symbol}_H1.parquet`; a symbol that already ends with `_` (e.g. `ETHUSD_`) produces a double underscore: `ETHUSD__H1.parquet`. `parse_parquet_filename` handles this correctly via `rsplit("_", 1)`.
- Resume training: `train_file.py` auto-resumes from `checkpoints/ckpt_{symbol}_step_*.pt` unless `--from-scratch`. The `--steps` value is a TOTAL step target — resuming with a value ≤ the checkpoint step is a no-op that just re-saves the strategy. A short 60-step run may never hit a checkpoint save boundary, leaving nothing to resume from.
- `TradingRunner._refresh_bindings` tracks strategy file mtime and hot-reloads a retrained strategy without restarting the runner (log line "检测到重新训练，已热更新").
