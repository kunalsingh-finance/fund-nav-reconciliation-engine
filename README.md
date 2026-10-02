# Fund NAV Reconciliation Engine

One-line problem: fund administrators and investment operations teams need a repeatable way to calculate daily NAV, compare it against custodian records, and surface breaks before reporting sign-off.

This project is a public-safe Python workflow that simulates a daily fund administration process:

```text
Trade blotter -> Position ledger -> Daily NAV -> Custodian comparison -> Break report -> Output pack
```

The bundled inputs are synthetic demo records. The project does not include client data, custodian files, bank records, account identifiers, credentials, or proprietary fund data.

## What I Built

- A synthetic fund input generator for trades, prices, FX, security master records, corporate actions, and custodian NAV records.
- A position-ledger builder that handles buys, cash contributions, dividends, and stock splits.
- A daily NAV engine that calculates market value, cash, total NAV, and NAV per share.
- A reconciliation engine that compares internal NAV and positions against custodian-style records.
- A break report that classifies NAV, position, price, missing-security and missing-NAV-source exceptions by severity.
- SQLite persistence for daily positions, NAV, reconciliation records, breaks, and run logs.
- CSV and JSON output packs for review.
- A pytest suite covering NAV, positions, and reconciliation behavior.

## Tools Used

Python, pandas, NumPy, SQLite, PyYAML, pytest, SQL, CSV, JSON.

## Key Features

- Synthetic demo data with intentional breaks on the final as-of date.
- Policy-driven tolerance setting in `policy.yaml`.
- Corporate-action support for dividends and stock splits.
- Fund-level run logging with success/failure metadata.
- Public-market-data builder kept optional and excluded from committed output data.
- Reproducible command-line workflow.

## Sample Output

Sample output from `outputs/2026-03-31_FUND1/summary.json`:

| Metric | Value |
|---|---:|
| Fund | FUND1 |
| As-of date | 2026-03-31 |
| Break count | 4 |
| High severity breaks | 2 |
| Output pack | `outputs/2026-03-31_FUND1/` |

The output pack includes:

- `positions.csv`
- `nav.csv`
- `reconciliation.csv`
- `break_report.csv`
- `summary.json`

## How To Run

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Generate synthetic input data:

```bash
python scripts/build_dummy_data.py --out-dir ./data --start 2024-01-01 --end 2026-03-31 --fund-id FUND1 --tickers AAPL MSFT GOOGL AMZN
```

Run the daily NAV and reconciliation workflow:

```bash
python -m src.run_daily --asof 2026-03-31 --data-dir ./data
```

Run tests:

```bash
pytest -q
```

The regression suite covers split chronology, reverse/multiple splits, dividend entitlement, missing custodian/internal records, duplicate NAV keys, invalid values, review status and clearing resolved breaks on reruns. GitHub Actions runs the suite on Linux and Windows.

## Reconciliation and corporate-action conventions

- A missing NAV record on either side is a high-severity exception, including a missing matching fund or date. Duplicate fund/date NAV keys and nonfinite NAV/cash values reject the run.
- A completed run with any exceptions is `REVIEW_REQUIRED` in its result, saved summary and database log. `SUCCESS` means the comparison completed without exceptions; it is not regulatory sign-off.
- Trades are recorded in the share units effective on their date. Date-only splits take effect before that day's trades, so each split changes only earlier quantities. For example, 100 shares before a 2:1 split plus 10 bought afterward produces 210 shares.
- Generated dividend flows use holdings entering the ex-date, adjusted for splits effective by that date. Trades on the ex-date are excluded from dividend entitlement; existing booked dividends are not generated again. Cash is recorded on the ex-date in this simplified demo rather than a separate payment date.
- The existing synthetic-price convention supplies a pre-split quote on the split date and divides it by that day's split ratio. Use prices consistent with this demo convention; vendor-adjusted prices require explicit normalization first.
- Rerunning a fund/date replaces its saved positions and exceptions, including clearing records after a position closes or a break resolves.
- Saved summary export paths are filenames relative to the output pack, so reports remain portable between machines.

## Project Structure

```text
fund-nav-reconciliation-engine/
|-- README.md
|-- requirements.txt
|-- policy.yaml
|-- data/
|-- outputs/
|-- scripts/
|   |-- build_dummy_data.py
|   `-- build_real_data.py
|-- sql/
|   `-- ddl.sql
|-- src/
|   |-- nav.py
|   |-- positions.py
|   |-- reconciliation.py
|   |-- report.py
|   `-- run_daily.py
`-- tests/
```

## Skills Demonstrated

- Investment operations workflow design
- NAV calculation and fund accounting logic
- Custodian reconciliation and exception reporting
- Position-ledger construction from transaction data
- Corporate-action handling
- SQL persistence and repeatable reporting outputs
- Python data engineering with pandas
- Testable finance workflow design

## Data And Confidentiality

- `data/` contains synthetic demo data generated for this public project.
- `outputs/` contains sample outputs generated from the synthetic demo.
- `data_real/`, local SQLite databases, caches, and credentials are intentionally excluded.
- Optional public market-data scripts require user-supplied inputs and should not be treated as source-of-truth fund data.

## Limitations

- The workflow is a simplified public demo, not a production fund-accounting platform.
- The custodian records are synthetic and intentionally include breaks.
- Tax lots, full accrual accounting, multi-currency settlement, fee waterfalls, and investor-level allocations are outside the current scope.
- The output pack is designed for workflow demonstration, not regulatory reporting.

## Future Improvements

- Add lot-level realized/unrealized P&L.
- Add multi-fund dashboard views.
- Add aging and owner fields for break-management workflow.
- Add Excel report export with sign-off summary.
- Add benchmark-level performance and attribution outputs.
