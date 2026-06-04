CREATE TABLE IF NOT EXISTS nav_daily_positions (
    date TEXT NOT NULL,
    fund_id TEXT NOT NULL,
    security_id TEXT NOT NULL,
    quantity_eod REAL NOT NULL,
    price_local REAL NOT NULL,
    security_currency TEXT NOT NULL,
    fx_to_base REAL NOT NULL,
    market_value_base REAL NOT NULL,
    corporate_action_flag TEXT,
    PRIMARY KEY (date, fund_id, security_id)
);

CREATE TABLE IF NOT EXISTS nav_daily_nav (
    date TEXT NOT NULL,
    fund_id TEXT NOT NULL,
    securities_mv REAL NOT NULL,
    cash_balance REAL NOT NULL,
    accrued_income REAL NOT NULL,
    total_nav REAL NOT NULL,
    shares_outstanding REAL NOT NULL,
    nav_per_share REAL NOT NULL,
    PRIMARY KEY (date, fund_id)
);

CREATE TABLE IF NOT EXISTS nav_reconciliation (
    date TEXT NOT NULL,
    fund_id TEXT NOT NULL,
    break_type TEXT NOT NULL,
    severity TEXT NOT NULL,
    internal_value REAL,
    custodian_value REAL,
    diff_bps REAL,
    details TEXT,
    resolution TEXT
);

CREATE TABLE IF NOT EXISTS nav_breaks (
    date TEXT NOT NULL,
    fund_id TEXT NOT NULL,
    break_type TEXT NOT NULL,
    severity TEXT NOT NULL,
    internal_value REAL,
    custodian_value REAL,
    diff_bps REAL,
    details TEXT,
    resolution TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS nav_run_log (
    run_ts TEXT NOT NULL,
    asof_date TEXT NOT NULL,
    fund_id TEXT NOT NULL,
    total_nav REAL,
    nav_per_share REAL,
    break_count INTEGER NOT NULL,
    high_severity_breaks INTEGER NOT NULL,
    data_dir TEXT NOT NULL,
    status TEXT NOT NULL
);
