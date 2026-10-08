"""
Ocean King - Stock Movement Intelligence  (v4)

Improves product visibility from the Swords stock-movement export (any file name, CSV or Excel):
  * Overview            - sales, stock and a "visibility scorecard" (what we can and cannot see)
  * Fast movers         - what sells most, how much stock is left, where it lives
  * Slow movers         - what has stopped or barely sells, cash tied up, who bought it before
  * Product lookup      - everything about one product
  * Customers           - who buys what, and a customer lookup
  * AI recommendations  - priority actions, forecast & reorder suggestions, unusual activity,
                          and an optional Claude "AI advisor" (opt-in, summaries only)

Run locally - nothing is uploaded unless you switch the AI advisor on:
    pip install streamlit pandas openpyxl
    streamlit run ocean_king_movement_tool.py

Only the movement CSV is needed.  The 'Line no' column is never read or used.
"""

import io
import json
import os
import re
import urllib.error
import urllib.request

import numpy as np
import pandas as pd
import streamlit as st

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------
VALID_MOVE_TYPES = ["Sales inv", "Sales cr", "G'receipt", "Op bal", "Stock adj"]
TEMP_MAP = {1: "Frozen", 2: "Ambient", 3: "Chilled"}          # Swords Department code
TEMP_ORDER = ["Frozen", "Chilled", "Ambient", "Other"]
CLASS_ORDER = ["Fast", "Medium", "Slow", "Stalled", "Not moving", "Inactive"]
CLASS_ICON = {"Fast": "🚀", "Medium": "🙂", "Slow": "🐢", "Stalled": "⏸", "Not moving": "🧊", "Inactive": "⚪"}
SEV_ORDER = {"🔴 High": 0, "🟠 Medium": 1, "🟡 Check": 2, "🔵 Info": 3}
GBP = "£"
DEFAULT_LABELS = "1=Takeaway\n2=Restaurant\n3=Mini market\n4=Wholesaler\n5=Pub\n6=Staff\n7=Manufacturer\n0=Unassigned\n-1=Unassigned"
CLAUDE_URL = "https://api.anthropic.com/v1/messages"
DEFAULT_MODEL = "claude-sonnet-5-5"          # editable in the app (cheaper option: claude-haiku-4-5-20251001)


# ----------------------------------------------------------------------------
# Loading & cleaning
# ----------------------------------------------------------------------------
def normalise_bin(raw) -> dict:
    s = "" if pd.isna(raw) else re.sub(r"\s+", " ", str(raw).strip())
    m = re.match(r"^([A-Za-z])\s*(\d+)$", s)
    if not m:
        return {"bin_clean": "", "zone": "", "bin_status": "Blank" if s == "" else "Unrecognised"}
    zone, no = m.group(1).upper(), int(m.group(2))
    if no == 0:
        return {"bin_clean": f"{zone}-00", "zone": zone, "bin_status": "Placeholder"}
    return {"bin_clean": f"{zone}-{no:02d}", "zone": zone, "bin_status": "Valid"}


CORE_COLS = ["Move Date", "Move Type", "Invoice no", "Shop name", "Product", "Product Name", "Quantity", "Running Total",
             "Line Value", "Line Cost", "Department"]
# Columns that older / slightly different exports may not have. If missing, they are filled with a safe default
# and the tool says so, instead of refusing the file.
OPTIONAL_DEFAULTS = {"Unit": "", "Pack": "1", "Split Quantity": "0", "Weight": "", "Split Running Total": "0", "Unit Value": "",
                     "Unit Cost": "", "Bin Locn": "", "Customer Category": "", "Move Time": ""}
REQUIRED_COLS = CORE_COLS + list(OPTIONAL_DEFAULTS)


def read_table(raw: bytes, name: str = ""):
    """Reads the Swords movement export as CSV or Excel (.xlsx)."""
    if name.lower().endswith((".xlsx", ".xlsm", ".xls")):
        return pd.read_excel(io.BytesIO(raw), dtype=str).fillna(""), "excel"
    for enc in ("utf-8-sig", "cp1252", "latin1"):
        try:
            return pd.read_csv(io.BytesIO(raw), encoding=enc, dtype=str, keep_default_na=False), enc
        except UnicodeDecodeError:
            continue
    raise ValueError("Could not decode the file")


def parse_dates(col: pd.Series):
    """Works out whether dates are month/day/year, day/month/year or ISO (2026-09-01), whichever reads the most rows."""
    s = col.astype(str).str.strip().str.slice(0, 10)
    best, best_fmt = None, None
    for fmt in ("%m/%d/%Y", "%d/%m/%Y", "%Y-%m-%d"):
        d = pd.to_datetime(s, format=fmt, errors="coerce")
        valid = int((d.notna() & (d.dt.year > 2000)).sum())
        if best is None or valid > best[0]:
            best, best_fmt = (valid, d), fmt
    return best[1], {"%m/%d/%Y": "month/day/year", "%d/%m/%Y": "day/month/year", "%Y-%m-%d": "year-month-day"}[best_fmt]


def load_movements(raw: bytes, name: str = ""):
    """Returns (clean movements, issues). Broken rows (bad move type, no product, or a 1900 / invalid date) are removed."""
    df, enc = read_table(raw, name)
    canon = {c.lower(): c for c in REQUIRED_COLS + ["Line no"]}
    cleaned = [re.sub(r"\s+", " ", str(c).strip()) for c in df.columns]
    df.columns = [canon.get(c.lower(), c) for c in cleaned]          # ignores capital letters and extra spaces in headers
    missing = [c for c in CORE_COLS if c not in df.columns]
    if missing:
        raise ValueError("This does not look like a Swords stock-movement export (the file name does not matter, the column headings do). "
                         "Missing essential columns: " + ", ".join(missing) + ". Columns found: " + ", ".join(cleaned[:25]))
    miss_opt = [c for c in OPTIONAL_DEFAULTS if c not in df.columns]
    for c in miss_opt:
        df[c] = OPTIONAL_DEFAULTS[c]
    df = df.drop(columns=[c for c in df.columns if c.lower() == "line no"])      # never used
    for c in df.columns:
        df[c] = df[c].astype(str).str.strip()
    issues = {"encoding": enc, "raw_rows": len(df), "missing_optional": miss_opt}

    dt, issues["date_format"] = parse_dates(df["Move Date"])
    ok = df["Move Type"].isin(VALID_MOVE_TYPES) & df["Product"].ne("") & dt.notna() & (dt.dt.year > 2000)
    issues["n_removed"] = int((~ok).sum())
    df = df[ok].copy()
    df["Move Date"] = dt[ok]

    for c in ["Invoice no", "Product", "Pack", "Quantity", "Split Quantity", "Weight", "Running Total",
              "Split Running Total", "Unit Value", "Line Value", "Unit Cost", "Line Cost", "Department", "Customer Category"]:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["Product"] = df["Product"].astype(int)
    df["row"] = np.arange(len(df))
    df = df.sort_values(["Move Date", "row"], kind="mergesort").reset_index(drop=True)

    # Move Time: original Swords export is 'HH:MM' glued to a 4-digit sequence (e.g. 08:000617).
    # If the CSV was re-saved in Excel this column is destroyed (e.g. 1251:54:00).
    tm = df["Move Time"].str.extract(r"^(\d{2}):(\d{2})(\d{4})$")
    issues["time_ok"] = bool(tm[0].notna().mean() >= 0.8)
    df["Hour"] = pd.to_numeric(tm[0], errors="coerce") if issues["time_ok"] else np.nan

    shop = df["Shop name"].str.extract(r"^(\d+)\s+(.*)$")
    df["Shop ID"] = shop[0].fillna("")
    df["Shop"] = shop[1].fillna(df["Shop name"]).str.strip()
    df["Shop label"] = np.where(df["Shop ID"] != "", df["Shop ID"] + " - " + df["Shop"], df["Shop"])

    uniq = {x: normalise_bin(x) for x in df["Bin Locn"].unique()}
    b = pd.DataFrame([uniq[x] for x in df["Bin Locn"]], index=df.index)
    df = pd.concat([df, b], axis=1)
    df["Temp"] = df["Department"].map(TEMP_MAP).fillna("Other")
    df["Cust category"] = df["Customer Category"].fillna(-1).astype(int).astype(str)

    # Weight-based ("catch weight") lines: Unit = CWT. Swords counts these in kg, one line = one piece.
    df["is_cw"] = df["Unit"].eq("CWT")
    sale = df["Move Type"].eq("Sales inv")
    wt = (-df["Weight"]).where(df["Weight"].notna(), -df["Quantity"])
    df["qty_out"] = np.where(sale, np.where(df["is_cw"], wt, -df["Quantity"]), 0.0)
    df["splits_out"] = np.where(sale & ~df["is_cw"], -df["Split Quantity"], 0.0)
    df["sales_val"] = np.where(sale, -df["Line Value"], 0.0)
    df["sales_cost"] = np.where(sale, -df["Line Cost"], 0.0)
    df["credit_val"] = np.where(df["Move Type"].eq("Sales cr"), df["Line Value"], 0.0)
    df["has_split"] = sale & ~df["is_cw"] & df["Split Quantity"].ne(0)

    # Stock before a product's first weekly 'Op bal' is not initialised (it starts from 0), so it is not trusted.
    first_open = df[df["Move Type"] == "Op bal"].groupby("Product")["Move Date"].min()
    df["first_open"] = df["Product"].map(first_open)
    df["neg_after"] = sale & df["Running Total"].lt(0) & df["first_open"].notna() & (df["Move Date"] >= df["first_open"])

    start = df["Move Date"].min()
    df["wk"] = ((df["Move Date"] - start).dt.days // 7).astype(int)
    issues["start"], issues["end"] = start, df["Move Date"].max()
    issues["n_weeks"] = int(df["wk"].max()) + 1
    issues["trading_days"] = int(df.loc[sale, "Move Date"].nunique())
    issues["has_opbal"] = bool((df["Move Type"] == "Op bal").any())
    return df, issues


# ----------------------------------------------------------------------------
# Product table: velocity, stock, class, suggested action
# ----------------------------------------------------------------------------
def default_rules():
    return dict(a_cut=0.80, fast_freq=0.50, slow_freq=0.20, stalled_days=14, overstock_days=60, low_cover=5)


def build_products(mov, issues, rules):
    td = max(issues["trading_days"], 1)
    end = issues["end"]
    sales = mov[mov["Move Type"] == "Sales inv"]

    base = (mov.groupby("Product")
            .agg(Name=("Product Name", "last"), Unit=("Unit", "last"), Pack=("Pack", "last"), Temp=("Temp", "last"),
                 is_cw=("is_cw", "last")).reset_index())
    last = (mov.groupby("Product").tail(1).set_index("Product")
            [["Running Total", "Split Running Total", "Bin Locn", "bin_clean", "bin_status", "zone"]]
            .rename(columns={"Running Total": "Stock now", "Split Running Total": "Splits now", "Bin Locn": "Bin"}))
    s = (sales.groupby("Product")
         .agg(Lines=("Invoice no", "size"), Invoices=("Invoice no", "nunique"), Customers=("Shop ID", "nunique"),
              Sale_days=("Move Date", "nunique"), Units_out=("qty_out", "sum"), Splits_out=("splits_out", "sum"),
              Sales=("sales_val", "sum"), Cost=("sales_cost", "sum"), Last_sale=("Move Date", "max"),
              Neg_lines=("neg_after", "sum"), Split_lines=("has_split", "sum"))
         .reset_index())
    cr = mov[mov["Move Type"] == "Sales cr"].groupby("Product").agg(Credits=("credit_val", "sum"), Credit_lines=("Invoice no", "size"))
    rc = mov[mov["Move Type"] == "G'receipt"].groupby("Product").agg(Received=("Quantity", "sum"), Last_receipt=("Move Date", "max"))

    p = base.merge(s, on="Product", how="left").set_index("Product")
    p = p.join(last).join(cr).join(rc)
    for c in ["Lines", "Invoices", "Customers", "Sale_days", "Units_out", "Splits_out", "Sales", "Cost", "Neg_lines",
              "Split_lines", "Credits", "Credit_lines", "Received"]:
        p[c] = p[c].fillna(0)

    # who bought it most (top 3 shops by sales £)
    bs = (sales.groupby(["Product", "Shop label"])["sales_val"].sum().reset_index()
          .sort_values(["Product", "sales_val"], ascending=[True, False]))
    p["Top buyers"] = bs.groupby("Product").head(3).groupby("Product")["Shop label"].agg(lambda x: "; ".join(x))
    p["Top buyers"] = p["Top buyers"].fillna("")

    # cost per stock unit: per case normally, per kg for weight-based lines
    c1 = mov[mov["Move Type"].isin(["Sales inv", "G'receipt"]) & ~mov["is_cw"] & (mov["Unit Cost"] > 0) & (mov["Quantity"] != 0)]
    c1 = c1.groupby("Product")["Unit Cost"].last()
    cw = mov[mov["is_cw"] & (mov["Move Type"] == "Sales inv") & (mov["Weight"] < 0) & (mov["Line Cost"] < 0)]
    c2 = (cw["Line Cost"] / cw["Weight"]).groupby(cw["Product"]).last()
    p["Cost unit"] = c1.combine_first(c2)
    p["UoM"] = np.where(p["is_cw"], "kg", "cases")

    # weekly sales (units) and trend
    wk = (sales.pivot_table(index="Product", columns="wk", values="qty_out", aggfunc="sum")
          .reindex(columns=range(issues["n_weeks"])).fillna(0))
    p["Weekly"] = pd.Series(wk.values.tolist(), index=wk.index)
    rec = sales[sales["Move Date"] > end - pd.Timedelta(days=14)].groupby("Product")["qty_out"].sum()
    pri = sales[(sales["Move Date"] <= end - pd.Timedelta(days=14)) & (sales["Move Date"] > end - pd.Timedelta(days=28))].groupby("Product")["qty_out"].sum()
    p["Recent14"], p["Prior14"] = rec, pri
    p[["Recent14", "Prior14"]] = p[["Recent14", "Prior14"]].fillna(0)
    ratio = p["Recent14"] / p["Prior14"].replace(0, np.nan)
    p["Trend"] = np.select(
        [p["Lines"] < 4, (p["Prior14"] == 0) & (p["Recent14"] > 0), ratio >= 1.3, ratio <= 0.7],
        ["Too little data", "New / restarted", "Rising", "Falling"], default="Steady")

    # velocity & cover
    p["Per_day"] = p["Units_out"] / td
    p["Sale_freq"] = p["Sale_days"] / td
    p["Cover_days"] = np.where(p["Per_day"] > 0, p["Stock now"] / p["Per_day"].replace(0, np.nan), np.nan)
    p["Days_since_sale"] = (end - p["Last_sale"]).dt.days
    p["Stock value"] = p["Stock now"].clip(lower=0) * p["Cost unit"]
    p["Margin %"] = np.where(p["Sales"] > 0, (p["Sales"] - p["Cost"]) / p["Sales"].replace(0, np.nan) * 100, np.nan)

    # Pareto on sales value
    sold = p["Sales"] > 0
    rank = p.loc[sold, "Sales"].sort_values(ascending=False)
    cum_before = rank.cumsum().shift(fill_value=0) / rank.sum()
    p["Sales share %"] = p["Sales"] / rank.sum() * 100
    p["ABC"] = pd.Series(np.select([cum_before < rules["a_cut"], cum_before < 0.95], ["A", "B"], default="C"), index=rank.index)
    p["ABC"] = p["ABC"].fillna("-")
    p["Sales rank"] = rank.rank(ascending=False, method="first")
    p["Pick rank"] = p["Lines"].where(p["Lines"] > 0).rank(ascending=False, method="first")

    # movement class
    has_stock = p["Stock now"] > 0
    p["Class"] = np.select(
        [~sold & has_stock, ~sold,
         sold & (p["Days_since_sale"] >= rules["stalled_days"]) & has_stock,
         sold & ((p["ABC"] == "A") | (p["Sale_freq"] >= rules["fast_freq"])),
         sold & (p["Sale_freq"] < rules["slow_freq"])],
        ["Not moving", "Inactive", "Stalled", "Fast", "Slow"], default="Medium")

    # excess stock above the cover limit
    keep_units = rules["overstock_days"] * p["Per_day"]
    excess_units = np.where(has_stock, np.maximum(p["Stock now"] - keep_units, 0), 0)
    p["Excess value"] = pd.Series(excess_units, index=p.index) * p["Cost unit"]

    p["Suggested action"] = suggest_actions(p, rules)
    p = p.reset_index()
    p["Product label"] = p["Product"].astype(str) + " – " + p["Name"].astype(str)
    return p


def suggest_actions(p, rules):
    fast = p["Class"] == "Fast"
    conds = [
        fast & (p["Stock now"] <= 0),
        fast & (p["Cover_days"] <= rules["low_cover"]),
        fast & (p["bin_status"] != "Valid"),
        p["Class"] == "Stalled",
        p["Class"] == "Not moving",
        (p["Cover_days"] > rules["overstock_days"]) & p["Class"].isin(["Fast", "Medium", "Slow"]),
        (p["Class"] == "Slow") & (p["Customers"] <= 1),
        p["Class"] == "Slow",
        (p["Class"] == "Medium") & (p["bin_status"] != "Valid") & (p["Stock now"] > 0),
    ]
    texts = [
        "Fast mover out of stock / negative: replenish and check goods-in booking",
        "Fast mover running low: reorder now",
        "Fast mover with no real bin: give it a fixed front pick bin",
        "Stopped selling but still has stock: stop reordering, push to customers or return to supplier",
        "No sales in the period: stop reordering, promote or return to supplier",
        "Overstocked vs demand: cut order size / frequency",
        "Slow and bought by one shop only: make to order or consider delisting",
        "Slow mover: keep minimal stock, store in a back location",
        "No real bin: assign a location",
    ]
    return np.select(conds, texts, default="OK - no action needed")


# ----------------------------------------------------------------------------
# Specific analyses
# ----------------------------------------------------------------------------
def weekly_rebases(mov, p):
    """Weekly 'Op bal' rows reset stock. The gap between the opening balance and the running total just before it
    is stock that appeared / disappeared without a normal movement."""
    m = mov.copy()
    m["prev_rt"] = m.groupby("Product")["Running Total"].shift(1)
    first = m.loc[m["Move Type"] == "Op bal", "Move Date"].min()
    ob = m[(m["Move Type"] == "Op bal") & (m["Move Date"] > first)].dropna(subset=["prev_rt"]).copy()
    ob["Difference"] = ob["Running Total"] - ob["prev_rt"]
    ob = ob[ob["Difference"] != 0]
    ob = ob.merge(p[["Product", "Cost unit", "UoM"]], on="Product", how="left")
    ob["Value"] = ob["Difference"] * ob["Cost unit"]
    out = ob[["Move Date", "Product", "Product Name", "UoM", "prev_rt", "Running Total", "Difference", "Value"]].rename(
        columns={"prev_rt": "System said", "Running Total": "Opening balance", "Move Date": "Week"})
    return out.sort_values("Value", key=lambda s: s.abs().fillna(0), ascending=False)


def price_checks(sales):
    s = sales[(sales["sales_cost"] > 0)].copy()
    s = s[s["sales_val"] < s["sales_cost"]]
    s["Shortfall"] = s["sales_cost"] - s["sales_val"]
    s["Kind"] = np.where(s["sales_val"] < s["sales_cost"] / 3, "Cost looks wrong (cost > 3x price)", "Sold below cost")
    return s


def pair_table(sales, p, top_n=120):
    inv_n = sales.groupby("Product")["Invoice no"].nunique().sort_values(ascending=False)
    prods = inv_n.head(top_n).index.tolist()
    sub = sales[sales["Product"].isin(prods)][["Invoice no", "Product"]].drop_duplicates()
    X = pd.crosstab(sub["Invoice no"], sub["Product"]).clip(upper=1)
    M = X.values.T @ X.values
    n = len(X)
    cols = X.columns.tolist()
    rows = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            c = M[i, j]
            if c >= 25:
                lift = c * n / (M[i, i] * M[j, j])
                rows.append((cols[i], cols[j], int(c), int(M[i, i]), int(M[j, j]), lift))
    if not rows:
        return pd.DataFrame()
    t = pd.DataFrame(rows, columns=["A", "B", "Orders together", "A orders", "B orders", "Lift"])
    info = p.set_index("Product")[["Name", "bin_clean", "zone", "Class"]]
    t = t.join(info.add_prefix("A_"), on="A").join(info.add_prefix("B_"), on="B")
    t["Same zone?"] = np.where((t["A_zone"] == t["B_zone"]) & (t["A_zone"] != ""), "yes", "no")
    t["% of A's orders"] = (t["Orders together"] / t["A orders"] * 100).round(0)
    return t.sort_values("Orders together", ascending=False)


def customer_table(sales, issues):
    end = issues["end"]
    g = (sales.groupby(["Shop ID", "Shop label"])
         .agg(Sales=("sales_val", "sum"), Cost=("sales_cost", "sum"), Invoices=("Invoice no", "nunique"), Lines=("Invoice no", "size"),
              Products=("Product", "nunique"), First_order=("Move Date", "min"), Last_order=("Move Date", "max"),
              Category=("Cust category", lambda s: s.mode().iat[0]))
         .reset_index())
    rec = sales[sales["Move Date"] > end - pd.Timedelta(days=14)].groupby("Shop label")["sales_val"].sum()
    pri = sales[(sales["Move Date"] <= end - pd.Timedelta(days=14)) & (sales["Move Date"] > end - pd.Timedelta(days=28))].groupby("Shop label")["sales_val"].sum()
    g["Last 14 days"] = g["Shop label"].map(rec).fillna(0)
    g["Previous 14 days"] = g["Shop label"].map(pri).fillna(0)
    g["Change %"] = np.where(g["Previous 14 days"] > 0, (g["Last 14 days"] / g["Previous 14 days"].replace(0, np.nan) - 1) * 100, np.nan)
    g["Margin %"] = np.where(g["Sales"] > 0, (g["Sales"] - g["Cost"]) / g["Sales"].replace(0, np.nan) * 100, np.nan)
    g["Avg invoice"] = g["Sales"] / g["Invoices"].replace(0, np.nan)
    g["Lines/order"] = g["Lines"] / g["Invoices"].replace(0, np.nan)
    g["Share %"] = g["Sales"] / g["Sales"].sum() * 100
    return g.sort_values("Sales", ascending=False)


# ----------------------------------------------------------------------------
# Smart suggestions: forecast, reorder, unusual activity (statistical, runs offline)
# ----------------------------------------------------------------------------
def weekly_matrix(sales, end, n=4):
    """Units sold per product in the last n weeks counted back from the last date. Column 0 = most recent week."""
    s = sales[sales["Move Date"] > end - pd.Timedelta(days=7 * n)].copy()
    s["wb"] = ((end - s["Move Date"]).dt.days // 7).astype(int)
    return s.pivot_table(index="Product", columns="wb", values="qty_out", aggfunc="sum").reindex(columns=range(n)).fillna(0)


def forecast_table(sales, p, issues, lead_days=7, safety_days=3):
    """Next-week demand = weighted average of the last 4 weeks (recent weeks count more).
    Reorder = demand over (lead time + safety days) minus stock now, rounded up."""
    end = issues["end"]
    w = weekly_matrix(sales, end, 4)
    weights = np.array([4, 3, 2, 1]) / 10.0
    f = pd.DataFrame(index=w.index)
    f["Forecast next 7 days"] = (w.values * weights).sum(axis=1)
    f["Last week"] = w[0]
    f["Avg of previous 3 weeks"] = w[[1, 2, 3]].mean(axis=1)
    mean = w.mean(axis=1).replace(0, np.nan)
    f["Volatility"] = (w.std(axis=1) / mean).fillna(0)
    lines28 = sales[sales["Move Date"] > end - pd.Timedelta(days=28)].groupby("Product").size()
    f["Lines (28d)"] = lines28.reindex(f.index).fillna(0)
    f["Confidence"] = np.select([(f["Lines (28d)"] >= 12) & (f["Volatility"] < 0.6), f["Lines (28d)"] >= 6], ["High", "Medium"], default="Low")
    f = f.reset_index().merge(p[["Product", "Name", "Temp", "Class", "UoM", "Stock now", "Cost unit", "bin_status"]], on="Product", how="left")
    f["Per day"] = f["Forecast next 7 days"] / 7
    f["Days of stock"] = np.where(f["Per day"] > 0, f["Stock now"] / f["Per day"].replace(0, np.nan), np.nan)
    need = f["Per day"] * (lead_days + safety_days)
    f["Suggested order"] = np.ceil(np.maximum(need - f["Stock now"].clip(lower=0), 0))
    f["Order value"] = f["Suggested order"] * f["Cost unit"]
    f["When"] = np.select(
        [(f["Stock now"] < 0) & (f["Suggested order"] > 0),
         (f["Suggested order"] > 0) & (f["Days of stock"].fillna(0) <= lead_days),
         (f["Suggested order"] > 0)],
        ["Check stock record first (negative)", "Order now (will run out before delivery)", "Order soon"], default="No order needed")
    dchg = f["Last week"] / f["Avg of previous 3 weeks"].replace(0, np.nan)
    f["Demand change"] = np.select([(f["Avg of previous 3 weeks"] == 0) & (f["Last week"] > 0), dchg >= 1.5, dchg <= 0.5],
                                   ["Restarted", "Spike (up 50%+)", "Drop (down 50%+)"], default="Normal")
    return f.sort_values("Order value", ascending=False)


def backtest_forecast(sales, p, issues):
    """Hide the most recent week, forecast it from the 3 weeks before, compare with what really sold."""
    end = issues["end"]
    w = weekly_matrix(sales, end, 4)
    act = w[0]
    fc = (w[1] * 3 + w[2] * 2 + w[3] * 1) / 6
    naive = w[1]
    keep = p.set_index("Product").reindex(w.index)["Class"].isin(["Fast", "Medium"])
    a, f_, n_ = act[keep], fc[keep], naive[keep]
    if a.sum() <= 0:
        return None
    return dict(n=int(keep.sum()), wape=float((a - f_).abs().sum() / a.sum() * 100), naive_wape=float((a - n_).abs().sum() / a.sum() * 100),
                bias=float((f_.sum() - a.sum()) / a.sum() * 100),
                within30=float(((a - f_).abs() <= 0.3 * a.clip(lower=1)).mean() * 100))


def unusual_activity(sales):
    out = {}
    s = sales[sales["qty_out"] > 0].copy()
    g = s.groupby("Product")["qty_out"]
    s["usual"] = g.transform("median")
    s["n_lines"] = g.transform("size")
    big = s[(s["n_lines"] >= 8) & (s["qty_out"] >= 5) & (s["qty_out"] >= 5 * s["usual"].clip(lower=1))].copy()
    big["x usual"] = big["qty_out"] / big["usual"].clip(lower=1)
    out["big"] = (big[["Move Date", "Shop label", "Product", "Product Name", "qty_out", "usual", "x usual", "sales_val"]]
                  .rename(columns={"qty_out": "Units", "usual": "Usual per line", "sales_val": "Sales £", "Move Date": "Date"})
                  .sort_values("Sales £", ascending=False))
    c = sales[(~sales["is_cw"]) & (sales["Quantity"] < 0) & (sales["Unit Value"] > 0)].copy()
    c["price"] = c["Unit Value"]
    c["usual_price"] = c.groupby("Product")["price"].transform("median")
    c["n"] = c.groupby("Product")["price"].transform("size")
    low = c[(c["n"] >= 8) & (c["price"] < 0.75 * c["usual_price"])].copy()
    low["Gap £"] = (low["usual_price"] - low["price"]) * (-low["Quantity"])
    out["price"] = (low[["Move Date", "Shop label", "Cust category", "Product", "Product Name", "price", "usual_price", "Gap £"]]
                    .rename(columns={"price": "Price per case", "usual_price": "Usual price", "Move Date": "Date"})
                    .sort_values("Gap £", ascending=False))
    return out


# ----------------------------------------------------------------------------
# Action list (rule-based "so what")
# ----------------------------------------------------------------------------
def fmt_money(x):
    return f"{GBP}{x:,.0f}"


def build_actions(p, mov, issues, rules):
    A = []
    sales = mov[mov["Move Type"] == "Sales inv"]
    fast = p[p["Class"] == "Fast"]
    cols_base = ["Product", "Name", "Temp", "Class", "Sales", "Lines"]

    f = fast[fast["bin_status"] != "Valid"].sort_values("Sales", ascending=False)
    if len(f):
        A.append(dict(
            sev="🔴 High", title="Fast movers with no real bin location",
            headline=f"{len(f)} of {len(fast)} fast movers ({f['Sales'].sum() / max(fast['Sales'].sum(), 1) * 100:.0f}% of fast-mover sales) have no fixed bin",
            why="These are the products picked most often. Without a fixed place the picker has to search, and nobody can say where stock is.",
            do=["Give each one a fixed bin in the pick area closest to dispatch, fastest first (top to bottom of the list).",
                "Label the bin and the product, and put the bin number on the pick sheet.",
                "Stop using 'zone only' placeholders such as A 0 / C 0 for anything that sells weekly."],
            table=f[cols_base + ["Bin", "bin_status", "Stock now", "Suggested action"]], key="fast_no_bin"))

    f = fast[(fast["Stock now"] <= 0) | (fast["Cover_days"] <= rules["low_cover"])].sort_values("Sales", ascending=False)
    if len(f):
        A.append(dict(
            sev="🔴 High", title="Fast movers out of stock or about to run out",
            headline=f"{len(f)} fast movers have under {rules['low_cover']} trading days of stock (or show negative stock)",
            why="A missed line on a fast mover is a missed sale and an unhappy customer. Negative stock usually means goods arrived but were booked in late.",
            do=["Raise or chase purchase orders for these first.",
                "Where stock is negative, check whether a delivery is waiting to be booked in; correct the system.",
                "Set a minimum stock level for each fast mover so the system warns you before it runs out."],
            table=f[cols_base + ["UoM", "Stock now", "Per_day", "Cover_days", "Last_receipt", "Trend"]], key="fast_short"))

    n = p[p["Neg_lines"] >= 3].sort_values("Neg_lines", ascending=False)
    elig = sales[sales["Move Date"] >= sales["first_open"]]
    pct = sales["neg_after"].sum() / max(len(elig), 1) * 100
    if len(n):
        A.append(dict(
            sev="🟠 Medium", title="Selling products the system says are not in stock",
            headline=f"{pct:.1f}% of sales lines (after the first weekly opening balance) left stock below zero; {len(n)} products affected repeatedly",
            why="Either goods are being booked in after they are sold, or stock records are wrong. Both make pick lists, purchasing and cover figures unreliable.",
            do=["Book goods in the same day they arrive (agree a cut-off with the receiving team).",
                "Cycle-count the top products on this list this week and correct the system.",
                "Consider blocking invoicing of a line when system stock is zero, so someone checks first."],
            table=n[cols_base + ["Neg_lines", "UoM", "Stock now", "Bin"]], key="neg_stock"))

    dead = p[p["Class"].isin(["Stalled", "Not moving", "Slow"]) & (p["Stock now"] > 0)].copy()
    dead_val = dead["Stock value"].sum()
    if len(dead):
        top = dead.sort_values("Stock value", ascending=False)
        by = dead.groupby("Class")["Stock value"].sum()
        A.append(dict(
            sev="🔴 High" if dead_val > 50000 else "🟠 Medium",
            title="Cash tied up in slow, stalled and non-moving stock",
            headline=f"{fmt_money(dead_val)} of stock across {len(dead)} products (stalled {fmt_money(by.get('Stalled', 0))}, "
                     f"not moving {fmt_money(by.get('Not moving', 0))}, slow {fmt_money(by.get('Slow', 0))})",
            why="Slow stock takes up pick faces and cash, and frozen/chilled stock that sits too long is a write-off risk.",
            do=["Stop re-ordering anything in this list until stock is below your cover target.",
                "Offer it first to the customers who bought it before (see 'Top buyers').",
                "Bundle with fast movers, discount, or return to supplier where possible.",
                "Move it out of the front pick area to a back location so fast movers get the best faces."],
            table=top[cols_base + ["UoM", "Stock now", "Cost unit", "Stock value", "Days_since_sale", "Customers", "Top buyers", "Bin"]], key="dead_stock"))

    o = p[(p["Excess value"] > 0) & p["Class"].isin(["Fast", "Medium", "Slow", "Stalled"])].sort_values("Excess value", ascending=False)
    if len(o):
        A.append(dict(
            sev="🟠 Medium", title=f"Stock above {rules['overstock_days']} trading days of cover",
            headline=f"{len(o)} products hold more than {rules['overstock_days']} trading days of stock; {fmt_money(o['Excess value'].sum())} sits above that level",
            why="Buying more than you sell inside your cover target ties up cash and space. For imported frozen lines longer cover may be deliberate, so check against supplier lead times.",
            do=["Reduce order quantities or frequency for the top items.",
                "Check which lines are deliberate bulk buys (long lead time) and exclude them from the review.",
                "Use the 'Cover (days)' column to set a target cover per product group."],
            table=o[cols_base + ["UoM", "Stock now", "Per_day", "Cover_days", "Excess value", "Received"]], key="overstock"))

    if issues["has_opbal"]:
        rb = weekly_rebases(mov, p)
        if len(rb):
            tot = rb["Value"].abs().sum()
            A.append(dict(
                sev="🟠 Medium", title="Stock changes between weekly snapshots with no movement",
                headline=f"{len(rb)} weekly opening balances differed from the running stock (about {fmt_money(tot)} at cost, {rb['Product'].nunique()} products)",
                why="Each week Swords re-sets stock to an opening balance. When that differs from what the movements say, stock was found, lost, or booked wrongly in between.",
                do=["Cycle-count the products that appear repeatedly on this list.",
                    "Check receiving notes and pick errors for the biggest differences.",
                    "Treat 'JH Storage' and similar pass-through lines separately; they re-set to zero every week."],
                table=rb.head(300), key="rebase"))

    pc = price_checks(sales)
    if len(pc):
        wrong = pc[pc["Kind"].str.startswith("Cost")]
        real = pc[pc["Kind"] == "Sold below cost"]
        tab = (pc.groupby(["Product", "Product Name", "Kind"]).agg(Lines=("Shortfall", "size"), Shortfall=("Shortfall", "sum"))
               .reset_index().sort_values("Shortfall", ascending=False))
        A.append(dict(
            sev="🟠 Medium", title="Lines priced below cost",
            headline=f"{len(real)} lines sold below cost (about {fmt_money(real['Shortfall'].sum())}); {len(wrong)} more where the cost looks wrong (about {fmt_money(wrong['Shortfall'].sum())} of apparent loss)",
            why="Some are genuine under-pricing. Where cost is more than 3x the selling price, the cost per unit on Swords is probably set up wrongly (e.g. per CWT vs per kg), which also distorts margins and stock value.",
            do=["Fix cost prices for the 'cost looks wrong' products first (it removes false losses).",
                "For genuine below-cost lines, check price lists and whether the sale was authorised.",
                "Stop reps overriding prices below cost without sign-off."],
            table=tab.head(200), key="price"))

    r = p[(p["Credit_lines"] >= 3) & (p["Sales"] > 0)].copy()
    r["Returns %"] = r["Credits"] / r["Sales"] * 100
    r = r[r["Returns %"] >= 8].sort_values("Credits", ascending=False)
    if len(r):
        A.append(dict(
            sev="🟡 Check", title="Products with high credit notes (returns)",
            headline=f"{len(r)} products have credits of 8% or more of their sales ({fmt_money(r['Credits'].sum())} credited)",
            why="Repeated credits point to pick errors, quality problems, short-dated stock or wrong pack sizes.",
            do=["Look at the shops and dates behind the credits for the top items.",
                "Check pack size / unit on the invoice versus what is picked.",
                "Feed repeat issues back to purchasing and quality."],
            table=r[cols_base + ["Credits", "Credit_lines", "Returns %"]], key="returns"))

    pt = pair_table(sales, p)
    if len(pt):
        diff = pt[(pt["Same zone?"] == "no") & (pt["Lift"] >= 1.5)].head(60)
        if len(diff):
            A.append(dict(
                sev="🟡 Check", title="Products often ordered together but stored in different zones",
                headline=f"{len(diff)} product pairs appear on the same order often and sit in different zones",
                why="Pickers walk the warehouse once per order. Keeping frequent partners together shortens every pick walk.",
                do=["Place these pairs side by side, starting with the fastest.",
                    "Sort pick sheets by bin location so the route follows the shelves.",
                    "Use the pairs to build pick zones by order type (e.g. restaurant standard orders)."],
                table=diff[["A", "A_Name", "A_bin_clean", "B", "B_Name", "B_bin_clean", "Orders together", "% of A's orders", "Lift"]], key="pairs"))

    sp = p[(~p["is_cw"]) & (p["Lines"] >= 20) & (p["Split_lines"] / p["Lines"].replace(0, np.nan) >= 0.3)].copy()
    if len(sp):
        sp["% lines with loose units"] = sp["Split_lines"] / sp["Lines"] * 100
        sp = sp.sort_values("Lines", ascending=False)
        A.append(dict(
            sev="🟡 Check", title="Products often picked as loose units instead of full cases",
            headline=f"{len(sp)} products have 30% or more of their pick lines as loose units",
            why="Breaking cases for loose picks is slow and a common source of count errors.",
            do=["Keep an open 'loose pick' case in the pick face for these products.",
                "Check whether a smaller pack size or pre-split pack could be bought.",
                "Record splits as separate stock so counts stay accurate."],
            table=sp[cols_base + ["Pack", "Split_lines", "% lines with loose units"]], key="splits"))

    ct = customer_table(sales, issues)
    if len(ct):
        top1 = ct.iloc[0]
        top10 = ct["Share %"].head(10).sum()
        A.append(dict(
            sev="🔵 Info", title="Customer concentration",
            headline=f"{top1['Shop label']} is {top1['Share %']:.1f}% of sales; the top 10 customers are {top10:.0f}%",
            why="One large account can distort fast/slow rankings and creates risk if it changes buying.",
            do=["Make sure the top accounts always get priority stock.",
                "Confirm whether any large account is an internal / group account (use the sidebar switch to exclude it)."],
            table=ct.head(25)[["Shop label", "Sales", "Share %", "Invoices", "Products", "Margin %", "Change %"]], key="cust_conc"))
        dr = ct[(ct["Previous 14 days"] >= 500) & (ct["Change %"] <= -40)].sort_values("Previous 14 days", ascending=False)
        if len(dr):
            A.append(dict(
                sev="🔵 Info", title="Customers whose orders dropped",
                headline=f"{len(dr)} customers spent at least 40% less in the last 14 days than the 14 days before",
                why="Early warning of lost accounts or supply gaps (items they could not get).",
                do=["Call the biggest drops; ask what they could not get from you.",
                    "Cross-check their usual products against the out-of-stock list."],
                table=dr[["Shop label", "Category", "Previous 14 days", "Last 14 days", "Change %", "Last_order"]], key="cust_drop"))
    return sorted(A, key=lambda a: SEV_ORDER[a["sev"]])


# ----------------------------------------------------------------------------
# AI advisor (optional, opt-in): sends SUMMARY text only to the Claude API
# ----------------------------------------------------------------------------
AI_SYSTEM = (
    "You are an operations analyst for Ocean King, a UK food wholesaler (frozen, chilled and ambient products) with one warehouse. "
    "You will be given summary tables from the Swords stock-movement export. Use ONLY the numbers supplied; never invent figures, products or customers. "
    "If something cannot be answered from the supplied data, say so and say what extra data would be needed. "
    "Write for a warehouse / operations manager in plain UK English with short sentences. Always quote product codes. "
    "Be practical: say what to do, in what order, and why. Weight-based products are measured in kg, others in cases.")


def build_ai_context(p, actions, fc, ct, issues, rules, anonymise=True, include_customers=True, labels=None):
    """Plain-text summary of the analysis. This is exactly what is sent to the model (no raw rows, no 'Line no')."""
    labels = labels or {}
    sold = p[p["Sales"] > 0]
    tot = sold["Sales"].sum()
    L = []
    L.append(f"PERIOD: {issues['start']:%d %b %Y} to {issues['end']:%d %b %Y}, {issues['trading_days']} trading days.")
    L.append(f"TOTALS: sales {fmt_money(tot)}, margin {(tot - sold['Cost'].sum()) / max(tot, 1) * 100:.1f}%, products listed {len(p)}, products sold {len(sold)}, "
             f"stock value at cost {fmt_money(p['Stock value'].sum())}.")
    cs = (p.groupby("Class").agg(products=("Product", "size"), sales=("Sales", "sum"), stock_value=("Stock value", "sum")).reindex(CLASS_ORDER).fillna(0))
    L.append("CLASS SUMMARY (class: products, sales £, stock £):")
    for c, r in cs.iterrows():
        L.append(f"- {c}: {int(r['products'])}, {r['sales']:,.0f}, {r['stock_value']:,.0f}")
    L.append("RULES USED: Fast = top " + f"{rules['a_cut'] * 100:.0f}% of sales or sold on {rules['fast_freq'] * 100:.0f}%+ of trading days; "
             f"Slow = sold on under {rules['slow_freq'] * 100:.0f}% of days; Stalled = no sale for {rules['stalled_days']}+ days with stock left.")
    L.append("TOP 25 FAST MOVERS (code | name | sales £ | units/day | stock now | cover days | bin status | trend):")
    for _, r in p[p["Class"] == "Fast"].sort_values("Sales", ascending=False).head(25).iterrows():
        cov = f"{r['Cover_days']:.0f}" if pd.notna(r["Cover_days"]) else "n/a"
        L.append(f"- {r['Product']} | {str(r['Name'])[:40]} | {r['Sales']:,.0f} | {r['Per_day']:.1f} {r['UoM']}/day | {r['Stock now']:,.0f} | {cov} | {r['bin_status']} | {r['Trend']}")
    L.append("TOP 20 STOCK TO REVIEW - slow/stalled/not moving by stock £ (code | name | class | stock £ | days since last sale | customers):")
    for _, r in p[p["Class"].isin(["Slow", "Stalled", "Not moving"]) & (p["Stock now"] > 0)].sort_values("Stock value", ascending=False).head(20).iterrows():
        ds = f"{r['Days_since_sale']:.0f}" if pd.notna(r["Days_since_sale"]) else "no sale in period"
        L.append(f"- {r['Product']} | {str(r['Name'])[:40]} | {r['Class']} | {r['Stock value']:,.0f} | {ds} | {r['Customers']:.0f}")
    if fc is not None and len(fc):
        L.append("TOP 20 REORDER SUGGESTIONS from the statistical forecast (code | name | when | forecast next 7 days | stock now | suggested order | confidence):")
        for _, r in fc[fc["Suggested order"] > 0].sort_values("Order value", ascending=False).head(20).iterrows():
            L.append(f"- {r['Product']} | {str(r['Name'])[:40]} | {r['When']} | {r['Forecast next 7 days']:.0f} {r['UoM']} | {r['Stock now']:,.0f} | {r['Suggested order']:,.0f} | {r['Confidence']}")
    L.append("RULE-BASED FINDINGS (priority | finding):")
    for a in actions:
        if a["key"].startswith("cust_") and (anonymise or not include_customers):
            continue          # these headlines contain real customer names
        L.append(f"- {a['sev']} | {a['title']}: {re.sub(r'<[^>]+>', '', a['headline'])}")
    if include_customers and ct is not None and len(ct):
        L.append("CUSTOMER CATEGORIES (category: customers, sales £, share %):")
        cc = ct.assign(Cat=ct["Category"].map(lambda k: labels.get(k, k))).groupby("Cat").agg(c=("Shop ID", "size"), s=("Sales", "sum")).sort_values("s", ascending=False)
        for k, r in cc.iterrows():
            L.append(f"- {k}: {int(r['c'])}, {r['s']:,.0f}, {r['s'] / max(cc['s'].sum(), 1) * 100:.1f}%")
        L.append("TOP 15 CUSTOMERS (" + ("anonymised" if anonymise else "real names") + ") (name | category | sales £ | last 14 days vs previous 14 days %):")
        for i, (_, r) in enumerate(ct.head(15).iterrows(), 1):
            name = f"Customer {i}" if anonymise else r["Shop label"]
            chg = f"{r['Change %']:.0f}%" if pd.notna(r["Change %"]) else "n/a"
            L.append(f"- {name} | {labels.get(r['Category'], r['Category'])} | {r['Sales']:,.0f} | {chg}")
    return "\n".join(L)


def call_claude(api_key, model, system, prompt, max_tokens=1500, timeout=120):
    """Plain HTTPS call to the Claude Messages API. Returns (text, error)."""
    body = json.dumps({"model": model, "max_tokens": max_tokens, "system": system,
                       "messages": [{"role": "user", "content": prompt}]}).encode("utf-8")
    req = urllib.request.Request(CLAUDE_URL, data=body, method="POST",
                                 headers={"x-api-key": api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return None, f"The AI service replied with an error ({e.code}): {e.read().decode('utf-8', 'ignore')[:300]}"
    except Exception as e:
        return None, f"Could not reach the AI service: {e}"
    text = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
    return (text or None), (None if text else "The AI service returned no text.")


def local_briefing(p, actions, fc, issues, rules):
    """Plain-English summary written by rules (no AI, works offline)."""
    sold = p[p["Sales"] > 0]
    tot = sold["Sales"].sum()
    fast = p[p["Class"] == "Fast"]
    L = [f"**{len(fast)} fast movers make {fast['Sales'].sum() / max(tot, 1) * 100:.0f}% of sales** "
         f"({issues['start']:%d %b}–{issues['end']:%d %b}). Protect these first."]
    if fc is not None and len(fc):
        urgent = fc[fc["When"].str.startswith("Order now")].sort_values("Order value", ascending=False).head(5)
        if len(urgent):
            L.append("**Order now (will run out before a delivery could arrive):** " +
                     ", ".join(f"{r.Product} {str(r.Name)[:28]}" for r in urgent.itertuples()) + ".")
    dead = p[p["Class"].isin(["Stalled", "Not moving", "Slow"]) & (p["Stock now"] > 0)].sort_values("Stock value", ascending=False).head(5)
    if len(dead):
        L.append("**Stop buying / review first:** " +
                 ", ".join(f"{r['Product']} {str(r['Name'])[:28]} ({fmt_money(r['Stock value'])})" for _, r in dead.iterrows()) + ".")
    L.append("**Top things to fix:** " + "; ".join(a["title"].lower() for a in actions[:4]) + ".")
    return L


# ----------------------------------------------------------------------------
# Streamlit helpers
# ----------------------------------------------------------------------------
def show(df, **kw):
    try:
        st.dataframe(df, width="stretch", hide_index=True, **kw)
    except TypeError:
        st.dataframe(df, use_container_width=True, hide_index=True, **kw)


def pretty(df):
    """Round numbers and tidy names for display."""
    d = df.copy()
    ren = {"Per_day": "Units/day", "Cover_days": "Cover (days)", "Sale_days": "Days sold", "Units_out": "Units out",
           "Splits_out": "Loose units out", "Days_since_sale": "Days since sale", "Last_sale": "Last sale",
           "Last_receipt": "Last receipt", "Neg_lines": "Lines when stock < 0", "Cost unit": "Cost per unit",
           "Stock value": "Stock £", "Excess value": "Excess £", "Sales": "Sales £", "Cost": "Cost £", "Credits": "Credits £",
           "Credit_lines": "Credit lines", "Split_lines": "Lines with loose units", "bin_status": "Bin status",
           "bin_clean": "Bin (clean)", "Received": "Received in period", "Sales share %": "Share of sales %",
           "Sale_freq": "Days sold %", "Shortfall": "Shortfall £", "Value": "Value £", "Last_order": "Last order",
           "First_order": "First order", "Previous 14 days": "Previous 14 days £", "Last 14 days": "Last 14 days £",
           "Avg invoice": "Avg invoice £", "Stock_value": "Stock £", "Order value": "Order £"}
    d = d.rename(columns={k: v for k, v in ren.items() if k in d.columns})
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].round(1)
        if pd.api.types.is_datetime64_any_dtype(d[c]):
            d[c] = d[c].dt.strftime("%d %b")
    return d


def csv_bytes(df):
    return df.to_csv(index=False).encode("utf-8")


def hbar(df, row_px=24):
    """Horizontal bars showing EVERY label. df: index = labels, one numeric column."""
    labels = [str(i) for i in df.index]
    long = df.copy()
    long.index = labels
    long.index.name = "Label"
    long = long.reset_index().melt(id_vars="Label", var_name="Series", value_name="Value")
    spec = {"height": {"step": row_px}, "mark": {"type": "bar", "tooltip": True, "color": "#3b82f6"},
            "encoding": {"y": {"field": "Label", "type": "nominal", "sort": labels,
                               "axis": {"title": None, "labelLimit": 360, "labelOverlap": False}},
                         "x": {"field": "Value", "type": "quantitative", "axis": {"title": None}},
                         "tooltip": [{"field": "Label"}, {"field": "Value", "format": ",.0f"}]}}
    try:
        st.vega_lite_chart(long, spec, use_container_width=True)
    except Exception:
        st.vega_lite_chart(long, spec)


def metric_row(items):
    cols = st.columns(len(items))
    for c, (label, val, *h) in zip(cols, items):
        c.metric(label, val, help=h[0] if h else None)


def product_table(v, cols, key):
    """Product table with a weekly sparkline column where Streamlit supports it."""
    tab = v[cols].copy()
    if "Sale_freq" in tab:
        tab["Sale_freq"] = tab["Sale_freq"] * 100
    tab = pretty(tab)
    cfg = {}
    if "Weekly" in tab:
        try:
            cfg["Weekly"] = st.column_config.LineChartColumn("Weekly sales (units)")
        except Exception:
            tab = tab.drop(columns=["Weekly"])
    try:
        st.dataframe(tab, width="stretch", hide_index=True, column_config=cfg)
    except TypeError:
        st.dataframe(tab, use_container_width=True, hide_index=True, column_config=cfg)
    dl = v[[c for c in cols if c != "Weekly"]].copy()
    if "Sale_freq" in dl:
        dl["Sale_freq"] = dl["Sale_freq"] * 100
    st.download_button("⬇ Download this table", csv_bytes(pretty(dl)), f"{key}.csv", key=f"dl_{key}")


# ----------------------------------------------------------------------------
# Product lookup
# ----------------------------------------------------------------------------
def render_lookup(p, mov, issues, labels, rules):
    st.subheader("🔎 Look up any product")
    q = st.text_input("Product number or name", key="lk_q", placeholder="e.g. 102295  or  prawn").strip()
    if not q:
        st.info("Type a product number or part of a name.")
        top = p.sort_values("Sales", ascending=False).head(15)
        show(pretty(top[["Product", "Name", "Class", "Sales", "Suggested action"]]))
        return
    hit = p[p["Product"].astype(str).str.contains(q, regex=False) | p["Name"].str.contains(q, case=False, regex=False)]
    if hit.empty:
        st.warning(f"Nothing matches '{q}'.")
        return
    hit = hit.sort_values(["Sales"], ascending=False).head(100)
    pick = st.selectbox(f"{len(hit)} match(es)", hit["Product label"].tolist())
    r = hit[hit["Product label"] == pick].iloc[0]
    pid = int(r["Product"])
    uom = r["UoM"]

    st.markdown(f"### {CLASS_ICON[r['Class']]} {r['Name']}")
    st.caption(f"Product {pid} • {r['Temp']} • {r['Unit']} (pack {r['Pack']:.0f}) • stock counted in {uom}")
    st.success(f"**Verdict – {r['Class']}.** {r['Suggested action']}")

    c = st.columns(4)
    c[0].metric("Stock now", f"{r['Stock now']:,.0f} {uom}")
    c[1].metric("Cover", f"{r['Cover_days']:,.0f} days" if pd.notna(r["Cover_days"]) else "no sales")
    c[2].metric("Sold per trading day", f"{r['Per_day']:,.1f} {uom}")
    c[3].metric("Bin", r["Bin"] or "(none)", help=f"Status: {r['bin_status']}")
    c = st.columns(4)
    c[0].metric("Sales", f"{GBP}{r['Sales']:,.0f}", help=f"Rank {r['Sales rank']:.0f} of {int((p['Sales'] > 0).sum())}" if r["Sales"] > 0 else None)
    c[1].metric("Days sold", f"{r['Sale_days']:.0f} of {issues['trading_days']}")
    c[2].metric("Customers", f"{r['Customers']:.0f}")
    c[3].metric("Trend (14 vs prev 14 days)", r["Trend"])
    c = st.columns(4)
    c[0].metric("Last sale", f"{r['Last_sale']:%d %b}" if pd.notna(r["Last_sale"]) else "none in period")
    c[1].metric("Last goods receipt", f"{r['Last_receipt']:%d %b}" if pd.notna(r["Last_receipt"]) else "none in period")
    c[2].metric("Stock value", f"{GBP}{r['Stock value']:,.0f}" if pd.notna(r["Stock value"]) else "no cost on file")
    c[3].metric("Margin", f"{r['Margin %']:.0f}%" if pd.notna(r["Margin %"]) else "-")

    pm = mov[mov["Product"] == pid]
    ps = pm[pm["Move Type"] == "Sales inv"]
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**Sold per week ({uom})**")
        wk = ps.groupby("wk")["qty_out"].sum().reindex(range(issues["n_weeks"]), fill_value=0)
        wk.index = [f"{(issues['start'] + pd.Timedelta(days=7 * int(i))):%d %b}" for i in wk.index]
        st.bar_chart(wk)
    with c2:
        st.markdown(f"**Stock level after each movement ({uom})**")
        ts = pm.set_index("Move Date")["Running Total"]
        st.line_chart(ts.groupby(level=0).last())

    if len(ps):
        ps = ps.copy()
        ps["Cust category"] = ps["Cust category"].map(lambda k: labels.get(k, k))
        st.markdown("**Who buys it**")
        show(pretty(ps.groupby(["Shop label", "Cust category"]).agg(Lines=("Invoice no", "size"), Units_out=("qty_out", "sum"),
                                                                    Sales=("sales_val", "sum"), Last_order=("Move Date", "max"))
                    .reset_index().sort_values("Units_out", ascending=False).head(25)))
    st.markdown("**Movement log (newest first)**")
    cols = ["Move Date", "Move Type", "Invoice no", "Shop label", "Quantity", "Split Quantity", "Weight", "Running Total", "Bin Locn", "Line Value"]
    show(pm[cols].iloc[::-1].head(500))


# ----------------------------------------------------------------------------
# Customer lookup
# ----------------------------------------------------------------------------
def render_customer(ct, p, sales, issues, labels, rules):
    st.markdown("### 🔎 Look up a customer")
    q = st.text_input("Customer name or number", key="cu_q", placeholder="e.g. 52116  or  Dragon").strip()
    if not q:
        st.info("Type part of a customer name or number.")
        return
    hit = ct[ct["Shop label"].str.contains(q, case=False, regex=False)].head(100)
    if hit.empty:
        st.warning(f"No customer matches '{q}'.")
        return
    pick = st.selectbox(f"{len(hit)} match(es)", hit["Shop label"].tolist(), key="cu_pick")
    r = hit[hit["Shop label"] == pick].iloc[0]
    s = sales[sales["Shop label"] == pick]
    end = issues["end"]

    st.markdown(f"#### {pick}")
    st.caption(f"Category: {labels.get(r['Category'], r['Category'])} • first order {r['First_order']:%d %b} • last order {r['Last_order']:%d %b}")
    c = st.columns(4)
    c[0].metric("Sales", f"{GBP}{r['Sales']:,.0f}", help=f"{r['Share %']:.2f}% of all sales")
    c[1].metric("Orders", f"{r['Invoices']:.0f}")
    c[2].metric("Average order", f"{GBP}{r['Avg invoice']:,.0f}")
    c[3].metric("Lines per order", f"{r['Lines/order']:.1f}")
    c = st.columns(4)
    c[0].metric("Different products", f"{r['Products']:.0f}")
    c[1].metric("Margin", f"{r['Margin %']:.0f}%" if pd.notna(r["Margin %"]) else "-")
    c[2].metric("Last 14 days vs previous", f"{r['Change %']:+.0f}%" if pd.notna(r["Change %"]) else "n/a",
                help=f"{GBP}{r['Last 14 days']:,.0f} vs {GBP}{r['Previous 14 days']:,.0f}")
    gaps = s["Move Date"].drop_duplicates().sort_values().diff().dt.days.dropna()
    c[3].metric("Typical days between orders", f"{gaps.median():.0f}" if len(gaps) else "n/a")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Spend per week (£)**")
        wk = s.groupby("wk")["sales_val"].sum().reindex(range(issues["n_weeks"]), fill_value=0)
        wk.index = [f"{(issues['start'] + pd.Timedelta(days=7 * int(i))):%d %b}" for i in wk.index]
        st.bar_chart(wk)
    with c2:
        st.markdown("**Spend by temperature class (£)**")
        st.bar_chart(s.groupby("Temp")["sales_val"].sum())

    info = p.set_index("Product")[["Class", "Stock now", "Cover_days", "UoM", "Suggested action"]]
    prod = (s.groupby(["Product", "Product Name"])
            .agg(Orders=("Invoice no", "nunique"), Units=("qty_out", "sum"), Sales=("sales_val", "sum"), Last_bought=("Move Date", "max"))
            .reset_index().sort_values("Sales", ascending=False))
    prod = prod.join(info, on="Product")
    st.markdown("**What they buy (and whether we can supply it)**")
    show(pretty(prod.head(40).rename(columns={"Class": "Product class"})))

    risk = prod[(prod["Stock now"] <= 0) | (prod["Cover_days"] <= rules["low_cover"])].head(15)
    if len(risk):
        st.warning(f"**Supply risk:** {len(risk)} of this customer's regular products are out of stock or about to run out.")
        show(pretty(risk[["Product", "Product Name", "Sales", "Stock now", "Cover_days", "UoM"]]))
    stopped = prod[(prod["Orders"] >= 2) & (prod["Last_bought"] <= end - pd.Timedelta(days=14))].sort_values("Sales", ascending=False).head(15)
    if len(stopped):
        st.info(f"**Stopped buying recently:** {len(stopped)} products they bought at least twice but not in the last 14 days. Ask why.")
        show(pretty(stopped[["Product", "Product Name", "Orders", "Sales", "Last_bought", "Product class"] if "Product class" in stopped else
                          ["Product", "Product Name", "Orders", "Sales", "Last_bought", "Class"]]))


# ----------------------------------------------------------------------------
# App
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner="Reading and cleaning the file…")
def cached_load(raw: bytes, name: str = ""):
    return load_movements(raw, name)


def main():
    st.set_page_config(page_title="Ocean King – Stock Movement Intelligence", page_icon="🦐", layout="wide")
    st.title("🦐 Ocean King – Stock Movement Intelligence")
    st.caption("Better product visibility from stock levels and stock movement • files are processed in memory and not saved by the tool")

    with st.sidebar:
        st.header("1. File")
        f = st.file_uploader("Swords stock-movement export (CSV or Excel, any file name)", type=["csv", "CSV", "xlsx", "XLSX", "xlsm", "XLSM"])
    if not f:
        st.info("👈 Upload the Swords stock-movement export to begin. The file name does not matter: CSV or Excel, new or old data.")
        st.stop()
    try:
        mov, issues = cached_load(f.getvalue(), getattr(f, "name", ""))
    except Exception as e:
        st.error(f"Could not read the file: {e}")
        st.stop()

    if issues.get("missing_optional"):
        st.warning("This file has no column(s): " + ", ".join(issues["missing_optional"]) +
                   ". The tool filled them with defaults, so views that depend on them (bins, customer categories, time of day, catch-weight items) will be limited.")
    with st.sidebar:
        st.header("2. Scope")
        jh_prod = mov["Product Name"].str.contains("JH Storage", case=False, na=False)
        jh_cust = mov["Shop name"].str.contains("JH Trade", case=False, na=False)
        ex_prod = st.checkbox(f"Exclude 'JH Storage' products ({mov.loc[jh_prod, 'Product'].nunique()})", value=False,
                              help="These look like pass-through lines that reset to zero every week.")
        ex_cust = st.checkbox("Exclude customer 'JH Trade Ltd'", value=False,
                              help="Biggest account by volume; may be a group/related company rather than a normal customer.")
        temps = st.multiselect("Temperature class", TEMP_ORDER, default=TEMP_ORDER)
        st.header("3. Rules (change if you disagree)")
        rules = default_rules()
        rules["a_cut"] = st.slider("'A' products = top x% of sales £", 60, 95, 80) / 100
        rules["fast_freq"] = st.slider("Fast if sold on at least x% of trading days", 20, 90, 50) / 100
        rules["slow_freq"] = st.slider("Slow if sold on fewer than x% of trading days", 5, 40, 20) / 100
        rules["stalled_days"] = st.slider("Stalled if no sale for x days (and stock left)", 7, 28, 14)
        rules["overstock_days"] = st.slider("Overstock above x trading days of cover", 20, 120, 60)
        rules["low_cover"] = st.slider("Running low below x trading days of cover", 1, 15, 5)
        with st.expander("Customer category labels"):
            st.caption("Edit if the meanings change. One per line, code=name.")
            lab_txt = st.text_area("Labels", DEFAULT_LABELS)
    labels = {}
    for line in lab_txt.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            labels[k.strip()] = v.strip()

    keep = pd.Series(True, index=mov.index)
    if ex_prod:
        keep &= ~jh_prod
    if ex_cust:
        keep &= ~jh_cust
    keep &= mov["Temp"].isin(temps)
    m = mov[keep].copy()
    iss = dict(issues)
    sales_m = m[m["Move Type"] == "Sales inv"]
    iss["trading_days"] = int(sales_m["Move Date"].nunique())
    if sales_m.empty:
        st.warning("No sales left after the current scope settings.")
        st.stop()

    p = build_products(m, iss, rules)
    ct = customer_table(sales_m, iss)
    ct["Cat label"] = ct["Category"].map(lambda k: labels.get(k, k))

    t_over, t_fast, t_slow, t_prod, t_cust, t_ai, t_data = st.tabs(
        ["📊 Overview", "🚀 Fast movers", "🐢 Slow movers", "🔎 Product lookup", "🏪 Customers", "🤖 AI recommendations", "🧹 Data check"])

    sold = p[p["Sales"] > 0]
    fast = p[p["Class"] == "Fast"]
    slowish = p[p["Class"].isin(["Slow", "Stalled", "Not moving"]) & (p["Stock now"] > 0)]
    sales_total = sold["Sales"].sum()
    n80 = int((sold.sort_values("Sales", ascending=False)["Sales"].cumsum() / sales_total < rules["a_cut"]).sum() + 1)
    stocked = p[p["Stock now"] > 0]

    # ======================= OVERVIEW =======================
    with t_over:
        st.caption(f"Period: {iss['start']:%d %b} – {iss['end']:%d %b %Y} • {iss['trading_days']} trading days • {len(p):,} products in the file")
        metric_row([
            ("Sales", f"{GBP}{sales_total:,.0f}"),
            ("Margin", f"{(sales_total - sold['Cost'].sum()) / sales_total * 100:.1f}%"),
            ("Products sold", f"{len(sold):,} of {len(p):,}", "The rest had no sales in the period"),
            (f"Products making {rules['a_cut'] * 100:.0f}% of sales", f"{n80:,}"),
            ("Stock value (at cost)", f"{GBP}{p['Stock value'].sum():,.0f}"),
            ("In slow / stalled / non-moving", f"{GBP}{slowish['Stock value'].sum():,.0f}"),
        ])
        st.markdown("### Product visibility scorecard")
        st.caption("Can we see where each product is, how much there is, and how fast it goes? Every gap below is a place where staff have to guess.")
        neg_n = int((p["Stock now"] < 0).sum())
        nocost = int(((p["Stock now"] > 0) & p["Cost unit"].isna()).sum())
        metric_row([
            ("Stocked products with a real bin", f"{(stocked['bin_status'] == 'Valid').mean() * 100:.0f}%", "Real bin = zone letter + a number above 0"),
            ("Fast movers with a real bin", f"{(fast['bin_status'] == 'Valid').mean() * 100:.0f}%" if len(fast) else "n/a"),
            ("Pick lines from a real bin", f"{(sales_m['bin_status'] == 'Valid').mean() * 100:.0f}%"),
            ("Products with negative stock", f"{neg_n:,}", "System says less than zero in stock"),
            ("Sales when stock was below zero", f"{sales_m['neg_after'].sum() / max(len(sales_m[sales_m['Move Date'] >= sales_m['first_open']]), 1) * 100:.1f}%"),
            ("Stocked products with no cost", f"{nocost:,}", "Value of these stock lines is unknown"),
        ])
        st.markdown("### The big picture")
        bullets = [
            f"**{n80} products ({n80 / max(len(sold), 1) * 100:.0f}% of those sold) make {rules['a_cut'] * 100:.0f}% of sales.** Protect these, slot them near dispatch, never run out.",
            f"**{(len(p) - len(sold)):,} of {len(p):,} listed products had no sales** in the period; {int((p['Class'] == 'Not moving').sum())} of them still hold stock.",
            f"**Fast movers:** {len(fast)} products, {fast['Sales'].sum() / sales_total * 100:.0f}% of sales; {int((fast['bin_status'] != 'Valid').sum())} have no real bin.",
            f"**Slow / stalled / non-moving stock holds {GBP}{slowish['Stock value'].sum():,.0f}** ({slowish['Stock value'].sum() / max(p['Stock value'].sum(), 1) * 100:.0f}% of stock value).",
        ]
        for b in bullets:
            st.markdown(f"- {b}")

        cs = (p.groupby("Class").agg(Products=("Product", "size"), Sales=("Sales", "sum"), Lines=("Lines", "sum"), Stock_value=("Stock value", "sum"))
              .reindex(CLASS_ORDER).fillna(0))
        cs["% of sales"] = cs["Sales"] / sales_total * 100
        cs["% of pick lines"] = cs["Lines"] / max(p["Lines"].sum(), 1) * 100
        cs["% of stock £"] = cs["Stock_value"] / max(p["Stock value"].sum(), 1) * 100
        cs.index = [f"{CLASS_ICON[c]} {c}" for c in cs.index]
        c1, c2 = st.columns([3, 2])
        with c1:
            st.markdown("**How the business splits across movement classes**")
            show(pretty(cs.reset_index().rename(columns={"index": "Class"}).drop(columns=["Lines"])))
            st.caption("Fast = top sales group or sold most days • Slow = sold on few days • Stalled = stopped selling, stock left • "
                       "Not moving = stock but no sales • Inactive = no stock, no sales.")
        with c2:
            st.markdown("**Share of sales vs share of stock £**")
            st.bar_chart(cs[["% of sales", "% of stock £"]])
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Cumulative share of sales by product rank (Pareto)**")
            ranked = sold.sort_values("Sales", ascending=False)
            st.line_chart(pd.DataFrame({"Cumulative % of sales": (ranked["Sales"].cumsum() / sales_total * 100).values},
                                       index=np.arange(1, len(ranked) + 1)))
        with c2:
            st.markdown("**Stock £ by days of cover**")
            cov = stocked.copy()
            cov["Band"] = pd.cut(cov["Cover_days"].fillna(99999), [-1, 5, 20, 40, 60, 100, 9998, 999999],
                                 labels=["0-5 days", "5-20", "20-40", "40-60", "60-100", "100+", "No sales"])
            st.bar_chart(cov.groupby("Band", observed=False)["Stock value"].sum())
            st.caption("Left = sells out soon. Right = months of stock, or none selling.")

    # ======================= FAST MOVERS =======================
    with t_fast:
        st.subheader("🚀 Fast moving products")
        st.caption("In the top sales group, or sold on most trading days. These carry the business. Rules can be changed in the sidebar.")
        low = fast[(fast["Stock now"] <= 0) | (fast["Cover_days"] <= rules["low_cover"])]
        metric_row([("Fast movers", f"{len(fast):,}"), ("Share of sales", f"{fast['Sales'].sum() / sales_total * 100:.0f}%"),
                    ("Share of pick lines", f"{fast['Lines'].sum() / max(p['Lines'].sum(), 1) * 100:.0f}%"),
                    ("Stock held (at cost)", f"{GBP}{fast['Stock value'].sum():,.0f}"),
                    ("Running low / negative", f"{len(low):,}"), ("With no real bin", f"{int((fast['bin_status'] != 'Valid').sum()):,}")])
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Fast movers by days of stock left**")
            fb = pd.cut(fast["Cover_days"].fillna(99999), [-1, 5, 10, 20, 40, 9998, 999999],
                        labels=["0-5 days", "5-10", "10-20", "20-40", "40+", "no sales"])
            st.bar_chart(fb.value_counts().reindex(["0-5 days", "5-10", "10-20", "20-40", "40+"]).fillna(0))
        with c2:
            st.markdown("**Fast movers by bin status**")
            st.bar_chart(fast["bin_status"].value_counts())
        f1, f2, f3 = st.columns(3)
        ftxt = f1.text_input("Name / code contains", key="fast_txt")
        ftemp = f2.multiselect("Temperature", TEMP_ORDER, default=TEMP_ORDER, key="fast_temp")
        fonly = f3.checkbox("Only running low / no bin", value=False, key="fast_only")
        v = fast[fast["Temp"].isin(ftemp)]
        if ftxt:
            v = v[v["Product"].astype(str).str.contains(ftxt) | v["Name"].str.contains(ftxt, case=False, regex=False)]
        if fonly:
            v = v[(v["Stock now"] <= 0) | (v["Cover_days"] <= rules["low_cover"]) | (v["bin_status"] != "Valid")]
        v = v.sort_values("Sales", ascending=False)
        product_table(v, ["Pick rank", "Product", "Name", "Temp", "ABC", "Sales", "Sales share %", "UoM", "Units_out", "Per_day", "Customers",
                          "Trend", "Weekly", "Stock now", "Cover_days", "Bin", "bin_status", "Last_receipt", "Suggested action"], "fast_movers")
        st.caption(f"{len(v):,} products. 'Pick rank' 1 = picked most often, so should sit closest to dispatch. Units are cases (kg for weight-based lines).")
        pt = pair_table(sales_m, p)
        if len(pt):
            with st.expander("Fast movers that are often ordered together (store side by side)"):
                show(pretty(pt.head(40)[["A", "A_Name", "A_bin_clean", "B", "B_Name", "B_bin_clean", "Orders together", "% of A's orders", "Lift", "Same zone?"]]))

    # ======================= SLOW MOVERS =======================
    with t_slow:
        st.subheader("🐢 Slow moving products")
        st.caption("Slow = sold on few days. Stalled = was selling, nothing recently, stock left. Not moving = stock but no sales in the period.")
        cls3 = p[p["Class"].isin(["Slow", "Stalled", "Not moving"])]
        k = (cls3.groupby("Class").agg(Products=("Product", "size"), With_stock=("Stock now", lambda s: int((s > 0).sum())),
                                       Stock_value=("Stock value", "sum"), Sales=("Sales", "sum")).reindex(["Slow", "Stalled", "Not moving"]).fillna(0))
        metric_row([("Slow products", f"{int(k.loc['Slow', 'Products']):,}", f"{GBP}{k.loc['Slow', 'Stock_value']:,.0f} of stock"),
                    ("Stalled products", f"{int(k.loc['Stalled', 'Products']):,}", f"{GBP}{k.loc['Stalled', 'Stock_value']:,.0f} of stock"),
                    ("Not moving (with stock)", f"{int(k.loc['Not moving', 'Products']):,}", f"{GBP}{k.loc['Not moving', 'Stock_value']:,.0f} of stock"),
                    ("Cash tied up", f"{GBP}{slowish['Stock value'].sum():,.0f}"),
                    ("Over the cover limit", f"{GBP}{p['Excess value'].sum():,.0f}", "Stock above the overstock days in the sidebar")])
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Stock £ by class and temperature**")
            st.bar_chart(slowish.groupby(["Temp", "Class"])["Stock value"].sum().unstack(fill_value=0))
        with c2:
            st.markdown("**Biggest stock £ in slow items**")
            top10 = slowish.sort_values("Stock value", ascending=False).head(10)
            hbar(top10.assign(Label=top10["Product"].astype(str) + " " + top10["Name"].str[:28]).set_index("Label")[["Stock value"]], row_px=26)
        s1, s2, s3 = st.columns(3)
        scls = s1.multiselect("Class", ["Slow", "Stalled", "Not moving"], default=["Slow", "Stalled", "Not moving"], key="slow_cls")
        stxt = s2.text_input("Name / code contains", key="slow_txt")
        sstock = s3.checkbox("Only with stock", value=True, key="slow_stock")
        v = p[p["Class"].isin(scls)]
        if stxt:
            v = v[v["Product"].astype(str).str.contains(stxt) | v["Name"].str.contains(stxt, case=False, regex=False)]
        if sstock:
            v = v[v["Stock now"] > 0]
        v = v.sort_values("Stock value", ascending=False)
        product_table(v, ["Product", "Name", "Temp", "Class", "UoM", "Stock now", "Stock value", "Days_since_sale", "Last_sale", "Units_out",
                          "Customers", "Top buyers", "Bin", "Weekly", "Suggested action"], "slow_movers")
        st.caption("'Top buyers' = the shops that bought it most. Offer them the stock first.")

    # ======================= PRODUCT LOOKUP =======================
    with t_prod:
        render_lookup(p, m, iss, labels, rules)

    # ======================= CUSTOMERS =======================
    with t_cust:
        st.subheader("🏪 Customers")
        metric_row([("Customers", f"{len(ct):,}"), ("Orders (invoices)", f"{sales_m['Invoice no'].nunique():,}"),
                    ("Average order", f"{GBP}{ct['Sales'].sum() / max(sales_m['Invoice no'].nunique(), 1):,.0f}"),
                    ("Top 10 customers", f"{ct['Share %'].head(10).sum():.0f}% of sales")])
        cc = (ct.groupby("Cat label").agg(Customers=("Shop ID", "size"), Sales=("Sales", "sum"), Invoices=("Invoices", "sum"), Lines=("Lines", "sum"))
              .assign(Avg_order=lambda d: d["Sales"] / d["Invoices"], Lines_per_order=lambda d: d["Lines"] / d["Invoices"])
              .sort_values("Sales", ascending=False).reset_index().rename(columns={"Cat label": "Category"}))
        cc["Share %"] = cc["Sales"] / cc["Sales"].sum() * 100
        c1, c2 = st.columns([3, 2])
        with c1:
            st.markdown("**By customer category**")
            show(pretty(cc.drop(columns=["Lines"]).rename(columns={"Avg_order": "Avg order £", "Lines_per_order": "Lines per order"})))
        with c2:
            st.bar_chart(cc.set_index("Category")["Sales"])
        c1, c2 = st.columns(2)
        cat_pick = c1.multiselect("Category", sorted(ct["Cat label"].unique()), default=sorted(ct["Cat label"].unique()), key="cu_cat")
        topn = c2.slider("Show top customers", 10, 100, 25, key="cu_top")
        cv = ct[ct["Cat label"].isin(cat_pick)].head(topn)
        hbar(cv.set_index("Shop label")[["Sales"]], row_px=22)
        show(pretty(cv[["Shop label", "Cat label", "Sales", "Share %", "Invoices", "Lines/order", "Products", "Avg invoice", "Margin %",
                        "Previous 14 days", "Last 14 days", "Change %", "Last_order"]].rename(columns={"Cat label": "Category"})))
        st.download_button("⬇ Customer table", csv_bytes(pretty(ct.drop(columns=["Category"]))), "customers.csv", key="dl_cust")
        st.markdown("---")
        render_customer(ct, p, sales_m, iss, labels, rules)

    # ======================= AI RECOMMENDATIONS =======================
    with t_ai:
        st.subheader("🤖 Recommendations")
        actions = build_actions(p, m, iss, rules)
        fc = forecast_table(sales_m, p, iss, 7, 3)
        st.markdown("### Summary in plain English")
        st.caption("Written by rules from your numbers (no internet needed). The optional AI advisor below can write a fuller briefing.")
        for line in local_briefing(p, actions, fc, iss, rules):
            st.markdown(f"- {line}")

        st.markdown("### 1. Priority actions")
        st.caption("Most important first. Open each one for why it matters, what to do and the list of products.")
        for a in actions:
            with st.expander(f"{a['sev']}  {a['title']}  —  {a['headline']}", expanded=a["sev"].startswith("🔴")):
                st.markdown(f"**Why it matters:** {a['why']}")
                st.markdown("**What to do:**")
                for d in a["do"]:
                    st.markdown(f"- {d}")
                tab = pretty(a["table"])
                show(tab.head(100))
                st.download_button("⬇ Download this list", csv_bytes(tab), f"{a['key']}.csv", key=f"dl_{a['key']}")

        st.markdown("### 2. Forecast and reorder suggestions")
        st.caption("Statistical forecast (runs offline): next week's demand = last 4 weeks, with recent weeks counting more. "
                   "It does not know your open purchase orders or supplier minimums, so treat the quantity as a starting point.")
        c1, c2, c3 = st.columns(3)
        lead = c1.slider("Delivery lead time (days)", 1, 28, 7)
        safety = c2.slider("Safety stock (days)", 0, 14, 3)
        fcls = c3.multiselect("Classes", ["Fast", "Medium", "Slow"], default=["Fast", "Medium"], key="fc_cls")
        fc = forecast_table(sales_m, p, iss, lead, safety)
        bt = backtest_forecast(sales_m, p, iss)
        if bt:
            st.info(f"**How reliable is it?** Hiding the most recent week and forecasting it from the 3 weeks before, the forecast was off by "
                    f"**{bt['wape']:.0f}%** (volume-weighted) across {bt['n']} fast/medium products; simply repeating the previous week would have been off by "
                    f"{bt['naive_wape']:.0f}%. {bt['within30']:.0f}% of products were within 30%. Five weeks of data is short; accuracy will improve as history grows.")
        fv = fc[fc["Class"].isin(fcls)]
        need = fv[fv["Suggested order"] > 0]
        metric_row([("Order now", f"{int(need['When'].str.startswith('Order now').sum()):,}", "Will run out before a delivery could arrive"),
                    ("Order soon", f"{int((need['When'] == 'Order soon').sum()):,}"),
                    ("Check stock record first", f"{int(need['When'].str.startswith('Check stock').sum()):,}", "Stock is negative: probably goods not booked in"),
                    ("Order value (at cost)", f"{GBP}{need['Order value'].sum():,.0f}")])
        wsel = st.radio("Show", ["Order now", "Order soon", "Check stock record first", "No order needed", "Demand spikes / drops"], horizontal=True)
        if wsel == "Order now":
            tv = fv[fv["When"].str.startswith("Order now")]
        elif wsel == "Order soon":
            tv = fv[fv["When"] == "Order soon"]
        elif wsel == "Check stock record first":
            tv = fv[fv["When"].str.startswith("Check stock")]
        elif wsel == "No order needed":
            tv = fv[fv["When"] == "No order needed"]
        else:
            tv = fv[fv["Demand change"].isin(["Spike (up 50%+)", "Drop (down 50%+)", "Restarted"])]
        fcols = ["Product", "Name", "Class", "UoM", "Stock now", "Forecast next 7 days", "Days of stock", "Suggested order", "Order value",
                 "When", "Confidence", "Last week", "Avg of previous 3 weeks", "Demand change"]
        show(pretty(tv[fcols].sort_values("Order value", ascending=False)).head(300))
        st.download_button("⬇ Reorder suggestions", csv_bytes(pretty(need[fcols])), "reorder_suggestions.csv", key="dl_reorder")

        st.markdown("### 3. Unusual activity")
        ua = unusual_activity(sales_m)
        with st.expander(f"Unusually large orders ({len(ua['big'])} lines, 5x the product's usual line)"):
            st.caption("May be a one-off bulk buy, a data-entry slip (extra zero), or a new big customer. Worth a quick check.")
            show(pretty(ua["big"].head(100)))
        with st.expander(f"Sold well below the usual price ({len(ua['price'])} lines, 25%+ under the product's median case price)"):
            st.caption(f"Total gap about {fmt_money(ua['price']['Gap £'].sum())}. Some of this is legitimate (wholesaler pricing, promotions). "
                       "Look at repeat shops and categories first.")
            show(pretty(ua["price"].head(100)))

        st.markdown("### 4. AI advisor (optional – uses Claude)")
        st.warning("**Privacy:** this is OFF unless you press a button. It sends only the summary text shown below (totals, product codes and names, "
                   "stock figures, action headlines). No raw invoice rows are sent. Customer names are replaced with 'Customer 1, 2…' unless you choose otherwise. "
                   "Check Ocean King's data policy before using it.")
        c1, c2 = st.columns(2)
        key = c1.text_input("Anthropic API key", value=os.environ.get("ANTHROPIC_API_KEY", ""), type="password", key="ai_key",
                            help="Kept in this browser session only. You can also set the ANTHROPIC_API_KEY environment variable.")
        model = c2.text_input("Model", value=DEFAULT_MODEL, key="ai_model")
        o1, o2 = st.columns(2)
        incl = o1.checkbox("Include customer summary", value=True, key="ai_incl")
        anon = o2.checkbox("Hide customer names", value=True, key="ai_anon")
        ctx = build_ai_context(p, actions, fc, ct, iss, rules, anonymise=anon, include_customers=incl, labels=labels)
        with st.expander(f"Preview exactly what would be sent ({len(ctx):,} characters)"):
            st.code(ctx)
        if not key:
            st.info("Paste an API key to enable the advisor. Everything above works without it.")
        else:
            if st.button("Write this week's briefing", key="ai_brief"):
                with st.spinner("Asking Claude…"):
                    txt, err = call_claude(key, model, AI_SYSTEM,
                                           "Using only the data below, write a briefing for the warehouse manager with these parts:\n"
                                           "1. What is happening (3-4 bullets).\n2. The 5 most important actions this week, in order, with product codes and numbers.\n"
                                           "3. Risks if nothing is done.\n4. Three questions to ask the team or the Swords supplier to check the data.\n\nDATA:\n" + ctx)
                st.error(err) if err else st.markdown(txt)
            qn = st.text_input("Or ask a question about this summary", key="ai_q",
                               placeholder="e.g. Which frozen products should I stop reordering first, and why?")
            if qn and st.button("Ask", key="ai_ask"):
                with st.spinner("Asking Claude…"):
                    txt, err = call_claude(key, model, AI_SYSTEM, f"QUESTION: {qn}\n\nAnswer using only the data below.\n\nDATA:\n{ctx}", 1200)
                st.error(err) if err else st.markdown(txt)

    # ======================= DATA CHECK =======================
    with t_data:
        st.subheader("What was read, and how it was interpreted")
        metric_row([("Rows in file", f"{iss['raw_rows']:,}"), ("Rows used", f"{len(mov):,}"),
                    ("Broken rows removed", f"{iss['n_removed']:,}", "Bad move type, no product, or an invalid / 1900 date"),
                    ("Period", f"{iss['start']:%d %b} – {iss['end']:%d %b}")])
        if iss["n_removed"]:
            st.info(f"{iss['n_removed']} broken rows (invalid move type, no product, or a 1900 date) were removed and are not used anywhere.")
        if not iss["time_ok"]:
            st.warning("**Move Time is unusable in this file.** It looks like the CSV was re-saved in Excel, which turns the time "
                       "into values like 1251:54:00. Export straight from Swords if you need time of day. Nothing else here depends on it.")
        st.markdown("""
**How the numbers are worked out**
- **Sales £** = sum of sales invoice line values. **Margin** = sales minus the line cost Swords holds.
- **Units out**: cases normally. Weight-based lines (unit **CWT**) are counted in **kg**, because Swords keeps their stock in kg and each invoice line is one piece.
- **Trading days** = days with at least one sales line. **Days sold** = trading days the product had a sale. **Cover** = stock now ÷ units sold per trading day.
- **Stock now** = the running total after the product's last movement. Weekly **Op bal** rows are the weekly opening balances; stock before a
  product's first one is not initialised, so negative-stock counts only use lines after it.
- **Stock £** = stock now × latest unit cost (per case, or per kg). Products with no cost on file count as £0.
- Frozen / Chilled / Ambient comes from the Swords Department code (1 / 3 / 2). The 'Line no' column is not read.
""")
        dcols = ["Move Date", "Move Type", "Invoice no", "Shop label", "Cust category", "Product", "Product Name", "Temp", "Unit", "Pack",
                 "Quantity", "Split Quantity", "Weight", "Running Total", "Bin Locn", "bin_status", "Line Value", "Line Cost"]
        st.markdown("**Cleaned data (first 3,000 rows)**")
        show(m[dcols].head(3000))
        st.download_button("⬇ Cleaned movements", csv_bytes(m[dcols]), "clean_movements.csv", key="dl_clean")


if __name__ == "__main__":
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        _in_st = get_script_run_ctx() is not None
    except Exception:
        _in_st = False
    if _in_st:
        main()
    else:
        print("This file is a Streamlit app. Start it with:\n"
              "    streamlit run ocean_king_movement_tool.py\n"
              "(or use the launcher cell in your notebook). It cannot display inside a notebook cell.")
