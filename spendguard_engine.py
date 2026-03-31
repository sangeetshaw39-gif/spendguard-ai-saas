import pandas as pd
from ai_layer import categorize_transactions, detect_smart_anomalies, generate_ai_insights

# -------------------------------
# COLUMN MAPPING LOGIC
# -------------------------------

COLUMN_PATTERNS = {
    "date": [
        "date", "transaction_date", "txn_date", "payment_date", "purchase_date", 
        "entry_date", "value_date", "posting_date", "recorded_date", "day", "time", 
        "timestamp", "datetime", "txn_time", "transaction_time", "created_at", 
        "updated_at", "event_date", "booking_date"
    ],
    "amount": [
        "amount", "amt", "amount_paid", "amount paid", "payment_amount", 
        "payment amount", "value", "transaction_value", "txn_value", "txn amount", 
        "paid", "spent", "spending", "expense", "cost", "price", "total", 
        "total_amount", "debit", "debit_amount", "withdrawal", "charge", 
        "charges", "bill", "billing_amount", "net_amount", "gross_amount", 
        "final_amount", "paid_amount", "purchase_amount", "outflow", "money_out", 
        "expense_value", "transaction_amount"
    ],
    "vendor": [
        "vendor", "merchant", "merchant_name", "vendor_name", "payee", "party", 
        "company", "company_name", "seller", "supplier", "service_provider", 
        "business", "store", "shop", "brand", "counterparty", "receiver", 
        "beneficiary", "paid_to", "recipient", "biller", "organization", "entity"
    ],
    "category": [
        "category", "expense_type", "type", "group", "category_group", "segment", 
        "classification", "expense_category", "transaction_type", "spend_type", 
        "tag", "label", "bucket", "division", "class", "cost_type", "usage_type"
    ],
    "description": [
        "description", "desc", "details", "remarks", "note", "notes", "narration", 
        "transaction_details", "payment_details", "info", "comment", "comments", 
        "memo", "reference", "purpose", "explanation", "remarks_text", "txn_desc"
    ],
    "id": [
        "transaction_id", "txn_id", "reference_id", "ref_no", "receipt_no", 
        "invoice_no", "order_id", "payment_id", "id"
    ]
}


def normalize_column(col):
    return col.strip().lower().replace("_", " ").replace("-", " ")


def map_columns(df):
    column_map = {}
    used_cols = set()

    # Step 1: Exact matches first (highest confidence)
    for std_col, patterns in COLUMN_PATTERNS.items():
        for col in df.columns:
            if col in used_cols: continue
            col_clean = normalize_column(col)
            # Normalize patterns for exact comparison
            p_clean = [p.replace("_", " ").replace("-", " ").lower() for p in patterns]
            
            if col_clean in p_clean:
                column_map[std_col] = col
                used_cols.add(col)
                break

    # Step 2: Substring matches (fallback)
    for std_col, patterns in COLUMN_PATTERNS.items():
        if std_col in column_map: continue
        for col in df.columns:
            if col in used_cols: continue
            col_clean = normalize_column(col)
            p_clean = [p.replace("_", " ").replace("-", " ").lower() for p in patterns]
            
            if any(p in col_clean for p in p_clean):
                column_map[std_col] = col
                used_cols.add(col)
                break

    return column_map


# -------------------------------
# DATA CLEANING
# -------------------------------

def clean_data(df, column_map):
    df = df.copy()

    # Rename to standard columns
    rename_dict = {v: k for k, v in column_map.items()}
    df = df.rename(columns=rename_dict)

    warnings = []

    # Ensure required columns exist - Only Amount is strictly required now
    required_cols = ["amount"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"We couldn't detect a required column for '{col}'. Please ensure your file has at least one numerical column for transaction values.")

    # Pipeline Safety: Ensure other columns exist BEFORE processing
    if "vendor" not in df.columns:
        df["vendor"] = "Unknown Vendor"
        warnings.append("Vendor column missing → default applied")
    if "category" not in df.columns:
        df["category"] = "Uncategorized"
        warnings.append("Category column missing → default applied")
    if "date" not in df.columns:
        df["date"] = "Unknown Date"
        warnings.append("Date column missing → default applied")
    if "description" not in df.columns:
        df["description"] = "No description provided"
        warnings.append("Description column missing → default applied")

    detected_currency = "₹"
    currencies = ["$", "€", "£", "₹", "¥"]
    
    # Fast scan for currency symbol in first 100 rows
    for val in df["amount"].dropna().astype(str).head(100):
        for sym in currencies:
            if sym in val:
                detected_currency = sym
                break
        if detected_currency != "₹":
            break

    # Date parsing
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime('%Y-%m-%d')
        df["date"] = df["date"].fillna("Unknown Date")

    # Clean non-numeric characters before cast
    df["amount"] = df["amount"].astype(str).str.replace(r'[^\d.-]', '', regex=True)
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")

    # Fill missing strings securely to prevent CSV blank cells
    df["category"] = df.get("category", "Uncategorized")
    df["category"] = df["category"].fillna("Uncategorized")
    df["category"] = df["category"].replace(r'^\s*$', 'Uncategorized', regex=True)

    df["description"] = df.get("description", "No description provided")
    df["description"] = df["description"].fillna("No description provided")
    df["description"] = df["description"].replace(r'^\s*$', 'No description provided', regex=True)
    
    df["vendor"] = df["vendor"].fillna("Unknown Vendor")
    df["vendor"] = df["vendor"].replace(r'^\s*$', 'Unknown Vendor', regex=True)

    # Drop strictly invalid numeric rows
    df = df.dropna(subset=["amount"])
    
    if len(df) == 0:
        raise ValueError("We couldn't extract any valid amount figures from your file. Please ensure the transactions contain proper numerical values.")

    # Remove duplicates
    df = df.drop_duplicates()

    return df, detected_currency, warnings


# -------------------------------
# ANOMALY DETECTION
# -------------------------------

def detect_anomalies(df):
    mean = df["amount"].mean()
    std = df["amount"].std()

    threshold = mean + 2 * std

    df["anomaly"] = df["amount"] > threshold

    anomalies = df[df["anomaly"] == True]

    return df, anomalies


# -------------------------------
# BASIC INSIGHTS
# -------------------------------

def generate_insights(df, currency="₹"):
    total_spend = float(df["amount"].sum())
    top_categories = df.groupby("category")["amount"].sum().sort_values(ascending=False).head(5).to_dict()
    top_vendors = df.groupby("vendor")["amount"].sum().sort_values(ascending=False).head(5).to_dict()

    return {
        "currency": currency,
        "total_spend": total_spend,
        "top_categories": top_categories,
        "top_vendors": top_vendors
    }


# -------------------------------
# MAIN PIPELINE FUNCTION
# -------------------------------

def run_pipeline(file_path, memory=None):
    # ... (existing file loading)
    if file_path.endswith(".csv"):
        df = pd.read_csv(file_path)
    elif file_path.endswith(".xlsx"):
        df = pd.read_excel(file_path, engine="openpyxl")
    else:
        raise ValueError("We only support CSV and Excel (.xlsx) files. Please check your format and try again.")
        
    if len(df) == 0:
        raise ValueError("The uploaded file appears to be completely empty. No financial data was detected.")

    print(f"📊 Total rows loaded: {len(df)}")

    # -------------------------------
    # COLUMN MAPPING
    # -------------------------------
    # ... (skipping some logic)
    column_map = map_columns(df)
    df_clean, detected_currency, warnings = clean_data(df, column_map)
    basic_insights = generate_insights(df_clean, detected_currency)
    df_clean["ai_category"] = df_clean["category"]
    df_clean, anomalies = detect_smart_anomalies(df_clean)
    sample_df = df_clean.head(5).copy()

    # -------------------------------
    # AI INSIGHTS (ON SUMMARY ONLY)
    # -------------------------------
    try:
        ai_insights = generate_ai_insights(basic_insights, memory)
        if not ai_insights or "unavailable" in ai_insights.lower():
            raise Exception("AI failed")
    except:
        ai_insights = "<div class='alert-box alert-warning' style='margin-bottom:16px;'><span class='material-symbols-rounded'>cloud_off</span> <span>AI connection temporarily unavailable. Displaying Smart Heuristic Insights:</span></div>" + generate_fallback_insights(basic_insights)

    return {
        "clean_data": sample_df,   # sample shown
        "anomalies": anomalies,    # sample anomalies
        "insights": basic_insights, # FULL DATA insights ✅
        "ai_insights": ai_insights,
        "column_map": column_map,
        "clean_data_csv": df_clean.to_csv(index=False),
        "warnings": warnings
    }

# -------------------------------
# CLI TEST
# -------------------------------

if __name__ == "__main__":
    result = run_pipeline("messy_expense_data.csv")

    print("\n📊 Column Mapping:", result["column_map"])
    print("\n💰 Insights:", result["insights"])
    print("\n🤖 AI Insights:\n", result["ai_insights"])
    print("\n🚨 Anomalies Count:", len(result["anomalies"]))

def generate_fallback_insights(insights):
    total = insights.get("total_spend", 0)
    
    top_cat = ("Unknown", 0)
    if insights.get("top_categories"):
        top_cat = list(insights["top_categories"].items())[0]
        
    top_vendor = ("Unknown", 0)
    if insights.get("top_vendors"):
        top_vendor = list(insights["top_vendors"].items())[0]

    category_percent = (top_cat[1] / total * 100) if total > 0 else 0
    vendor_percent = (top_vendor[1] / total * 100) if total > 0 else 0
    
    currency = insights.get("currency", "₹")
    total_fmt = f"{currency}{total:,.0f}" if total < 1000000 else f"{currency}{total/1000000:.1f}M"

    return f"""
### Cost Reduction Opportunities
- **Problem:** Intense structural spending in the `{top_cat[0]}` category, natively capturing **{category_percent:.1f}%** of your entire operational outlay.
- **Impact:** Over-allocation starves crucial growth initiatives of liquidity and severely restricts available cash reserves limit.
- **Recommendation:** Enact an immediate freeze on non-essential `{top_cat[0]}` purchases and systematically audit active corporate subscriptions for redundancies.

### Risk Alerts
- **Problem:** Extreme dependency on a single centralized third-party vendor (**{top_vendor[0]}**), currently consuming **{vendor_percent:.1f}%** of all expenditures.
- **Impact:** Creates a catastrophic single point of failure and eliminates any leverage during contract renewals or unexpected price hikes.
- **Recommendation:** Immediately solicit competitive, blind-bid proposals from at least two alternative regional providers by the end of this quarter.

### Strategic Actions
- **Problem:** Aggregate spend velocity and vendor routing is currently decentralized, operating without a top-down visibility net.
- **Impact:** Exponentially aggregates the probability of phantom-billing, ghost duplicate charges, or silent subscription creep.
- **Recommendation:** Enforce a hard Purchase Order (PO) pre-approval workflow for any singular outbound expense exceeding {currency}5,000.
"""