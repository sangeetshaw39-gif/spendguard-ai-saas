import os
import json
import time
from google import genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize client securely from environment
api_key = os.environ.get("GEMINI_API_KEY")
# If no key is set, we set client to None to trigger offline fallbacks safely
client = genai.Client(api_key=api_key) if api_key else None


# -------------------------------
# AI CATEGORIZATION (BATCH)
# -------------------------------

def categorize_transactions(df):
    if not client:
        print("⚠️ GEMINI_API_KEY missing. Operating in offline mode.")
        df["ai_category"] = ["Uncategorized"] * len(df)
        return df

    # Prepare data
    data = df[["vendor", "description"]].fillna("").to_dict(orient="records")

    prompt = f"""
    Categorize each expense into one of:
    Food, Travel, SaaS, Office, Entertainment, Utilities, Other

    Return ONLY a valid JSON array.
    No explanation, no text.

    Example:
    ["Food", "Travel", "SaaS"]

    Data:
    {json.dumps(data, indent=2)}
    """

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )

        raw_output = response.text.strip()

        # Clean markdown formatting if present
        raw_output = raw_output.replace("```json", "").replace("```", "").strip()

        try:
            categories = json.loads(raw_output)
        except:
            print("⚠️ RAW AI OUTPUT (PARSE FAILED):", raw_output)
            categories = ["Other"] * len(df)

    except Exception as e:
        print("❌ AI CATEGORY ERROR:", e)
        categories = ["Other"] * len(df)

    # Safety check
    if len(categories) != len(df):
        print("⚠️ Category length mismatch. Fixing...")
        categories = ["Other"] * len(df)

    df["ai_category"] = categories

    return df


# -------------------------------
# SMART ANOMALY DETECTION
# -------------------------------

def detect_smart_anomalies(df):
    df["smart_anomaly"] = False

    for category in df["ai_category"].unique():
        subset = df[df["ai_category"] == category]

        mean = subset["amount"].mean()
        std = subset["amount"].std()

        threshold = mean + 2 * std

        mask = (df["ai_category"] == category) & (df["amount"] > threshold)

        df.loc[mask, "smart_anomaly"] = True

    anomalies = df[df["smart_anomaly"] == True]

    return df, anomalies


# -------------------------------
# AI INSIGHTS
# -------------------------------

def generate_ai_insights(insights_dict, memory=None):
    if not client:
        print("⚠️ GEMINI_API_KEY missing. Falling back to offline insights.")
        raise Exception("AI Unavailable")

    insights_text = json.dumps(insights_dict, indent=2)
    memory_context = json.dumps(memory or {}, indent=2)

    prompt = f"""
    You are an elite AI CFO analyzing corporate expense data.
    
    USER HISTORICAL INTELLIGENCE:
    {memory_context}

    IMPORTANT:
    - Use 'trend', 'risk_score', and 'alerts' from the historical intelligence above to personalize your insights.
    - Provide forward-looking strategic decisions, not just historical summaries.
    - Specifically highlight risks proactively if the risk_score is high or alerts are present.

    Upgrade your analysis from basic findings to highly actionable 'Decision Intelligence'.
    Break down your response exactly into the following 3 sections using exactly these headers:
    
    ### 📊 Cost Reduction & Efficiency
    ### ⚠️ Risk Mitigation & Compliance
    ### 📈 Strategic Growth Actions
    
    Under each header, provide a highly detailed insight using this exact structural format:
    - **Observed Trend:** [Detailed description of the spending pattern including historical context]
    - **Economic Impact:** [Quantifiable risk or cash-flow constraint]
    - **Executive Action:** [Concrete, specific CFO directive to resolve or capitalize on this]

    Do NOT deviate from this structure. Use sharp, executive, and decisive language.

    CURRENT DATA:
    {insights_text}
    """

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        return response.text.strip()

    except Exception as e:
        print("⚠️ Rate limit hit. Retrying in 25 sec...")
        time.sleep(25)

        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )
            return response.text.strip()
        except:
            return "AI insights unavailable"

# -------------------------------
# RAG CHAT ASSISTANT
# -------------------------------

def generate_chat_response(query: str, context: dict):
    if not client:
        print("⚠️ GEMINI_API_KEY missing. Booting offline chat engine...")
        resp = generate_local_fallback_chat(query, context)
        return f"<div class='alert-box alert-warning' style='margin-bottom: 16px;'><span class='material-symbols-rounded'>cloud_off</span> <span>AI Server Unreachable — Proceeding with Offline Heuristics Engine.</span></div>{resp}"

    context_str = json.dumps(context, indent=2)

    prompt = f"""
    You are an elite CFO-level financial analyst assisting a business owner with their expenses.
    Use the structured expense data context provided below to precisely answer the user's question.
    Give actionable, concise, and business-focused advice. 
    Never mention that you are an AI or that you were provided JSON text. Communicate naturally as an expert.
    
    Data Context:
    {context_str}

    User Question:
    {query}
    """

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        return response.text.strip()
    except Exception as e:
        print("⚠️ API Unavailable, routing to local semantic engine...")
        resp = generate_local_fallback_chat(query, context)
        return f"<div class='alert-box alert-warning' style='margin-bottom: 16px;'><span class='material-symbols-rounded'>cloud_off</span> <span>AI Server Unreachable — Proceeding with Offline Heuristics Engine.</span></div>{resp}"

def generate_local_fallback_chat(query: str, context: dict):
    q = query.lower()
    total = context.get("total_spend", 0)
    
    top_cat_name, top_cat_val = "Unknown", 0
    if context.get("top_categories"):
        top_cat_name = list(context["top_categories"].keys())[0]
        top_cat_val = list(context["top_categories"].values())[0]
        
    top_vendor_name, top_vendor_val = "Unknown", 0
    if context.get("top_vendors"):
        top_vendor_name = list(context["top_vendors"].keys())[0]
        top_vendor_val = list(context["top_vendors"].values())[0]

    cat_pct = (top_cat_val / total * 100) if total > 0 else 0
    ven_pct = (top_vendor_val / total * 100) if total > 0 else 0
    total_str = context.get("total_str")
    if not total_str:
        curr = context.get("currency", "₹")
        if curr == "₹":
            total_str = f"₹{total:,.0f}" # Simple fallback for IN
        else:
            total_str = f"{curr}{total:,.0f}" # Simple fallback for others

    if any(word in q for word in ["highest", "top", "most", "majority", "where"]):
        return f"""
Based on your current active data profile, the vast majority of your capital is being routed into **{top_cat_name}**, representing **{cat_pct:.1f}%** of your entire spend. 

Additionally, your primary payout goes to **{top_vendor_name}**, capturing **{ven_pct:.1f}%** of your outbound cash.
**Strategic Advice:** Consistently review your master agreements with `{top_vendor_name}` to ensure you are continually receiving maximum bulk volume discounts.
"""

    elif any(word in q for word in ["save", "cut", "reduce", "optimize", "cost", "lower"]):
        return f"""
To proactively reduce your **{total_str}** total expenditure, I recommend targeting your heaviest structural dependencies.

1. **Category Cap:** Implement a strict departmental cap on **{top_cat_name}** spending since it entirely dominates your operating budget.
2. **Vendor Negotiation:** Issue an RFP to two direct competitors of **{top_vendor_name}**. Inform them you are looking to relocate your massive volume to leverage a minimum 15% corporate rate reduction.
"""

    elif any(word in q for word in ["anomalies", "weird", "unusual", "error", "mistake", "fraud"]):
        return f"""
The underlying analytics engine has actively scanned your dataset rows against established deviation curves.

If your dashboard shows anomalies, **run an immediate compliance audit** on those specific lines to categorically rule out duplicate vendor billing or unauthorized shadow-IT software purchases!
"""

    else:
        return f"""
*(Offline Heuristic Engine Enabled)*

Looking deeply at your **{total_str}** infrastructure map:
- Your greatest operational vulnerability is a strong reliance on **{top_cat_name}** operations.
- Your highest absolute vendor pricing risk lies with **{top_vendor_name}**.

I highly recommend actively diversifying your vendor relationships and establishing rigorous pre-approval routing for any invoices relating to your top categories going forward! 
"""