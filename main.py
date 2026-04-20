from fastapi import FastAPI, UploadFile, File, Response
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import shutil
import os
import json
import datetime
import uvicorn
from spendguard_engine import run_pipeline
from ai_layer import generate_chat_response
from supabase import create_client
import jwt
from fastapi import Depends, HTTPException, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from dotenv import load_dotenv

load_dotenv()

# Supabase Configuration
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY") # Use Service Key for Backend Admin tasks
SUPABASE_JWT_SECRET = os.getenv("SUPABASE_JWT_SECRET")

if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
    print("⚠️ Supabase Credentials missing. User Memory will be disabled.")
    supabase = None
else:
    supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

security = HTTPBearer()

async def get_current_user(auth: HTTPAuthorizationCredentials = Security(security)):
    """Verifies the Supabase session token securely with defensive checks."""
    if not supabase:
        raise HTTPException(status_code=500, detail="Backend configuration error: Supabase client missing.")
        
    try:
        token = auth.credentials
        # Let Supabase handle the verification logic
        user_res = supabase.auth.get_user(token)
        
        # 🔥 SAFETY CHECK: Ensure we have a valid response and user object
        if not user_res or not hasattr(user_res, 'user') or not user_res.user:
            print("🔒 [AUTH] Failed: user_res or user_res.user is missing.")
            raise HTTPException(status_code=401, detail="Invalid session.")
            
        user = user_res.user
        print(f"✅ [AUTH] Verified: {user.email}")
        
        return {
            "sub": user.id,
            "email": user.email,
            "id": user.id
        }
    except Exception as e:
        print(f"❌ [AUTH ERROR]: {str(e)}")
        # If it's already an HTTPException, re-raise it
        if isinstance(e, HTTPException): raise e
        raise HTTPException(status_code=401, detail="Authentication failed.")

# -------------------------------
# INTELLIGENCE LAYER MODULES
# -------------------------------

def calculate_trend(memory, current_spend):
    history = memory.get("spend_history", [])
    history.append(current_spend)
    # Keep last 5 records for rolling analysis
    history = history[-5:]
    memory["spend_history"] = history

    if len(history) >= 2:
        if history[-1] > history[-2]:
            memory["trend"] = "increasing"
        elif history[-1] < history[-2]:
            memory["trend"] = "decreasing"
        else:
            memory["trend"] = "stable"
    else:
        memory["trend"] = "baseline"
    return memory

def calculate_risk(memory, insights):
    anomalies = insights.get("anomalies_count", 0)
    spend = insights.get("total_spend", 0)
    risk = 0

    if anomalies > 5:
        risk += 40
    if spend > 100000:
        risk += 30
    if memory.get("trend") == "increasing":
        risk += 20
        
    memory["risk_score"] = min(risk, 100)
    return memory

def update_vendor_intelligence(memory, insights):
    vendors = insights.get("top_vendors", {})
    vendor_risk = memory.get("vendor_risk", {})

    for vendor, amount in vendors.items():
        if vendor not in vendor_risk:
            vendor_risk[vendor] = 0
        if amount > 50000:
            vendor_risk[vendor] += 10
            
    memory["vendor_risk"] = vendor_risk
    memory["top_vendors_snapshot"] = list(vendors.keys())
    return memory

def generate_alerts(memory):
    alerts = []
    if memory.get("trend") == "increasing":
        alerts.append("Spending is rising consistently")
    if memory.get("risk_score", 0) > 60:
        alerts.append("High financial risk detected")
    
    for vendor, score in memory.get("vendor_risk", {}).items():
        if score > 50:
            alerts.append(f"Vendor {vendor} shows risk pattern")
            
    memory["alerts"] = alerts
    return memory

async def check_admin(user_payload: dict = Depends(get_current_user)):
    """Verifies if the current user is the Master Admin."""
    admin_email = "sangeetshaw39@gmail.com"
    if user_payload.get("email") != admin_email:
        print(f"🚫 Unauthorized Admin access attempt by {user_payload.get('email')}")
        raise HTTPException(status_code=403, detail="Forbidden: Admin access only.")
    return user_payload

# -------------------------------
# USER MEMORY ENGINE
# -------------------------------

def get_user_memory(user_id: str):
    if not supabase: return {}
    try:
        from postgrest.exceptions import APIError
        res = supabase.table("user_memory").select("memory").eq("user_id", user_id).maybe_single().execute()
        return res.data["memory"] if (res.data and "memory" in res.data) else {}
    except Exception as e:
        print("Memory Fetch Failed:", e)
        return {}

def update_user_memory(user_id: str, insights: dict, email: str = None):
    if not supabase: return
    try:
        # Fetch existing
        memory = get_user_memory(user_id)
        
        # Initialize Intelligence if missing
        if "spend_history" not in memory:
            memory.update({
                "spend_history": [],
                "trend": "baseline",
                "risk_score": 0,
                "vendor_risk": {},
                "alerts": []
            })

        # Run Intelligence Sequence
        memory = calculate_trend(memory, insights.get("total_spend", 0))
        memory = calculate_risk(memory, insights)
        memory = update_vendor_intelligence(memory, insights)
        memory = generate_alerts(memory)

        # Baseline details
        memory["last_total_spend"] = insights.get("total_spend")
        memory["total_analyses_count"] = memory.get("total_analyses_count", 0) + 1
        
        # Admin Metadata: Store email if provided
        if email:
            memory["email"] = email

        supabase.table("user_memory").upsert({
            "user_id": user_id,
            "memory": memory,
            "updated_at": datetime.datetime.utcnow().isoformat()
        }).execute()
        print(f"🧠 Intelligence updated for user {user_id}")
    except Exception as e:
        print("Memory Update Failed:", e)

from fastapi.staticfiles import StaticFiles

app = FastAPI()

# Mount static files BEFORE the home route to catch config.js, etc.
app.mount("/static", StaticFiles(directory="."), name="static")

# Explicit route for config.js at root level
@app.get("/config.js")
def serve_config_root():
    return FileResponse("config.js")

# Configuration
HISTORY_DIR = "history_data"
os.makedirs(HISTORY_DIR, exist_ok=True)

class ChatRequest(BaseModel):
    user_query: str
    context: dict

class RenameRequest(BaseModel):
    new_name: str

# ✅ CORS FIX
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Home route (serves UI)
@app.get("/", response_class=HTMLResponse)
def home():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return Response(status_code=204)

@app.get("/health")
def health():
    return {"status": "ok"}


# Upload + Analyze
@app.post("/analyze")
async def analyze_file(
    file: UploadFile = File(...),
    user_payload: dict = Depends(get_current_user)
):
    user_id = user_payload.get("sub")
    print(f"📥 File received from user {user_id}: {file.filename}")

    file_location = f"temp_{file.filename}"
    
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        # 1. Fetch Memory
        memory = get_user_memory(user_id)
        
        # 2. Run Pipeline with Memory
        result = run_pipeline(file_location, memory)

        # 3. Update Memory after successful analysis
        update_user_memory(user_id, result["insights"], user_payload.get("email"))

        print("✅ Pipeline completed")

        response = {
            "status": "success",
            "insights": result["insights"],
            "ai_insights": result["ai_insights"],
            "anomalies_count": len(result["anomalies"]),
            "clean_csv_string": result.get("clean_data_csv", ""),
            "warnings": result.get("warnings", [])
        }

    except Exception as e:
        print("❌ ERROR:", str(e))
        response = {
            "status": "error",
            "message": str(e)
        }

    if os.path.exists(file_location):
        os.remove(file_location)

    # Permanent History Save
    if response.get("status") == "success":
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        history_id = f"{timestamp}_{file.filename}"
        
        # 1. Save Full Payload
        payload_path = os.path.join(HISTORY_DIR, f"payload_{history_id}.json")
        with open(payload_path, "w", encoding="utf-8") as f:
            json.dump({
                "filename": file.filename,
                "date": datetime.datetime.now().isoformat(),
                "payload": response
            }, f)
            
        # 2. Save Tiny Metadata (for fast sidebar loading)
        meta_path = os.path.join(HISTORY_DIR, f"meta_{history_id}.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({
                "id": history_id,
                "filename": file.filename,
                "date": datetime.datetime.now().isoformat(),
                "spend": response.get("insights", {}).get("total_spend", 0),
                "currency": response.get("insights", {}).get("currency", "₹")
            }, f)
        
        response["history_id"] = history_id

    return response


@app.post("/chat")
async def chat_with_ai(req: ChatRequest):
    """Answers financial questions based on the provided context."""
    try:
        response_text = generate_chat_response(req.user_query, req.context)
        return {"status": "success", "response": response_text}
    except Exception as e:
        print(f"❌ [CHAT ERROR]: {str(e)}")
        return {"status": "error", "message": "The AI assistant is temporarily unavailable."}


@app.get("/history")
def list_history():
    history_list = []
    if not os.path.exists(HISTORY_DIR):
        return []
        
    for f in os.listdir(HISTORY_DIR):
        # 1. Standard Metadata Files
        filepath = os.path.join(HISTORY_DIR, f)
        if f.startswith("meta_") and f.endswith(".json"):
            try:
                if os.path.getsize(filepath) > 1 * 1024 * 1024:  # Skip files larger than 1MB
                    continue
                with open(filepath, "r", encoding="utf-8") as r:
                    history_list.append(json.load(r))
            except:
                continue
        # 2. Legacy Migration: Handle older files that don't have prefixes
        elif f.endswith(".json") and not f.startswith("payload_"):
            try:
                if os.path.getsize(filepath) > 5 * 1024 * 1024:  # Skip files larger than 5MB
                    continue
                with open(filepath, "r", encoding="utf-8") as r:
                    data = json.load(r)
                    history_list.append({
                        "id": f, # If it's legacy, the ID is just the filename
                        "is_legacy": True,
                        "filename": data.get("filename", f),
                        "date": data.get("date", ""),
                        "spend": data.get("payload", {}).get("insights", {}).get("total_spend", 0),
                        "currency": data.get("payload", {}).get("insights", {}).get("currency", "₹")
                    })
            except:
                continue
    
    # Sort by date (newest first)
    history_list.sort(key=lambda x: x.get("date", "") or "", reverse=True)
    return history_list

@app.get("/history/{id}")
def get_history_item(id: str):
    # Try payload first
    payload_path = os.path.join(HISTORY_DIR, f"payload_{id}.json")
    if os.path.exists(payload_path):
        with open(payload_path, "r", encoding="utf-8") as f:
            return json.load(f)
            
    # Fallback to legacy (where ID is the filename itself)
    legacy_path = os.path.join(HISTORY_DIR, id)
    if os.path.exists(legacy_path):
        with open(legacy_path, "r", encoding="utf-8") as f:
            return json.load(f)

    return {"status": "error", "message": "File not found"}

@app.delete("/history/{id}")
def delete_history_item(id: str):
    meta_path = os.path.join(HISTORY_DIR, f"meta_{id}.json")
    payload_path = os.path.join(HISTORY_DIR, f"payload_{id}.json")
    
    deleted = False
    if os.path.exists(meta_path):
        os.remove(meta_path)
        deleted = True
    if os.path.exists(payload_path):
        os.remove(payload_path)
        deleted = True
        
    legacy_path = os.path.join(HISTORY_DIR, id)
    if os.path.exists(legacy_path):
        os.remove(legacy_path)
        deleted = True
        
    if deleted:
        return {"status": "success"}
    return {"status": "error", "message": "File not found"}

@app.put("/history/{id}")
def rename_history_item(id: str, req: RenameRequest):
    print(f"🔄 Rename request for ID: {id} -> New Name: {req.new_name}")
    
    # 1. Try Standard (Meta + Payload)
    meta_path = os.path.join(HISTORY_DIR, f"meta_{id}.json")
    payload_path = os.path.join(HISTORY_DIR, f"payload_{id}.json")
    
    if os.path.exists(meta_path):
        print(f"📍 Updating metadata: {meta_path}")
        with open(meta_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        data["filename"] = req.new_name
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(data, f)
            
        if os.path.exists(payload_path):
            print(f"📍 Updating payload: {payload_path}")
            with open(payload_path, "r", encoding="utf-8") as f:
                p_data = json.load(f)
            p_data["filename"] = req.new_name
            with open(payload_path, "w", encoding="utf-8") as f:
                json.dump(p_data, f)
                
        return {"status": "success"}
        
    # 2. Try Legacy
    legacy_path = os.path.join(HISTORY_DIR, id)
    if os.path.exists(legacy_path):
        print(f"📍 Updating legacy file: {legacy_path}")
        with open(legacy_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        data["filename"] = req.new_name
        with open(legacy_path, "w", encoding="utf-8") as f:
            json.dump(data, f)
        return {"status": "success"}
        
    print(f"❌ FAILED: Metadata file not found for ID: {id}")
    return {"status": "error", "message": "File not found"}

@app.post("/history/{id}/reanalyze")
def reanalyze_history_item(id: str, user_id: str = Depends(get_current_user)):
    print(f"🔄 Re-analyzing Report ID: {id} for user {user_id}")
    payload_path = os.path.join(HISTORY_DIR, f"payload_{id}.json")
    
    if not os.path.exists(payload_path):
        # Fallback to legacy if no prefix found
        payload_path = os.path.join(HISTORY_DIR, id)
        
    if os.path.exists(payload_path):
        with open(payload_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Extract core insights for the AI
        # Handle both new 'payload' nested structure and legacy flat structure
        insights = data.get("payload", {}).get("insights") or data.get("insights")
        
        if not insights:
            return {"status": "error", "message": "No data insights found to analyze."}
            
        try:
            from ai_layer import generate_ai_insights
            new_ai_text = generate_ai_insights(insights)
            
            # Update the stored text
            if "payload" in data:
                data["payload"]["ai_insights"] = new_ai_text
            else:
                data["ai_insights"] = new_ai_text
                
            with open(payload_path, "w", encoding="utf-8") as f:
                json.dump(data, f)
                
            return {"status": "success", "payload": data.get("payload", data)}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    return {"status": "error", "message": "File not found"}

# -------------------------------
# ADMIN ENDPOINTS (Restricted)
# -------------------------------

@app.get("/admin/stats")
async def get_admin_stats(admin: dict = Depends(check_admin)):
    if not supabase: return {"status": "error", "message": "Supabase not connected"}
    
    try:
        # Total Users (from memory table)
        users_res = supabase.table("user_memory").select("user_id", count="exact").execute()
        total_users = users_res.count if hasattr(users_res, 'count') else len(users_res.data)

        # Global Analysis Stats (from analyses table)
        analyses_res = supabase.table("analyses").select("spend").execute()
        total_analyses = len(analyses_res.data)
        total_spend = sum([item.get("spend", 0) for item in analyses_res.data])

        return {
            "status": "success",
            "total_users": total_users,
            "total_analyses": total_analyses,
            "total_spend": total_spend
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/admin/users")
async def get_admin_users(admin: dict = Depends(check_admin)):
    if not supabase: return {"status": "error", "message": "Supabase not connected"}
    
    try:
        res = supabase.table("user_memory").select("*").order("updated_at", desc=True).execute()
        return {"status": "success", "users": res.data}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.delete("/admin/user/{user_id}")
async def delete_user_data(user_id: str, admin: dict = Depends(check_admin)):
    if not supabase: return {"status": "error", "message": "Supabase not connected"}
    
    try:
        # Delete Memory
        supabase.table("user_memory").delete().eq("user_id", user_id).execute()
        # Delete all Analyses
        supabase.table("analyses").delete().eq("user_id", user_id).execute()
        
        return {"status": "success", "message": f"Data for user {user_id} deleted."}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/admin")
def admin_page():
    with open("admin.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

if __name__ == "__main__":
    # Render binds dynamically to a port provided by the environment
    port = int(os.environ.get("PORT", 8000))
    print(f"🚀 SpendGuard AI is active and serving on port {port}...")
    uvicorn.run(app, host="0.0.0.0", port=port)