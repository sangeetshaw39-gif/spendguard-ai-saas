from fastapi import FastAPI, UploadFile, File, Response
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import shutil
import os
import json
import datetime
import uvicorn
from spendguard_engine import run_pipeline
from ai_layer import generate_chat_response
import jwt
from fastapi import Depends, HTTPException, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from dotenv import load_dotenv

load_dotenv()

# Supabase Auth Configuration
SUPABASE_JWT_SECRET = os.getenv("SUPABASE_JWT_SECRET")
security = HTTPBearer()

async def get_current_user(auth: HTTPAuthorizationCredentials = Security(security)):
    """Decodes and verifies the Supabase session token."""
    if not SUPABASE_JWT_SECRET:
        raise HTTPException(status_code=500, detail="Backend configuration error: JWT Secret missing.")
        
    try:
        token = auth.credentials
        # Supabase uses HS256 and standard JWT claims
        payload = jwt.decode(
            token, 
            SUPABASE_JWT_SECRET, 
            algorithms=["HS256"], 
            options={"verify_aud": False} # Supabase uses project-specific audience
        )
        return payload.get("sub") # Returns the User UUID
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token has expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid authentication token")
    except Exception as e:
        raise HTTPException(status_code=401, detail=str(e))

app = FastAPI()

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
    user_id: str = Depends(get_current_user)
):
    print(f"📥 File received from user {user_id}: {file.filename}")

    file_location = f"temp_{file.filename}"
    
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        print("⚙️ Running pipeline...")

        result = run_pipeline(file_location)

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

@app.post("/chat")
def chat_endpoint(request: ChatRequest, user_id: str = Depends(get_current_user)):
    try:
        print(f"💬 Chat request from user {user_id}")
        response = generate_chat_response(request.user_query, request.context)
        return {"status": "success", "response": response}
    except Exception as e:
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    # Render binds dynamically to a port provided by the environment
    port = int(os.environ.get("PORT", 8000))
    print(f"🚀 SpendGuard AI is active and serving on port {port}...")
    uvicorn.run(app, host="0.0.0.0", port=port)