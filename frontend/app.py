import os
import sys
import uuid
from typing import Dict, Any, Optional
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# Ensure meddibuddy_assignment root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from graph.workflow import run_advisory_agent
from policies.sop_engine import SOPEngine

app = FastAPI(
    title="Weather-Advisory Support Bot",
    description="Safety policy compliance bot backed by LangGraph and live Open-Meteo data",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

sop_engine = SOPEngine()
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    simulate_weather_failure: bool = False

class ChatResponse(BaseModel):
    session_id: str
    response: str
    response_type: str
    location_name: Optional[str] = None
    activity: Optional[str] = None
    timeframe: Optional[str] = None
    primary_sop: Optional[Dict[str, Any]] = None
    contributing_sops: Optional[list] = None
    weather_data: Optional[Dict[str, Any]] = None
    decision_trace: Optional[Dict[str, Any]] = None

@app.post("/api/chat", response_model=ChatResponse)
def handle_chat(req: ChatRequest):
    if not req.message or not req.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    session_id = req.session_id or f"session_{uuid.uuid4().hex[:8]}"

    try:
        agent_result = run_advisory_agent(
            session_id=session_id,
            user_input=req.message.strip(),
            simulate_weather_failure=req.simulate_weather_failure
        )

        return ChatResponse(
            session_id=session_id,
            response=agent_result.get("final_response", ""),
            response_type=agent_result.get("response_type", "unknown"),
            location_name=agent_result.get("location_name"),
            activity=agent_result.get("activity"),
            timeframe=agent_result.get("timeframe"),
            primary_sop=agent_result.get("primary_sop"),
            contributing_sops=agent_result.get("contributing_sops"),
            weather_data=agent_result.get("weather_data"),
            decision_trace=agent_result.get("decision_trace")
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Graph execution failed: {str(e)}")

@app.get("/api/sops")
def get_sops():
    """Returns active SOPs from sops.json. Updates dynamically without restart."""
    sops = sop_engine.get_all_sops()
    return {"total": len(sops), "sops": sops}

@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "openrouter_key_set": bool(os.getenv("OPENROUTER_API_KEY")),
        "openrouter_model": os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct:free"),
        "total_active_sops": len(sop_engine.get_all_sops())
    }

# Mount frontend static files
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/")
def serve_index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))

if __name__ == "__main__":
    import uvicorn
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", 8000))
    print(f"Starting Weather Advisory Support Bot server on http://{host}:{port}")
    uvicorn.run("frontend.app:app", host=host, port=port, reload=True)
