"""
API REST do Modelo Híbrido de LLM.
==================================
FastAPI expondo /process, /chat e /agent. Usa config, pipeline e chat_session.
"""

from __future__ import annotations
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional
from uuid import uuid4

# Raiz do projeto
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
except ImportError:
    FastAPI = None  # type: ignore
    HTTPException = None  # type: ignore
    BaseModel = object  # type: ignore


# -----------------------------------------------------------------------------
# Modelos de request/response
# -----------------------------------------------------------------------------
class ProcessRequest(BaseModel):
    prompt: str
    session_id: Optional[str] = None
    use_agent: Optional[bool] = None
    skip_l5: bool = False


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None


class ProcessResponse(BaseModel):
    response: str
    truth_value: float
    state: str
    certainty: float
    contradiction: float
    confidence_label: str
    session_id: Optional[str] = None


# -----------------------------------------------------------------------------
# Estado global (sessões de chat, pipeline, config)
# -----------------------------------------------------------------------------
def _load_app_state():
    from config_loader import load_config
    from pipeline import HybridLLMPipeline
    from chat_session import ChatSession
    config = load_config()
    pipeline = HybridLLMPipeline(config=config, verbose=False)
    sessions: Dict[str, ChatSession] = {}
    max_turns = config.get("chat", {}).get("max_turns_in_context", 10)
    return config, pipeline, sessions, max_turns


if FastAPI is None:
    app = None
else:
    app = FastAPI(title="Modelo Híbrido de LLM", version="1.0")
    _config, _pipeline, _sessions, _max_turns = _load_app_state()

    @app.get("/health")
    def health():
        return {"status": "ok", "model": "hybrid_llm"}

    @app.post("/process", response_model=ProcessResponse)
    def process(req: ProcessRequest):
        session_id = req.session_id or str(uuid4())
        session = _sessions.get(session_id)
        if session:
            session.add_user(req.prompt)
        try:
            result = _pipeline.process(
                req.prompt,
                chat_session=session,
                use_agent=req.use_agent,
                skip_l5=req.skip_l5,
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        if session:
            session.add_assistant(result.response)
        return ProcessResponse(
            response=result.response,
            truth_value=result.truth_value,
            state=result.state,
            certainty=result.certainty,
            contradiction=result.contradiction,
            confidence_label=result.confidence_label,
            session_id=session_id,
        )

    @app.post("/chat", response_model=ProcessResponse)
    def chat(req: ChatRequest):
        session_id = req.session_id or str(uuid4())
        if session_id not in _sessions:
            from chat_session import ChatSession
            _sessions[session_id] = ChatSession(max_turns=_max_turns)
        session = _sessions[session_id]
        session.add_user(req.message)
        try:
            result = _pipeline.process(req.message, chat_session=session, use_agent=None, skip_l5=False)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        session.add_assistant(result.response)
        return ProcessResponse(
            response=result.response,
            truth_value=result.truth_value,
            state=result.state,
            certainty=result.certainty,
            contradiction=result.contradiction,
            confidence_label=result.confidence_label,
            session_id=session_id,
        )

    class AgentRequest(BaseModel):
        query: str

    @app.post("/agent")
    def agent_search(req: AgentRequest):
        """Chama apenas o agente de pesquisa (busca local + internet)."""
        try:
            from agente_busca_web import run_search_for_context
            text = run_search_for_context(req.query)
            return {"answer": text, "query": req.query}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))


def run_api():
    if app is None:
        print("Instale fastapi e uvicorn: pip install fastapi uvicorn", file=sys.stderr)
        sys.exit(1)
    import uvicorn
    from config_loader import load_config
    cfg = load_config()
    api_cfg = cfg.get("api", {})
    host = api_cfg.get("host", "0.0.0.0")
    port = int(api_cfg.get("port", 8000))
    uvicorn.run("api:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    run_api()
