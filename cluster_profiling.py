fastapi==0.112.2
uvicorn[standard]==0.30.6
requests==2.32.3
langchain==0.2.14
langchain-openai==0.1.23
azure-identity==1.17.1



AZURE_OPENAI_ENDPOINT=https://<your-resource>.openai.azure.com

# Your deployment names
CHAT_DEPLOYMENT=gpt-4o
EMBED_DEPLOYMENT=text-embedding-3-large

# API versions
CHAT_API_VERSION=2024-08-01-preview
EMB_API_VERSION=2025-02-01-preview

# Sessions (idle timeout in seconds)
SESSION_TTL_SECONDS=1800



import os
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings

AZURE_OPENAI_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT", "").rstrip("/")
CHAT_DEPLOYMENT = os.environ.get("CHAT_DEPLOYMENT", "gpt-4o")
EMBED_DEPLOYMENT = os.environ.get("EMBED_DEPLOYMENT", "text-embedding-3-large")
CHAT_API_VERSION = os.environ.get("CHAT_API_VERSION", "2024-08-01-preview")
EMB_API_VERSION  = os.environ.get("EMB_API_VERSION",  "2025-02-01-preview")

if not AZURE_OPENAI_ENDPOINT:
    raise RuntimeError("AZURE_OPENAI_ENDPOINT is required")

# If AZURE_OPENAI_API_KEY is unset, fall back to Azure CLI (AAD)
USE_AAD = "AZURE_OPENAI_API_KEY" not in os.environ
azure_ad_token_provider = None
if USE_AAD:
    from azure.identity import AzureCliCredential, get_bearer_token_provider
    cred = AzureCliCredential()
    scope = "https://cognitiveservices.azure.com/.default"
    azure_ad_token_provider = get_bearer_token_provider(cred, scope)

def build_chat_client(deployment: str | None = None) -> AzureChatOpenAI:
    dep = deployment or CHAT_DEPLOYMENT
    return AzureChatOpenAI(
        azure_deployment=dep,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_version=CHAT_API_VERSION,
        azure_ad_token_provider=azure_ad_token_provider,
    )

def build_embed_client(deployment: str | None = None) -> AzureOpenAIEmbeddings:
    dep = deployment or EMBED_DEPLOYMENT
    return AzureOpenAIEmbeddings(
        model=dep,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_version=EMB_API_VERSION,
        azure_ad_token_provider=azure_ad_token_provider,
    )





from time import time
from typing import Dict, Literal
from uuid import uuid4

from app.services.azure_clients import build_chat_client, build_embed_client

class Session:
    def __init__(self, kind: Literal["chat","embed"], deployment: str):
        self.id = str(uuid4())
        self.kind = kind
        self.deployment = deployment
        self.created_at = time()
        self.last_used = time()
        if kind == "chat":
            self.client = build_chat_client(deployment)
        else:
            self.client = build_embed_client(deployment)

    def touch(self):
        self.last_used = time()

class SessionStore:
    def __init__(self, ttl_seconds: int):
        self.ttl = ttl_seconds
        self._sessions: Dict[str, Session] = {}

    def _gc(self):
        now = time()
        purge = [sid for sid,s in self._sessions.items() if (now - s.last_used) > self.ttl]
        for sid in purge:
            self._sessions.pop(sid, None)

    def create(self, kind: Literal["chat","embed"], deployment: str) -> Session:
        self._gc()
        s = Session(kind, deployment)
        self._sessions[s.id] = s
        return s

    def get(self, sid: str) -> Session | None:
        self._gc()
        s = self._sessions.get(sid)
        if s:
            s.touch()
        return s

    def delete(self, sid: str) -> bool:
        self._gc()
        return bool(self._sessions.pop(sid, None))





import os
from fastapi import APIRouter
from app.services.azure_clients import (
    AZURE_OPENAI_ENDPOINT, CHAT_DEPLOYMENT, EMBED_DEPLOYMENT,
    CHAT_API_VERSION, EMB_API_VERSION
)

router = APIRouter(prefix="/models", tags=["models"])

@router.get("")
def list_models():
    return {
        "endpoint": AZURE_OPENAI_ENDPOINT,
        "auth": "aad" if "AZURE_OPENAI_API_KEY" not in os.environ else "api_key",
        "chat":  {"deployments": [CHAT_DEPLOYMENT],  "api_version": CHAT_API_VERSION},
        "embed": {"deployments": [EMBED_DEPLOYMENT], "api_version": EMB_API_VERSION},
    }

@router.get("/health")
def health():
    return {"ok": True}






from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Literal
from langchain.schema import HumanMessage

from app.stores.sessions import SessionStore
from app.services.azure_clients import CHAT_DEPLOYMENT, EMBED_DEPLOYMENT
import os

SESSION_TTL_SECONDS = int(os.environ.get("SESSION_TTL_SECONDS", "1800"))
store = SessionStore(ttl_seconds=SESSION_TTL_SECONDS)

router = APIRouter(prefix="/sessions", tags=["sessions"])

class CreateSession(BaseModel):
    type: Literal["chat","embed"]
    deployment: str | None = None

class InvokeReq(BaseModel):
    prompt: str

class EmbedReq(BaseModel):
    text: str

@router.post("")
def create_session(body: CreateSession):
    dep = body.deployment or (CHAT_DEPLOYMENT if body.type == "chat" else EMBED_DEPLOYMENT)
    s = store.create(kind=body.type, deployment=dep)
    return {"session_id": s.id, "type": s.kind, "deployment": s.deployment, "expires_in_seconds": SESSION_TTL_SECONDS}

def _get_or_404(sid: str):
    s = store.get(sid)
    if not s:
        raise HTTPException(404, "session not found or expired")
    return s

@router.post("/{sid}/invoke")
def invoke(sid: str, body: InvokeReq):
    s = _get_or_404(sid)
    if s.kind != "chat":
        raise HTTPException(400, "session is not of type 'chat'")
    resp = s.client.invoke([HumanMessage(content=body.prompt)])
    return {"content": resp.content}

@router.post("/{sid}/embed")
def embed(sid: str, body: EmbedReq):
    s = _get_or_404(sid)
    if s.kind != "embed":
        raise HTTPException(400, "session is not of type 'embed'")
    vec = s.client.embed_query(body.text)
    return {"embedding": vec}

@router.delete("/{sid}")
def close_session(sid: str):
    ok = store.delete(sid)
    if not ok:
        raise HTTPException(404, "session not found")
    return {"ok": True}





from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers.models_router import router as models_router
from app.routers.sessions_router import router as sessions_router

app = FastAPI(title="OpenAI Model Gateway", version="0.1.0")

# allow all for simplicity (tighten later if needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(models_router)
app.include_router(sessions_router)








import os, requests

BASE = os.environ.get("GATEWAY_BASE", "http://localhost:8000")
TIMEOUT = 30

def main():
    print("Models:", requests.get(f"{BASE}/models", timeout=TIMEOUT).json())

    # chat
    r = requests.post(f"{BASE}/sessions", json={"type":"chat"}, timeout=TIMEOUT); r.raise_for_status()
    chat_sid = r.json()["session_id"]
    r = requests.post(f"{BASE}/sessions/{chat_sid}/invoke", json={"prompt":"Say hi!"}, timeout=TIMEOUT); r.raise_for_status()
    print("Chat:", r.json()["content"])

    # embed
    r = requests.post(f"{BASE}/sessions", json={"type":"embed"}, timeout=TIMEOUT); r.raise_for_status()
    emb_sid = r.json()["session_id"]
    r = requests.post(f"{BASE}/sessions/{emb_sid}/embed", json={"text":"hello world"}, timeout=TIMEOUT); r.raise_for_status()
    vec = r.json()["embedding"]
    print("Embedding dims:", len(vec))

    requests.delete(f"{BASE}/sessions/{chat_sid}", timeout=TIMEOUT)
    requests.delete(f"{BASE}/sessions/{emb_sid}", timeout=TIMEOUT)

if __name__ == "__main__":
    main()
