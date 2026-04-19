import os
import json
import hmac
import uuid
import base64
import hashlib
import random
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from typing import Dict, List, Optional, Set, Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Form, UploadFile, File, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

APP_TITLE = "MicroLab Nexus"
APP_SECRET = os.getenv("MICROLAB_SECRET", "change-this-secret-in-production")
DB_PATH = os.getenv("MICROLAB_DB", "microlab_nexus.db")
MAX_PUBLIC_MESSAGES = 150
MAX_ACTIVITIES = 20
ALLOWED_CHANNELS = {"public", "lab", "random"}

MICROBE_NAMES = [
    "E. coli", "Bacillus subtilis", "Saccharomyces cerevisiae", "Lactobacillus",
    "Streptococcus pyogenes", "Pseudomonas aeruginosa", "Aspergillus niger",
    "Staphylococcus aureus", "Vibrio cholerae", "Clostridium tetani",
    "Rhizobium", "Cyanobacteria", "Deinococcus radiodurans", "Thermus aquaticus",
    "Helicobacter pylori", "Candida albicans", "Penicillium chrysogenum"
]

FACTS = [
    "Agar plates were introduced into microbiology by Fanny Hesse.",
    "Biofilms can protect microbes from antibiotics and disinfectants.",
    "Thermus aquaticus gave science Taq polymerase for PCR.",
    "Some bacteria can survive intense radiation and desiccation.",
    "The human microbiome influences digestion, immunity, and metabolism.",
]

QUOTES = [
    "Label the tubes before the coffee kicks in.",
    "PCR is just hope with temperature cycles.",
    "The incubator remembers what the notebook forgot.",
    "Every colony tells a story if you plate it right.",
]

BADGES = {
    "first_chat": "💬 First Words",
    "file_upload": "📎 Sharer",
    "secret_sender": "🤫 Whisperer",
    "poll_creator": "📊 Poll Master",
    "shake_master": "🧪 Flask Shaker",
    "inoculator": "🦠 Streaker",
    "game_master": "🎮 Game Master",
}

# ---------- Game state (in‑memory) ----------
active_games: Dict[str, dict] = {}   # game_id -> {type, players, state, ...}

app = FastAPI(title=APP_TITLE)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

active_ws: Dict[str, WebSocket] = {}
typing_by_channel: Dict[str, Set[str]] = {"public": set(), "lab": set(), "random": set()}
user_channel_state: Dict[str, str] = {}
lab_activities: List[dict] = []

# ------------------------------------------------------------------
# Database helpers (thread-safe with WAL mode)
# ------------------------------------------------------------------
def get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=10.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn

def init_db() -> None:
    with closing(get_db()) as conn:
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS users (
                username TEXT PRIMARY KEY,
                microbe TEXT NOT NULL,
                avatar_color TEXT NOT NULL,
                bio TEXT DEFAULT 'Lab enthusiast',
                points INTEGER DEFAULT 0,
                badges_json TEXT DEFAULT '[]',
                created_at TEXT NOT NULL
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                username TEXT NOT NULL,
                message TEXT NOT NULL,
                channel TEXT NOT NULL,
                created_at TEXT NOT NULL,
                edited INTEGER DEFAULT 0
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS private_messages (
                id TEXT PRIMARY KEY,
                sender TEXT NOT NULL,
                recipient TEXT NOT NULL,
                message TEXT NOT NULL,
                created_at TEXT NOT NULL,
                is_read INTEGER DEFAULT 0
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS polls (
                id TEXT PRIMARY KEY,
                question TEXT NOT NULL,
                options_json TEXT NOT NULL,
                votes_json TEXT NOT NULL,
                created_by TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS files (
                id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                mime TEXT NOT NULL,
                data_b64 TEXT NOT NULL,
                uploader TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        conn.commit()

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

# ------------------------------------------------------------------
# Auth helpers (no passwords)
# ------------------------------------------------------------------
def sign_value(value: str) -> str:
    sig = hmac.new(APP_SECRET.encode(), value.encode(), hashlib.sha256).hexdigest()
    return f"{value}.{sig}"

def verify_value(signed_value: Optional[str]) -> Optional[str]:
    if not signed_value or "." not in signed_value:
        return None
    value, sig = signed_value.rsplit(".", 1)
    expected = hmac.new(APP_SECRET.encode(), value.encode(), hashlib.sha256).hexdigest()
    if hmac.compare_digest(sig, expected):
        return value
    return None

def session_cookie_for_user(username: str) -> str:
    payload = json.dumps({"u": username}, separators=(",", ":"))
    token = base64.urlsafe_b64encode(payload.encode()).decode()
    return sign_value(token)

def decode_session(cookie_value: Optional[str]) -> Optional[str]:
    raw = verify_value(cookie_value)
    if not raw:
        return None
    try:
        payload = json.loads(base64.urlsafe_b64decode(raw.encode()).decode())
        return payload.get("u")
    except Exception:
        return None

def get_user(username: str) -> Optional[sqlite3.Row]:
    with closing(get_db()) as conn:
        cur = conn.cursor()
        cur.execute("SELECT * FROM users WHERE username = ?", (username,))
        return cur.fetchone()

def create_user_if_not_exists(username: str) -> None:
    if get_user(username):
        return
    with closing(get_db()) as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO users (username, microbe, avatar_color, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (
                username,
                random.choice(MICROBE_NAMES),
                "#{:06x}".format(random.randint(0x0C8A6E, 0x8EEFD2)),
                now_iso(),
            ),
        )
        conn.commit()

def update_points(username: str, delta: int, action: Optional[str] = None) -> None:
    with closing(get_db()) as conn:
        cur = conn.cursor()
        cur.execute("UPDATE users SET points = COALESCE(points, 0) + ? WHERE username = ?", (delta, username))
        conn.commit()
    if action:
        add_activity(username, action, delta)

def get_badges(username: str) -> List[str]:
    row = get_user(username)
    if not row:
        return []
    try:
        return json.loads(row["badges_json"] or "[]")
    except Exception:
        return []

def award_badge(username: str, badge_key: str) -> None:
    badges = get_badges(username)
    if badge_key in badges:
        return
    badges.append(badge_key)
    with closing(get_db()) as conn:
        cur = conn.cursor()
        cur.execute("UPDATE users SET badges_json = ? WHERE username = ?", (json.dumps(badges), username))
        conn.commit()
    add_activity(username, f"earned badge {BADGES.get(badge_key, badge_key)}", 10)

def add_activity(user: str, action: str, points: int = 0) -> None:
    lab_activities.insert(0, {
        "time": datetime.now().strftime("%H:%M:%S"),
        "user": user,
        "action": action,
        "points": points,
    })
    del lab_activities[MAX_ACTIVITIES:]

def save_public_message(username: str, message: str, channel: str) -> dict:
    msg = {
        "id": str(uuid.uuid4()),
        "username": username,
        "message": message,
        "channel": channel,
        "timestamp": now_iso(),
        "edited": False,
    }
    with closing(get_db()) as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO messages (id, username, message, channel, created_at, edited)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (msg["id"], username, message, channel, msg["timestamp"], 0),
        )
        conn.commit()
    return msg

def delete_message(message_id: str, requester: str) -> bool:
    with closing(get_db()) as conn:
        cur = conn.cursor()
        cur.execute("SELECT username FROM messages WHERE id = ?", (message_id,))
        row = cur.fetchone()
        if not row or row["username"] != requester:
            return False
        cur.execute("DELETE FROM messages WHERE id = ?", (message_id,))
        conn.commit()
        return True

def list_users_online() -> List[dict]:
    users = []
    for username in sorted(active_ws.keys()):
        row = get_user(username)
        if not row:
            continue
        users.append({
            "username": username,
            "points": row["points"],
            "badges": json.loads(row["badges_json"] or "[]"),
            "avatar_color": row["avatar_color"],
            "channel": user_channel_state.get(username, "public"),
            "online": True,
        })
    return users

def list_leaderboard(limit: int = 10) -> List[dict]:
    with closing(get_db()) as conn:
        cur = conn.cursor()
        cur.execute("SELECT username, points FROM users ORDER BY points DESC, username ASC LIMIT ?", (limit,))
        rows = cur.fetchall()
        return [{"username": r["username"], "points": r["points"]} for r in rows]

def list_files() -> List[dict]:
    with closing(get_db()) as conn:
        cur = conn.cursor()
        cur.execute("SELECT id, filename, uploader, created_at FROM files ORDER BY created_at DESC")
        rows = cur.fetchall()
        return [{
            "id": r["id"],
            "filename": r["filename"],
            "uploader": r["uploader"],
            "time": datetime.fromisoformat(r["created_at"]).strftime("%H:%M:%S"),
        } for r in rows]

def delete_file(file_id: str, requester: str) -> bool:
    with closing(get_db()) as conn:
        cur = conn.cursor()
        cur.execute("SELECT uploader FROM files WHERE id = ?", (file_id,))
        row = cur.fetchone()
        if not row or row["uploader"] != requester:
            return False
        cur.execute("DELETE FROM files WHERE id = ?", (file_id,))
        conn.commit()
        return True

def list_polls() -> List[dict]:
    with closing(get_db()) as conn:
        cur = conn.cursor()
        cur.execute("SELECT * FROM polls ORDER BY created_at DESC")
        rows = cur.fetchall()
        return [{
            "id": r["id"],
            "question": r["question"],
            "options": json.loads(r["options_json"]),
            "votes": json.loads(r["votes_json"]),
            "created_by": r["created_by"],
        } for r in rows]

def list_unread_private_messages(username: str) -> List[dict]:
    with closing(get_db()) as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT * FROM private_messages WHERE recipient = ? AND is_read = 0 ORDER BY created_at ASC",
            (username,),
        )
        rows = cur.fetchall()
        cur.execute("UPDATE private_messages SET is_read = 1 WHERE recipient = ?", (username,))
        conn.commit()
        return [{
            "id": r["id"],
            "from": r["sender"],
            "to": r["recipient"],
            "message": r["message"],
            "time": datetime.fromisoformat(r["created_at"]).strftime("%H:%M:%S"),
        } for r in rows]

def save_private_message(sender: str, recipient: str, message: str) -> dict:
    item = {
        "id": str(uuid.uuid4()),
        "from": sender,
        "to": recipient,
        "message": message,
        "time": datetime.now().strftime("%H:%M:%S"),
    }
    with closing(get_db()) as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO private_messages (id, sender, recipient, message, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (item["id"], sender, recipient, message, now_iso()),
        )
        conn.commit()
    return item

def get_unread_private_count(username: str) -> int:
    with closing(get_db()) as conn:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM private_messages WHERE recipient = ? AND is_read = 0", (username,))
        return cur.fetchone()[0]

def update_user_bio(username: str, bio: str) -> None:
    with closing(get_db()) as conn:
        cur = conn.cursor()
        cur.execute("UPDATE users SET bio = ? WHERE username = ?", (bio, username))
        conn.commit()

# ------------------------------------------------------------------
# WebSocket Manager
# ------------------------------------------------------------------
class ConnectionManager:
    async def connect(self, websocket: WebSocket, username: str) -> None:
        await websocket.accept()
        active_ws[username] = websocket
        user_channel_state[username] = user_channel_state.get(username, "public")
        add_activity(username, "joined the lab", 5)
        await self.broadcast_user_list()
        await self.broadcast_activity()

    def disconnect(self, username: str) -> None:
        active_ws.pop(username, None)
        for s in typing_by_channel.values():
            s.discard(username)
        add_activity(username, "left the lab", 0)

    async def safe_send(self, websocket: WebSocket, payload: dict) -> None:
        try:
            await websocket.send_text(json.dumps(payload))
        except Exception:
            pass

    async def broadcast_all(self, payload: dict, exclude_username: Optional[str] = None) -> None:
        for uname, wsock in list(active_ws.items()):
            if exclude_username and uname == exclude_username:
                continue
            await self.safe_send(wsock, payload)

    async def broadcast_channel(self, channel: str, payload: dict) -> None:
        for uname, wsock in list(active_ws.items()):
            if user_channel_state.get(uname, "public") == channel:
                await self.safe_send(wsock, payload)

    async def broadcast_user_list(self) -> None:
        users = list_users_online()
        await self.broadcast_all({"type": "user_list", "users": users, "count": len(users)})

    async def broadcast_activity(self) -> None:
        await self.broadcast_all({"type": "lab_activity", "activities": lab_activities[:MAX_ACTIVITIES]})

    async def broadcast_leaderboard(self) -> None:
        await self.broadcast_all({"type": "leaderboard", "leaderboard": list_leaderboard()})

    async def set_typing(self, username: str, channel: str, is_typing: bool) -> None:
        if channel not in typing_by_channel:
            channel = "public"
        if is_typing:
            typing_by_channel[channel].add(username)
        else:
            typing_by_channel[channel].discard(username)
        await self.broadcast_channel(channel, {
            "type": "typing",
            "channel": channel,
            "users": sorted(list(typing_by_channel[channel])),
        })

manager = ConnectionManager()

def get_request_user(request: Request) -> Optional[str]:
    cookie_user = decode_session(request.cookies.get("session"))
    if cookie_user and get_user(cookie_user):
        return cookie_user
    token_user = decode_session(request.query_params.get("token"))
    if token_user and get_user(token_user):
        return token_user
    return None

# ------------------------------------------------------------------
# Game helpers
# ------------------------------------------------------------------
def create_game(game_type: str, players: List[str]) -> str:
    game_id = str(uuid.uuid4())
    if game_type == "tictactoe":
        state = {
            "board": [""] * 9,
            "turn": players[0],
            "winner": None,
        }
    elif game_type == "rps":
        state = {
            "moves": {},
            "winner": None,
        }
    elif game_type == "truthdare":
        state = {"last": None}
    elif game_type == "numberguess":
        state = {
            "secret": random.randint(1, 100),
            "guesses": {},
            "winner": None,
        }
    else:
        raise ValueError("Unknown game type")
    active_games[game_id] = {
        "type": game_type,
        "players": players,
        "state": state,
        "created_at": now_iso(),
    }
    return game_id

# ------------------------------------------------------------------
# Startup & Routes
# ------------------------------------------------------------------
@app.on_event("startup")
async def on_startup() -> None:
    init_db()

@app.get("/")
async def root(request: Request):
    user = get_request_user(request)
    if user:
        return RedirectResponse("/dashboard")
    return HTMLResponse(LOGIN_HTML, headers={"Cache-Control": "no-store"})

@app.get("/dashboard")
async def dashboard(request: Request):
    user = get_request_user(request)
    if not user:
        return RedirectResponse("/")
    html = CHAT_HTML.replace("{{USERNAME}}", user).replace("{{SESSION}}", session_cookie_for_user(user))
    return HTMLResponse(html, headers={"Cache-Control": "no-store"})

@app.post("/join")
async def join(username: str = Form(...)):
    username = username.strip()
    if len(username) < 3:
        return JSONResponse({"success": False, "message": "Username too short (min 3 chars)."})
    try:
        create_user_if_not_exists(username)
    except Exception as e:
        return JSONResponse({"success": False, "message": f"Database error: {str(e)}"})
    token = session_cookie_for_user(username)
    response = JSONResponse({"success": True, "username": username, "token": token})
    response.set_cookie("session", token, httponly=True, samesite="lax", max_age=60 * 60 * 24 * 7)
    return response

@app.post("/logout")
async def logout():
    response = JSONResponse({"success": True})
    response.delete_cookie("session")
    return response

@app.post("/upload")
async def upload_file(request: Request, file: UploadFile = File(...)):
    username = get_request_user(request)
    if not username:
        raise HTTPException(status_code=401, detail="Not authenticated")
    raw = await file.read()
    file_id = str(uuid.uuid4())
    with closing(get_db()) as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO files (id, filename, mime, data_b64, uploader, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                file_id,
                file.filename or "uploaded.bin",
                file.content_type or "application/octet-stream",
                base64.b64encode(raw).decode(),
                username,
                now_iso(),
            ),
        )
        conn.commit()
    update_points(username, 10, f"uploaded file {file.filename}")
    award_badge(username, "file_upload")
    await manager.broadcast_all({
        "type": "file_upload",
        "file_id": file_id,
        "filename": file.filename,
        "uploader": username,
        "time": datetime.now().strftime("%H:%M:%S"),
    })
    await manager.broadcast_activity()
    await manager.broadcast_leaderboard()
    return JSONResponse({"success": True, "file_id": file_id})

@app.delete("/file/{file_id}")
async def delete_file_endpoint(file_id: str, request: Request):
    username = get_request_user(request)
    if not username:
        raise HTTPException(status_code=401)
    if delete_file(file_id, username):
        await manager.broadcast_all({"type": "file_deleted", "file_id": file_id})
        return JSONResponse({"success": True})
    raise HTTPException(status_code=403, detail="Not allowed to delete this file")

@app.get("/file/{file_id}")
async def get_file(file_id: str):
    with closing(get_db()) as conn:
        cur = conn.cursor()
        cur.execute("SELECT * FROM files WHERE id = ?", (file_id,))
        row = cur.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="File not found")
        return Response(
            content=base64.b64decode(row["data_b64"]),
            media_type=row["mime"],
            headers={"Content-Disposition": f'inline; filename="{row["filename"]}"'},
        )

@app.post("/create_poll")
async def create_poll(request: Request, question: str = Form(...), options: str = Form(...)):
    username = get_request_user(request)
    if not username:
        return JSONResponse({"success": False, "message": "Unauthorized"})
    opts = [o.strip() for o in options.split(",") if o.strip()]
    if len(opts) < 2:
        return JSONResponse({"success": False, "message": "At least two options are required."})
    poll_id = str(uuid.uuid4())
    record = {
        "id": poll_id,
        "question": question.strip(),
        "options": opts,
        "votes": {opt: [] for opt in opts},
        "created_by": username,
    }
    with closing(get_db()) as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO polls (id, question, options_json, votes_json, created_by, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (poll_id, record["question"], json.dumps(opts), json.dumps(record["votes"]), username, now_iso()),
        )
        conn.commit()
    update_points(username, 15, f"created poll {question[:30]}")
    award_badge(username, "poll_creator")
    await manager.broadcast_all({"type": "new_poll", "poll": record})
    await manager.broadcast_activity()
    await manager.broadcast_leaderboard()
    return JSONResponse({"success": True, "poll_id": poll_id})

@app.delete("/message/{message_id}")
async def delete_message_endpoint(message_id: str, request: Request):
    username = get_request_user(request)
    if not username:
        raise HTTPException(status_code=401)
    if delete_message(message_id, username):
        await manager.broadcast_all({"type": "message_deleted", "message_id": message_id})
        return JSONResponse({"success": True})
    raise HTTPException(status_code=403, detail="Not allowed to delete this message")

@app.post("/update_bio")
async def update_bio(request: Request, bio: str = Form(...)):
    username = get_request_user(request)
    if not username:
        raise HTTPException(status_code=401)
    update_user_bio(username, bio.strip()[:200])
    return JSONResponse({"success": True})

# ------------------------------------------------------------------
# WebSocket Endpoint
# ------------------------------------------------------------------
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    session_from_cookie = decode_session(websocket.cookies.get("session"))
    token_from_query = decode_session(websocket.query_params.get("token"))
    username = session_from_cookie or token_from_query
    if not username or not get_user(username):
        await websocket.close(code=4401, reason="Unauthorized")
        return

    await manager.connect(websocket, username)
    row = get_user(username)

    try:
        await manager.safe_send(websocket, {
            "type": "init",
            "username": username,
            "microbe": row["microbe"],
            "bio": row["bio"],
            "points": row["points"],
            "badges": json.loads(row["badges_json"] or "[]"),
            "activities": lab_activities[:MAX_ACTIVITIES],
            "files": list_files(),
            "polls": list_polls(),
            "private_messages": list_unread_private_messages(username),
            "unread_private_count": get_unread_private_count(username),
            "leaderboard": list_leaderboard(),
            "server_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "games": list(active_games.keys()),  # optional, can be used for game list
        })
    except Exception as e:
        print(f"Init error: {e}")
        await websocket.close()
        return

    await manager.broadcast_all({
        "type": "system",
        "username": "🔬 Lab System",
        "message": f"{username} entered the lab.",
    }, exclude_username=username)

    try:
        while True:
            try:
                raw = await websocket.receive_text()
            except RuntimeError:
                break
            except WebSocketDisconnect:
                break

            try:
                payload = json.loads(raw)
            except:
                continue

            # Typing indicator
            if "typing" in payload:
                channel = payload.get("channel", "public")
                await manager.set_typing(username, channel, bool(payload["typing"]))
                continue

            # Switch channel
            if payload.get("switch_channel"):
                new_channel = payload.get("channel", "public")
                if new_channel not in ALLOWED_CHANNELS:
                    new_channel = "public"
                user_channel_state[username] = new_channel
                await manager.broadcast_user_list()
                continue

            # Commands
            if "command" in payload:
                cmd = str(payload["command"]).strip().lower()
                if cmd == "random_fact":
                    await manager.broadcast_all({"type": "system", "username": "🧠 Fact Bot", "message": random.choice(FACTS)})
                elif cmd == "random_quote":
                    await manager.broadcast_all({"type": "system", "username": "🧫 Quote Bot", "message": random.choice(QUOTES)})
                elif cmd == "whoami":
                    me = get_user(username)
                    await manager.safe_send(websocket, {
                        "type": "system",
                        "username": "👤 Identity",
                        "message": f"{username} • {me['microbe']} • {me['points']} points",
                    })
                elif cmd == "shake":
                    update_points(username, 8, "shook the flask")
                    award_badge(username, "shake_master")
                    await manager.broadcast_all({"type": "system", "username": "🧪 Lab Event", "message": f"{username} shook the Erlenmeyer flask."})
                    await manager.broadcast_activity()
                    await manager.broadcast_leaderboard()
                elif cmd == "inoculate":
                    update_points(username, 8, "inoculated a plate")
                    award_badge(username, "inoculator")
                    await manager.broadcast_all({"type": "system", "username": "🦠 Lab Event", "message": f"{username} streaked a fresh agar plate."})
                    await manager.broadcast_activity()
                    await manager.broadcast_leaderboard()
                elif cmd == "leaderboard":
                    await manager.safe_send(websocket, {"type": "leaderboard", "leaderboard": list_leaderboard()})
                elif cmd == "mybadges":
                    names = [BADGES.get(b, b) for b in get_badges(username)]
                    await manager.safe_send(websocket, {
                        "type": "system",
                        "username": "🏅 Badges",
                        "message": ", ".join(names) if names else "No badges yet.",
                    })
                elif cmd == "help":
                    help_text = "Commands: /random_fact, /random_quote, /whoami, /shake, /inoculate, /leaderboard, /mybadges"
                    await manager.safe_send(websocket, {"type": "system", "username": "ℹ️ Help", "message": help_text})
                continue

            # Private message
            if payload.get("private_message"):
                target = str(payload.get("target", "")).strip()
                message = str(payload.get("message", "")).strip()
                if not target or not message or not get_user(target):
                    continue
                item = save_private_message(username, target, message)
                update_points(username, 3, f"sent private message to {target}")
                award_badge(username, "secret_sender")
                if target in active_ws:
                    await manager.safe_send(active_ws[target], {
                        "type": "private_message",
                        "from": username,
                        "to": target,
                        "message": message,
                        "time": item["time"],
                        "msg_id": item["id"],
                    })
                await manager.safe_send(websocket, {
                    "type": "private_message",
                    "from": username,
                    "to": target,
                    "message": message,
                    "time": item["time"],
                    "msg_id": item["id"],
                })
                await manager.broadcast_activity()
                await manager.broadcast_leaderboard()
                continue

            # Poll vote
            if payload.get("poll_vote"):
                poll_id = payload.get("poll_id")
                option = payload.get("option")
                with closing(get_db()) as conn:
                    cur = conn.cursor()
                    cur.execute("SELECT * FROM polls WHERE id = ?", (poll_id,))
                    row = cur.fetchone()
                    if row:
                        votes = json.loads(row["votes_json"])
                        already_voted = username in [u for sub in votes.values() for u in sub]
                        if (not already_voted) and option in votes:
                            votes[option].append(username)
                            cur.execute("UPDATE polls SET votes_json = ? WHERE id = ?", (json.dumps(votes), poll_id))
                            conn.commit()
                            update_points(username, 5, "voted in poll")
                            await manager.broadcast_all({
                                "type": "poll_update",
                                "poll_id": poll_id,
                                "results": {k: len(v) for k, v in votes.items()},
                            })
                            await manager.broadcast_activity()
                            await manager.broadcast_leaderboard()
                continue

            # Game actions
            if "game" in payload:
                action = payload["game"]
                if action == "challenge":
                    game_type = payload.get("game_type")
                    opponent = payload.get("opponent")
                    if opponent not in active_ws:
                        await manager.safe_send(websocket, {"type": "error", "message": "User not online."})
                        continue
                    game_id = create_game(game_type, [username, opponent])
                    # Notify opponent
                    await manager.safe_send(active_ws[opponent], {
                        "type": "game_challenge",
                        "from": username,
                        "game_type": game_type,
                        "game_id": game_id,
                    })
                    await manager.safe_send(websocket, {"type": "game_challenge_sent", "game_id": game_id, "opponent": opponent})
                elif action == "accept":
                    game_id = payload["game_id"]
                    game = active_games.get(game_id)
                    if not game or username not in game["players"]:
                        continue
                    # Both players get game start
                    for player in game["players"]:
                        if player in active_ws:
                            await manager.safe_send(active_ws[player], {
                                "type": "game_start",
                                "game_id": game_id,
                                "game_type": game["type"],
                                "state": game["state"],
                                "players": game["players"],
                            })
                elif action == "move":
                    game_id = payload["game_id"]
                    game = active_games.get(game_id)
                    if not game or username not in game["players"]:
                        continue
                    gtype = game["type"]
                    state = game["state"]
                    if gtype == "tictactoe":
                        if state["turn"] != username or state["winner"]:
                            continue
                        idx = payload.get("index")
                        if idx is None or idx < 0 or idx > 8 or state["board"][idx]:
                            continue
                        state["board"][idx] = "X" if username == game["players"][0] else "O"
                        # Check win
                        win_patterns = [[0,1,2],[3,4,5],[6,7,8],[0,3,6],[1,4,7],[2,5,8],[0,4,8],[2,4,6]]
                        for p in win_patterns:
                            if state["board"][p[0]] and state["board"][p[0]] == state["board"][p[1]] == state["board"][p[2]]:
                                state["winner"] = username
                                update_points(username, 20, f"won Tic-Tac-Toe")
                                award_badge(username, "game_master")
                                break
                        if not state["winner"] and "" not in state["board"]:
                            state["winner"] = "draw"
                        state["turn"] = game["players"][1] if state["turn"] == game["players"][0] else game["players"][0]
                        # Broadcast state
                        for player in game["players"]:
                            if player in active_ws:
                                await manager.safe_send(active_ws[player], {
                                    "type": "game_update",
                                    "game_id": game_id,
                                    "state": state,
                                })
                        if state["winner"]:
                            # Clean up game after a while
                            pass
                    elif gtype == "rps":
                        move = payload.get("move")  # rock, paper, scissors
                        if move not in ["rock", "paper", "scissors"]:
                            continue
                        state["moves"][username] = move
                        if len(state["moves"]) == 2:
                            p1, p2 = game["players"]
                            m1, m2 = state["moves"][p1], state["moves"][p2]
                            if m1 == m2:
                                winner = "draw"
                            elif (m1 == "rock" and m2 == "scissors") or (m1 == "paper" and m2 == "rock") or (m1 == "scissors" and m2 == "paper"):
                                winner = p1
                            else:
                                winner = p2
                            state["winner"] = winner
                            if winner != "draw":
                                update_points(winner, 15, "won Rock-Paper-Scissors")
                                award_badge(winner, "game_master")
                            for player in game["players"]:
                                if player in active_ws:
                                    await manager.safe_send(active_ws[player], {
                                        "type": "game_update",
                                        "game_id": game_id,
                                        "state": state,
                                    })
                    elif gtype == "truthdare":
                        choice = payload.get("choice")  # truth or dare
                        prompts = {
                            "truth": ["What is your biggest fear?", "Have you ever lied to a friend?", "What is your most embarrassing moment?"],
                            "dare": ["Do 5 push-ups now!", "Sing a song loudly!", "Send a funny message to the group."],
                        }
                        prompt = random.choice(prompts.get(choice, ["Think of something!"]))
                        for player in game["players"]:
                            if player in active_ws:
                                await manager.safe_send(active_ws[player], {
                                    "type": "game_prompt",
                                    "game_id": game_id,
                                    "prompt": prompt,
                                    "from": username,
                                })
                    elif gtype == "numberguess":
                        guess = payload.get("guess")
                        if guess is None:
                            continue
                        secret = state["secret"]
                        if guess == secret:
                            state["winner"] = username
                            update_points(username, 25, "guessed the number")
                            award_badge(username, "game_master")
                        else:
                            hint = "higher" if guess < secret else "lower"
                            state["guesses"][username] = {"guess": guess, "hint": hint}
                        for player in game["players"]:
                            if player in active_ws:
                                await manager.safe_send(active_ws[player], {
                                    "type": "game_update",
                                    "game_id": game_id,
                                    "state": state,
                                })
                continue

            # Chat message
            if "text" in payload:
                message = str(payload["text"]).strip()
                channel = payload.get("channel", "public")
                if channel not in ALLOWED_CHANNELS:
                    channel = "public"
                if not message:
                    continue
                msg = save_public_message(username, message, channel)
                update_points(username, 1, f"said: {message[:40]}")
                if get_user(username)["points"] == 1:
                    award_badge(username, "first_chat")
                await manager.broadcast_channel(channel, {"type": "chat", **msg})
                await manager.broadcast_activity()
                await manager.broadcast_leaderboard()
                continue

    except WebSocketDisconnect:
        pass
    finally:
        manager.disconnect(username)
        await manager.broadcast_user_list()
        await manager.broadcast_activity()
        await manager.broadcast_all({
            "type": "system",
            "username": "🔬 Lab System",
            "message": f"{username} left the lab.",
        })

# ------------------------------------------------------------------
# HTML Templates (optimized for Android, games added)
# ------------------------------------------------------------------
LOGIN_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=yes, viewport-fit=cover">
    <title>MicroLab Nexus • Enter</title>
    <style>
        :root{
            --bg1:#040b10;
            --bg2:#0a1924;
            --panel:rgba(8,24,32,0.8);
            --line:rgba(100,255,210,0.15);
            --text:#e0f7f0;
            --muted:#88b8b0;
            --brand:#6ee7c7;
            --brand-2:#7da8ff;
            --danger:#ff859a;
            --shadow:0 30px 50px rgba(0,0,0,0.4);
            --glow:0 0 30px rgba(110,231,199,0.15);
        }
        *{box-sizing:border-box}
        body{
            margin:0;
            min-height:100vh;
            font-family:'Inter',-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,sans-serif;
            color:var(--text);
            background:
                radial-gradient(circle at 10% 10%, rgba(110,231,199,0.12), transparent 30%),
                radial-gradient(circle at 90% 20%, rgba(125,168,255,0.1), transparent 30%),
                linear-gradient(145deg, #051016, #0b1c27);
            display:grid;
            place-items:center;
            overflow:hidden;
        }
        .grid{
            position:fixed;
            inset:0;
            background:
                linear-gradient(rgba(255,255,255,0.02) 1px, transparent 1px),
                linear-gradient(90deg, rgba(255,255,255,0.02) 1px, transparent 1px);
            background-size:40px 40px;
            mask-image:radial-gradient(circle at center, black, transparent 80%);
            pointer-events:none;
        }
        .card{
            position:relative;
            width:min(460px, 94vw);
            padding:34px 30px;
            border:1px solid var(--line);
            border-radius:42px;
            background:var(--panel);
            backdrop-filter:blur(20px);
            box-shadow:var(--shadow), var(--glow);
            transition:transform 0.2s ease;
        }
        .badge{
            display:inline-flex;
            align-items:center;
            gap:8px;
            padding:8px 14px;
            border-radius:999px;
            background:rgba(110,231,199,0.08);
            border:1px solid rgba(110,231,199,0.18);
            color:var(--brand);
            font-size:13px;
            font-weight:700;
            letter-spacing:.03em;
            backdrop-filter:blur(4px);
        }
        h1{
            margin:18px 0 8px;
            font-size:44px;
            line-height:1.1;
            font-weight:800;
            background:linear-gradient(135deg, #e0f7f0, #b8f0e0);
            -webkit-background-clip:text;
            background-clip:text;
            color:transparent;
        }
        p{
            margin:0 0 26px;
            color:var(--muted);
            line-height:1.6;
            font-size:16px;
        }
        .field{margin-bottom:18px}
        .field input{
            width:100%;
            padding:18px 20px;
            border-radius:20px;
            border:1px solid rgba(255,255,255,0.06);
            background:rgba(0,0,0,0.25);
            color:var(--text);
            outline:none;
            font-size:16px;
            backdrop-filter:blur(5px);
            transition:border 0.2s, box-shadow 0.2s;
        }
        .field input:focus{
            border-color:var(--brand);
            box-shadow:0 0 0 4px rgba(110,231,199,0.1);
        }
        button{
            width:100%;
            border:none;
            border-radius:20px;
            padding:18px 20px;
            font-weight:700;
            font-size:17px;
            color:#021008;
            cursor:pointer;
            background:linear-gradient(135deg, var(--brand), #a0f5dc);
            transition:all 0.2s ease;
            box-shadow:0 12px 28px rgba(0,0,0,0.3), 0 0 0 0 rgba(110,231,199,0);
        }
        button:hover{
            transform:translateY(-3px);
            box-shadow:0 20px 35px rgba(0,0,0,0.4), 0 0 20px rgba(110,231,199,0.3);
        }
        .msg{
            min-height:28px;
            margin-top:16px;
            color:var(--danger);
            text-align:center;
            font-weight:600;
            font-size:15px;
        }
        .footer{
            margin-top:24px;
            text-align:center;
            color:#6b9e95;
            font-size:13px;
            letter-spacing:0.5px;
            opacity:0.8;
        }
        .dev-credit{
            margin-top:8px;
            font-size:12px;
            color:#4f7e76;
        }
    </style>
</head>
<body>
    <div class="grid"></div>
    <div class="card">
        <div class="badge">🧬 DIRECT ENTRY • NO PASSWORD</div>
        <h1>MicroLab Nexus</h1>
        <p>Choose your lab name. You'll be assigned a random microbe identity. Real‑time collaboration awaits.</p>

        <div id="joinBox">
            <div class="field"><input id="usernameInput" placeholder="Your lab name (min 3 chars)" autofocus></div>
            <button onclick="joinLab()">🔬 Enter the Lab</button>
        </div>

        <div id="msg" class="msg"></div>
        <div class="footer">
            <div>Persistent • Smooth • Mobile‑ready</div>
            <div class="dev-credit">Developed by Nahiduzzman</div>
        </div>
    </div>

<script>
async function joinLab(){
    const username = document.getElementById("usernameInput").value.trim();
    if(username.length < 3){
        document.getElementById("msg").innerText = "Name must be at least 3 characters.";
        return;
    }
    const fd = new FormData();
    fd.append("username", username);
    const res = await fetch("/join", {method:"POST", body:fd});
    const data = await res.json();
    if(data.success){
        window.location.href = "/dashboard";
    } else {
        document.getElementById("msg").innerText = data.message || "Failed to join.";
    }
}
document.addEventListener("keydown", (e) => {
    if(e.key === "Enter") joinLab();
});
</script>
</body>
</html>
"""

CHAT_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=yes, viewport-fit=cover">
    <title>MicroLab Nexus • Dashboard</title>
    <style>
        :root{
            --bg:#040b10;
            --bg-soft:#08161e;
            --panel:rgba(8,24,32,0.75);
            --panel-2:rgba(12,28,38,0.85);
            --line:rgba(100,255,210,0.1);
            --text:#e2f3ef;
            --muted:#86b4aa;
            --brand:#6ee7c7;
            --brand-2:#7da8ff;
            --bubble:#0d222e;
            --bubble-own:#133e39;
            --danger:#ff859a;
            --warning:#ffd37d;
            --success:#9fffe0;
            --shadow:0 20px 40px rgba(0,0,0,0.4);
            --radius:24px;
            --transition:all 0.2s cubic-bezier(0.23,1,0.32,1);
        }
        *{box-sizing:border-box}
        body{
            margin:0;
            font-family:'Inter',-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,sans-serif;
            color:var(--text);
            background:
                radial-gradient(circle at 0% 0%, rgba(110,231,199,0.08), transparent 30%),
                radial-gradient(circle at 100% 20%, rgba(125,168,255,0.08), transparent 30%),
                linear-gradient(145deg, #030b10, #09161e 60%, #07121a);
            height:100vh;
            overflow:hidden;
        }
        .shell{
            display:grid;
            grid-template-columns:300px 1fr 320px;
            gap:16px;
            padding:16px;
            height:100vh;
        }
        .panel{
            background:var(--panel);
            border:1px solid var(--line);
            border-radius:var(--radius);
            box-shadow:var(--shadow);
            backdrop-filter:blur(20px);
            min-height:0;
        }
        .left, .right{
            padding:18px;
            display:flex;
            flex-direction:column;
            gap:16px;
            overflow:hidden;
        }
        .card{
            background:var(--panel-2);
            border:1px solid var(--line);
            border-radius:22px;
            padding:18px;
            backdrop-filter:blur(8px);
            transition:var(--transition);
        }
        .card:hover{
            border-color:rgba(110,231,199,0.2);
        }
        .profile{
            display:flex;
            flex-direction:column;
            gap:14px;
        }
        .profile-top{
            display:flex;
            align-items:center;
            gap:14px;
        }
        .avatar{
            width:60px;
            height:60px;
            border-radius:20px;
            display:grid;
            place-items:center;
            font-size:30px;
            background:linear-gradient(145deg, rgba(110,231,199,0.2), rgba(125,168,255,0.15));
            border:1px solid rgba(255,255,255,0.06);
            box-shadow:0 8px 16px rgba(0,0,0,0.2);
        }
        .muted{color:var(--muted); font-size:14px;}
        .pill{
            display:inline-flex;
            align-items:center;
            gap:8px;
            padding:8px 14px;
            border-radius:999px;
            border:1px solid var(--line);
            background:rgba(255,255,255,0.02);
            font-size:13px;
            font-weight:600;
            color:var(--success);
            backdrop-filter:blur(4px);
        }
        h2,h3,h4,p{margin:0}
        .title-row{
            display:flex;
            align-items:center;
            justify-content:space-between;
            gap:10px;
        }
        .title-row h3{font-size:16px; font-weight:700;}
        .list{
            overflow:auto;
            min-height:0;
            display:flex;
            flex-direction:column;
            gap:10px;
        }
        .member{
            display:flex;
            align-items:center;
            gap:12px;
            padding:14px;
            border-radius:20px;
            border:1px solid transparent;
            background:rgba(255,255,255,0.02);
            cursor:pointer;
            transition:var(--transition);
        }
        .member:hover{
            border-color:rgba(110,231,199,0.25);
            background:rgba(110,231,199,0.04);
            transform:translateY(-2px);
        }
        .dot{
            width:12px;height:12px;border-radius:999px;background:var(--brand);box-shadow:0 0 14px rgba(110,231,199,0.6);
        }
        .main{
            min-width:0;
            display:grid;
            grid-template-rows:auto auto 1fr auto;
            gap:16px;
            padding:16px;
        }
        .hero{
            padding:20px 24px;
            display:flex;
            justify-content:space-between;
            align-items:center;
            gap:16px;
            background:linear-gradient(135deg, rgba(110,231,199,0.1), rgba(125,168,255,0.08));
            border-radius:24px;
            border:1px solid var(--line);
            backdrop-filter:blur(8px);
        }
        .hero h1{
            font-size:32px;
            font-weight:800;
            margin-bottom:6px;
            background:linear-gradient(135deg, #fff, #b8f0e0);
            -webkit-background-clip:text;
            background-clip:text;
            color:transparent;
        }
        .tabs, .subtabs{
            display:flex;
            gap:8px;
            flex-wrap:wrap;
        }
        .tab, .subtab, .icon-btn, .send-btn{
            border:none;
            cursor:pointer;
            border-radius:18px;
            padding:10px 16px;
            font-weight:600;
            color:var(--text);
            background:rgba(255,255,255,0.03);
            border:1px solid transparent;
            transition:var(--transition);
            backdrop-filter:blur(4px);
        }
        .tab.active, .subtab.active{
            background:linear-gradient(135deg, rgba(110,231,199,0.2), rgba(125,168,255,0.15));
            border-color:rgba(110,231,199,0.3);
        }
        .tab:hover, .subtab:hover{
            background:rgba(110,231,199,0.1);
        }
        .workspace{
            overflow:hidden;
            min-height:0;
            display:flex;
            flex-direction:column;
        }
        .chat-wrap{
            min-height:0;
            display:flex;
            flex-direction:column;
            gap:12px;
            height:100%;
        }
        .messages{
            flex:1;
            overflow-y:auto;
            display:flex;
            flex-direction:column;
            gap:6px;  /* reduced distance between messages */
            padding-right:4px;
        }
        .message{
            max-width:min(74%, 780px);
            display:flex;
            flex-direction:column;
            gap:4px;
            position:relative;
            animation:fadeIn 0.2s ease;
        }
        @keyframes fadeIn{
            from{opacity:0; transform:translateY(8px);}
            to{opacity:1; transform:translateY(0);}
        }
        .message.own{align-self:flex-end}
        .message.other{align-self:flex-start}
        .message.system{
            align-self:center;
            max-width:86%;
        }
        .bubble{
            background:var(--bubble);
            border:1px solid var(--line);
            border-radius:24px;
            padding:14px 18px;
            line-height:1.5;
            word-break:break-word;
            box-shadow:0 8px 12px rgba(0,0,0,0.1);
        }
        .own .bubble{
            background:var(--bubble-own);
        }
        .system .bubble{
            background:rgba(255,255,255,0.02);
            border-style:dashed;
        }
        .meta{
            display:flex;
            gap:12px;
            align-items:center;
            flex-wrap:wrap;
            color:var(--muted);
            font-size:12px;
        }
        .delete-btn{
            background:transparent;
            border:none;
            color:var(--danger);
            cursor:pointer;
            font-size:14px;
            opacity:0.7;
            transition:opacity 0.2s;
        }
        .delete-btn:hover{opacity:1}
        .typing{
            min-height:24px;
            color:var(--warning);
            font-size:13px;
            font-style:italic;
        }
        .composer{
            display:grid;
            grid-template-columns:1fr auto auto;
            gap:8px;
            align-items:center;
            padding:8px 0;
        }
        .composer input{
            width:100%;
            padding:14px 16px;
            border-radius:22px;
            border:1px solid rgba(255,255,255,0.06);
            background:rgba(0,0,0,0.25);
            color:var(--text);
            outline:none;
            font-size:15px;
            backdrop-filter:blur(5px);
            transition:var(--transition);
        }
        .composer input:focus{
            border-color:var(--brand);
            box-shadow:0 0 0 3px rgba(110,231,199,0.15);
        }
        .icon-btn, .send-btn{
            min-width:48px;
            height:48px;
            border-radius:20px;
            display:grid;
            place-items:center;
            font-size:20px;
        }
        .send-btn{
            background:linear-gradient(135deg, var(--brand), #a4ffe5);
            color:#021008;
            box-shadow:0 8px 18px rgba(0,0,0,0.3);
        }
        .view{display:none; min-height:0; overflow:auto}
        .view.active{display:block}
        .poll-card, .file-card, .game-card{
            padding:18px;
            border-radius:22px;
            border:1px solid var(--line);
            background:rgba(255,255,255,0.02);
            margin-bottom:14px;
            backdrop-filter:blur(5px);
            transition:var(--transition);
        }
        .poll-card:hover, .file-card:hover, .game-card:hover{
            border-color:rgba(110,231,199,0.2);
        }
        .poll-option, .game-option{
            margin-top:10px;
            width:100%;
            text-align:left;
            border:none;
            padding:14px 16px;
            border-radius:18px;
            background:rgba(255,255,255,0.03);
            color:var(--text);
            cursor:pointer;
            transition:var(--transition);
        }
        .poll-option:hover, .game-option:hover{
            background:rgba(110,231,199,0.1);
        }
        .file-link{
            color:var(--brand);
            text-decoration:none;
            font-weight:700;
        }
        .back-btn{
            background:transparent;
            border:1px solid var(--line);
            color:var(--text);
            padding:10px 20px;
            border-radius:22px;
            margin-bottom:16px;
            cursor:pointer;
            font-weight:600;
            transition:var(--transition);
            align-self:flex-start;
        }
        .back-btn:hover{
            background:rgba(255,255,255,0.04);
        }
        .section-label{
            font-size:12px;
            color:var(--muted);
            text-transform:uppercase;
            letter-spacing:.1em;
            margin-bottom:10px;
        }
        .right .card{min-height:0}
        .feed{
            overflow:auto;
            display:flex;
            flex-direction:column;
            gap:10px;
            max-height:calc(50vh - 60px);
        }
        .activity-item, .leader-item{
            padding:14px 16px;
            border-radius:18px;
            background:rgba(255,255,255,0.02);
            border:1px solid var(--line);
            font-size:13px;
        }
        .grid-two{
            display:grid;
            grid-template-columns:1fr 1fr;
            gap:12px;
        }
        .logout{
            background:rgba(255,133,154,0.12);
            color:#ffd1d8;
            border-color:rgba(255,133,154,0.2);
        }
        .dev-credit{
            margin-top:8px;
            text-align:center;
            font-size:12px;
            color:var(--muted);
            opacity:0.7;
        }
        .notification-badge{
            background:var(--danger);
            color:white;
            border-radius:12px;
            padding:2px 8px;
            font-size:12px;
            margin-left:8px;
        }
        /* Game board styles */
        .board{
            display:grid;
            grid-template-columns:repeat(3,1fr);
            gap:8px;
            margin:16px 0;
        }
        .cell{
            aspect-ratio:1/1;
            background:rgba(255,255,255,0.05);
            border:1px solid var(--line);
            border-radius:16px;
            display:grid;
            place-items:center;
            font-size:32px;
            font-weight:bold;
            cursor:pointer;
        }
        .cell:hover{background:rgba(110,231,199,0.1);}
        @media (max-width: 1240px){
            .shell{grid-template-columns:280px 1fr}
            .right{display:none}
        }
        @media (max-width: 860px){
            .shell{grid-template-columns:1fr; padding:10px}
            .left{display:none}
            .main{padding:10px}
            .hero h1{font-size:28px;}
            .composer{gap:6px;}
        }
    </style>
</head>
<body>
<div class="shell">
    <aside class="panel left">
        <div class="card profile">
            <div class="profile-top">
                <div class="avatar">🧬</div>
                <div>
                    <h2 id="displayName">{{USERNAME}}</h2>
                    <div id="microbeName" class="muted">Loading...</div>
                </div>
            </div>
            <div id="userBioDisplay" class="muted" style="margin-bottom:8px;"></div>
            <div class="grid-two">
                <div class="pill">🧪 <span id="userPoints">0</span> pts</div>
                <div class="pill">🟢 <span id="onlineCount">0</span> online</div>
            </div>
            <button class="tab" onclick="editBio()">✏️ Edit Bio</button>
            <button class="tab logout" onclick="logout()">Logout</button>
        </div>

        <div class="card" style="flex:1; min-height:0;">
            <div class="title-row"><h3>Lab members</h3><span class="muted" id="memberCountSmall"></span></div>
            <div id="usersList" class="list" style="margin-top:12px;"></div>
        </div>
    </aside>

    <main class="panel main" id="mainPanel">
        <div id="normalView">
            <div class="hero">
                <div>
                    <h1>MicroLab Nexus</h1>
                    <div class="muted">Persistent • Real‑time • Games • /help</div>
                </div>
                <div class="pill">🔔 <span id="unreadCounter">0</span> • Server: <span id="serverTime">--</span></div>
            </div>

            <div class="tabs">
                <button class="tab active" data-view="chatView">💬 Chat</button>
                <button class="tab" data-view="pollsView">📊 Polls</button>
                <button class="tab" data-view="filesView">📁 Files</button>
                <button class="tab" data-view="gamesView">🎮 Games</button>
            </div>

            <section class="workspace">
                <div id="chatView" class="view active">
                    <div class="chat-wrap">
                        <div class="subtabs">
                            <button class="subtab active" data-channel="public">🌍 Public</button>
                            <button class="subtab" data-channel="lab">🔬 Lab</button>
                            <button class="subtab" data-channel="random">🎲 Random</button>
                        </div>
                        <div id="chatMessages" class="messages"></div>
                        <div id="typingIndicator" class="typing"></div>
                        <div class="composer">
                            <input id="messageInput" placeholder="Message or /random_fact, /whoami, /shake, /inoculate, /help">
                            <button class="icon-btn" id="fileUploadBtn">📎</button>
                            <button class="send-btn" id="sendBtn">➤</button>
                            <input type="file" id="fileInput" hidden>
                        </div>
                    </div>
                </div>

                <div id="pollsView" class="view">
                    <div class="card" style="margin-bottom:16px;">
                        <div class="section-label">Create poll</div>
                        <div class="grid-two">
                            <input id="pollQuestion" placeholder="Question" style="padding:14px;border-radius:18px;border:1px solid rgba(255,255,255,0.06);background:rgba(0,0,0,0.25);color:var(--text);">
                            <input id="pollOptions" placeholder="Option 1, Option 2, Option 3" style="padding:14px;border-radius:18px;border:1px solid rgba(255,255,255,0.06);background:rgba(0,0,0,0.25);color:var(--text);">
                        </div>
                        <button class="send-btn" id="createPollBtn" style="margin-top:14px; width:100%;">Create poll</button>
                    </div>
                    <div id="pollsContainer"></div>
                </div>

                <div id="filesView" class="view">
                    <div id="allFilesList"></div>
                </div>

                <div id="gamesView" class="view">
                    <div class="card">
                        <div class="section-label">Available Games</div>
                        <div id="gamesList"></div>
                    </div>
                    <div id="activeGamePanel" class="card" style="display:none;"></div>
                </div>
            </section>
            <div class="dev-credit">Developed by Nahiduzzman</div>
        </div>

        <div id="privateView" style="display:none; height:100%; display:flex; flex-direction:column;">
            <button class="back-btn" onclick="closePrivateChat()">← Back to lab</button>
            <div style="flex:1; display:flex; flex-direction:column; min-height:0;">
                <div class="hero" style="margin-bottom:16px;">
                    <div>
                        <h2>Private with <span id="privateTargetName"></span></h2>
                    </div>
                </div>
                <div id="privateMessagesList" class="messages" style="flex:1; overflow:auto;"></div>
                <div class="composer" style="margin-top:16px;">
                    <input id="privateMessageInput" placeholder="Type a private message">
                    <button class="send-btn" id="sendPrivateMsgBtn">Send</button>
                </div>
            </div>
        </div>
    </main>

    <aside class="panel right">
        <div class="card" style="flex:1; min-height:0;">
            <div class="title-row"><h3>Leaderboard</h3></div>
            <div id="leaderboard" class="feed" style="margin-top:12px;"></div>
        </div>
        <div class="card" style="flex:1; min-height:0;">
            <div class="title-row"><h3>Activity</h3></div>
            <div id="labActivity" class="feed" style="margin-top:12px;"></div>
        </div>
    </aside>
</div>

<audio id="notificationSound" src="data:audio/wav;base64,UklGRlwAAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YVoAAACAgICAf39/f39/f3+AgICAf39/f39/f3+AgICAf39/f39/f3+AgICAf39/f39/f3+AgICAf39/f39/f3+AgICAf39/f39/f38=" preload="auto"></audio>

<script>
const username = "{{USERNAME}}";
const sessionToken = "{{SESSION}}";
let ws = null;
let reconnectTimer = null;
let currentChannel = "public";
let currentPrivateTarget = null;
let publicMessages = [];
let privateMsgs = [];
let polls = [];
let soundEnabled = true;
let unreadPrivateCount = 0;
let activeGameId = null;
let gameState = null;

function escapeHtml(s){
    return String(s).replace(/[&<>"]/g, m => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[m]));
}
function fmtTime(value){
    try{ return new Date(value).toLocaleTimeString(); }catch{ return value||""; }
}
function showSystem(user, message){
    const msg = {id: crypto.randomUUID(), username:user, message, timestamp:new Date().toISOString(), system:true};
    renderSingleMessage(msg);
}
function logout(){
    fetch("/logout", {method:"POST"}).finally(() => window.location.href = "/");
}
function connectWS(){
    const protocol = location.protocol === "https:" ? "wss" : "ws";
    ws = new WebSocket(`${protocol}://${location.host}/ws?token=${encodeURIComponent(sessionToken)}`);

    ws.onopen = () => { if(reconnectTimer){ clearTimeout(reconnectTimer); reconnectTimer = null; } };
    ws.onmessage = (event) => { handleMessage(JSON.parse(event.data)); };
    ws.onclose = (event) => {
        if(event.code === 4401){ window.location.href = "/"; return; }
        reconnectTimer = setTimeout(connectWS, 1500);
    };
    ws.onerror = () => { try{ ws.close(); }catch(e){} };
}
function playNotification(){
    if(soundEnabled){
        document.getElementById("notificationSound").play().catch(()=>{});
    }
}
function handleMessage(data){
    switch(data.type){
        case "init":
            document.getElementById("microbeName").innerText = data.microbe || "";
            document.getElementById("userBioDisplay").innerText = data.bio || "";
            document.getElementById("userPoints").innerText = data.points ?? 0;
            document.getElementById("serverTime").innerText = data.server_time || "--";
            unreadPrivateCount = data.unread_private_count || 0;
            updateUnreadBadge();
            privateMsgs = Array.isArray(data.private_messages) ? data.private_messages : [];
            polls = Array.isArray(data.polls) ? data.polls : [];
            renderPolls();
            renderFiles(data.files || []);
            renderActivities(data.activities || []);
            renderLeaderboard(data.leaderboard || []);
            renderGamesList();
            break;
        case "chat":
            publicMessages.push(data);
            if(data.channel === currentChannel) renderSingleMessage(data);
            if(data.username !== username) playNotification();
            break;
        case "system": showSystem(data.username, data.message); break;
        case "user_list":
            renderUsers(data.users || []);
            document.getElementById("onlineCount").innerText = data.count ?? 0;
            document.getElementById("memberCountSmall").innerText = `${data.count ?? 0} online`;
            break;
        case "typing": renderTyping((data.users || []).filter(u => u !== username)); break;
        case "lab_activity": renderActivities(data.activities || []); break;
        case "leaderboard":
            renderLeaderboard(data.leaderboard || []);
            const me = (data.leaderboard || []).find(x => x.username === username);
            if(me) document.getElementById("userPoints").innerText = me.points;
            break;
        case "private_message":
            privateMsgs.push({ id: data.msg_id || crypto.randomUUID(), from: data.from, to: data.to, message: data.message, time: data.time });
            if(currentPrivateTarget === data.from || currentPrivateTarget === data.to) renderPrivateMessages();
            else {
                if(data.to === username) unreadPrivateCount++;
                updateUnreadBadge();
                showSystem("🔔", `New private message from ${data.from}`);
                playNotification();
            }
            break;
        case "new_poll": polls.unshift(data.poll); renderPolls(); break;
        case "poll_update":
            polls = polls.map(p => p.id === data.poll_id ? {...p, votes: Object.fromEntries(Object.entries(data.results).map(([k,v]) => [k, new Array(v).fill("x")]))} : p);
            updatePollResults(data.poll_id, data.results);
            break;
        case "file_upload": appendSingleFile({ id: data.file_id, filename: data.filename, uploader: data.uploader, time: data.time }); break;
        case "file_deleted": document.querySelector(`[data-file-id="${data.file_id}"]`)?.remove(); break;
        case "message_deleted":
            const el = document.querySelector(`[data-message-id="${data.message_id}"]`);
            if(el) el.remove();
            publicMessages = publicMessages.filter(m => m.id !== data.message_id);
            break;
        // Game events
        case "game_challenge":
            if(confirm(`${data.from} challenges you to ${data.game_type}. Accept?`)){
                ws.send(JSON.stringify({game:"accept", game_id:data.game_id}));
            }
            break;
        case "game_challenge_sent":
            showSystem("🎮", `Challenge sent to ${data.opponent}`);
            break;
        case "game_start":
            activeGameId = data.game_id;
            gameState = data.state;
            showGamePanel(data.game_type, data.state, data.players);
            break;
        case "game_update":
            gameState = data.state;
            updateGamePanel(data.game_type, data.state);
            break;
        case "game_prompt":
            alert(`🎲 ${data.prompt}`);
            break;
    }
}
function updateUnreadBadge(){
    document.getElementById("unreadCounter").innerText = unreadPrivateCount;
}
function renderTyping(users){ document.getElementById("typingIndicator").innerText = users.length ? `${users.join(", ")} typing…` : ""; }
function renderSingleMessage(msg){
    const container = document.getElementById("chatMessages");
    if(msg.system){
        const wrap = document.createElement("div");
        wrap.className = "message system";
        wrap.innerHTML = `<div class="bubble"><strong>${escapeHtml(msg.username)}</strong> • ${escapeHtml(msg.message)}</div>`;
        container.appendChild(wrap);
        wrap.scrollIntoView({behavior:"smooth", block:"end"});
        return;
    }
    const mine = msg.username === username;
    const wrap = document.createElement("div");
    wrap.className = `message ${mine ? "own" : "other"}`;
    wrap.dataset.messageId = msg.id;
    wrap.innerHTML = `
        <div class="bubble">
            <div style="font-weight:800; margin-bottom:6px;">${escapeHtml(msg.username)}</div>
            <div class="msg-text">${escapeHtml(msg.message)}${msg.edited?' <span class="muted">(edited)</span>':''}</div>
        </div>
        <div class="meta">
            <span>${fmtTime(msg.timestamp)}</span>
            ${mine ? `<button class="delete-btn" onclick="deleteMessage('${msg.id}')">🗑️</button>` : ''}
        </div>
    `;
    container.appendChild(wrap);
    wrap.scrollIntoView({behavior:"smooth", block:"end"});
}
function renderMessages(){
    const container = document.getElementById("chatMessages");
    container.innerHTML = "";
    publicMessages.filter(m => m.channel === currentChannel).forEach(renderSingleMessage);
}
async function deleteMessage(msgId){
    if(!confirm("Delete this message?")) return;
    const res = await fetch(`/message/${msgId}`, {method:"DELETE"});
    if(!res.ok) alert("Could not delete message.");
}
function sendMessage(){
    const input = document.getElementById("messageInput");
    const text = input.value.trim();
    if(!text || !ws || ws.readyState !== WebSocket.OPEN) return;
    if(text.startsWith("/")){ ws.send(JSON.stringify({command: text.slice(1)})); }
    else{ ws.send(JSON.stringify({text, channel: currentChannel})); }
    input.value = "";
    ws.send(JSON.stringify({typing:false, channel: currentChannel}));
}
function renderUsers(users){
    const box = document.getElementById("usersList");
    box.innerHTML = "";
    users.forEach(u => {
        const div = document.createElement("div");
        div.className = "member";
        div.innerHTML = `<div class="dot"></div><div style="min-width:0;"><div style="font-weight:700;">${escapeHtml(u.username)}</div><div class="muted">${escapeHtml(u.channel||"public")} • ${u.points} pts</div></div><div style="margin-left:auto;" class="pill">chat</div>`;
        div.onclick = () => openPrivateChat(u.username);
        box.appendChild(div);
    });
}
function renderActivities(items){
    const box = document.getElementById("labActivity");
    box.innerHTML = "";
    items.forEach(a => {
        const div = document.createElement("div");
        div.className = "activity-item";
        div.innerHTML = `<strong>${escapeHtml(a.user)}</strong> ${escapeHtml(a.action)}<div class="muted">${escapeHtml(a.time||"")}${a.points?` • +${a.points}`:""}</div>`;
        box.appendChild(div);
    });
}
function renderLeaderboard(items){
    const box = document.getElementById("leaderboard");
    box.innerHTML = "";
    items.forEach((u,i) => {
        const div = document.createElement("div");
        div.className = "leader-item";
        div.innerHTML = `<strong>#${i+1} ${escapeHtml(u.username)}</strong><div class="muted">${u.points} points</div>`;
        box.appendChild(div);
    });
}
function renderFiles(files){
    const box = document.getElementById("allFilesList");
    box.innerHTML = "";
    files.forEach(f => appendSingleFile(f, false));
}
function appendSingleFile(f, prepend=true){
    const box = document.getElementById("allFilesList");
    const div = document.createElement("div");
    div.className = "file-card";
    div.dataset.fileId = f.id;
    div.innerHTML = `
        <div style="display:flex; justify-content:space-between;">
            <a class="file-link" target="_blank" href="/file/${f.id}">${escapeHtml(f.filename)}</a>
            ${f.uploader === username ? `<button class="delete-btn" onclick="deleteFile('${f.id}')">🗑️</button>` : ''}
        </div>
        <div class="muted">Uploaded by ${escapeHtml(f.uploader||"")}${f.time?` • ${escapeHtml(f.time)}`:""}</div>
    `;
    if(prepend && box.firstChild) box.insertBefore(div, box.firstChild);
    else box.appendChild(div);
}
async function deleteFile(fileId){
    if(!confirm("Delete this file?")) return;
    const res = await fetch(`/file/${fileId}`, {method:"DELETE"});
    if(!res.ok) alert("Could not delete file.");
}
function renderPolls(){
    const box = document.getElementById("pollsContainer");
    box.innerHTML = "";
    polls.forEach(poll => {
        const div = document.createElement("div");
        div.className = "poll-card";
        div.id = `poll-${poll.id}`;
        const results = poll.votes || {};
        div.innerHTML = `<div style="font-weight:800;">${escapeHtml(poll.question)}</div><div class="muted">by ${escapeHtml(poll.created_by||"")}</div><div id="poll-options-${poll.id}"></div>`;
        const target = div.querySelector(`#poll-options-${poll.id}`);
        (poll.options || []).forEach(opt => {
            const count = Array.isArray(results[opt]) ? results[opt].length : 0;
            const btn = document.createElement("button");
            btn.className = "poll-option";
            btn.innerText = `${opt}${count ? ` (${count})` : ""}`;
            btn.onclick = () => { if(ws) ws.send(JSON.stringify({poll_vote:true, poll_id:poll.id, option:opt})); };
            target.appendChild(btn);
        });
        box.appendChild(div);
    });
}
function updatePollResults(pollId, results){
    const target = document.getElementById(`poll-options-${pollId}`);
    if(!target) return;
    const existing = [...target.children].map(x => x.innerText.split(" (")[0]);
    target.innerHTML = "";
    existing.forEach(opt => {
        const btn = document.createElement("button");
        btn.className = "poll-option";
        btn.innerText = `${opt} (${results[opt] || 0})`;
        btn.onclick = () => { if(ws) ws.send(JSON.stringify({poll_vote:true, poll_id:pollId, option:opt})); };
        target.appendChild(btn);
    });
}
function openPrivateChat(target){
    if(target === username){ showSystem("System", "You cannot chat with yourself."); return; }
    currentPrivateTarget = target;
    document.getElementById("privateTargetName").innerText = target;
    document.getElementById("normalView").style.display = "none";
    document.getElementById("privateView").style.display = "flex";
    renderPrivateMessages();
}
function closePrivateChat(){
    currentPrivateTarget = null;
    document.getElementById("normalView").style.display = "block";
    document.getElementById("privateView").style.display = "none";
    unreadPrivateCount = 0;
    updateUnreadBadge();
}
function renderPrivateMessages(){
    const box = document.getElementById("privateMessagesList");
    box.innerHTML = "";
    const relevant = privateMsgs.filter(m => currentPrivateTarget && ((m.from === currentPrivateTarget && m.to === username) || (m.from === username && m.to === currentPrivateTarget)));
    relevant.forEach(m => {
        const div = document.createElement("div");
        div.className = `message ${m.from === username ? "own" : "other"}`;
        div.innerHTML = `<div class="bubble"><div style="font-weight:800;">${escapeHtml(m.from)}</div><div>${escapeHtml(m.message)}</div></div><div class="meta">${escapeHtml(m.time||"")}</div>`;
        box.appendChild(div);
    });
}
function sendPrivateMessage(){
    const input = document.getElementById("privateMessageInput");
    const message = input.value.trim();
    if(!message || !currentPrivateTarget || !ws) return;
    ws.send(JSON.stringify({private_message:true, target:currentPrivateTarget, message}));
    input.value = "";
}
async function editBio(){
    const newBio = prompt("Enter your bio (max 200 chars):", document.getElementById("userBioDisplay").innerText);
    if(newBio === null) return;
    const fd = new FormData();
    fd.append("bio", newBio);
    const res = await fetch("/update_bio", {method:"POST", body:fd});
    if(res.ok){
        document.getElementById("userBioDisplay").innerText = newBio;
    } else {
        alert("Could not update bio.");
    }
}
function toggleSound(){
    soundEnabled = !soundEnabled;
    showSystem("🔊", `Sound ${soundEnabled?'enabled':'disabled'}`);
}

// ---------- Games ----------
function renderGamesList(){
    const container = document.getElementById("gamesList");
    container.innerHTML = `
        <div class="game-card">
            <h4>🎮 Tic-Tac-Toe</h4>
            <p class="muted">Challenge another player</p>
            <button class="game-option" onclick="startGame('tictactoe')">Play</button>
        </div>
        <div class="game-card">
            <h4>✊ Rock-Paper-Scissors</h4>
            <p class="muted">Quick match</p>
            <button class="game-option" onclick="startGame('rps')">Play</button>
        </div>
        <div class="game-card">
            <h4>🎲 Truth or Dare</h4>
            <p class="muted">Random prompts</p>
            <button class="game-option" onclick="startGame('truthdare')">Play</button>
        </div>
        <div class="game-card">
            <h4>🔢 Number Guess</h4>
            <p class="muted">Guess 1-100</p>
            <button class="game-option" onclick="startGame('numberguess')">Play</button>
        </div>
    `;
}
function startGame(gameType){
    const users = Array.from(document.querySelectorAll('#usersList .member')).map(el => {
        const name = el.querySelector('div[style*="font-weight:700;"]')?.innerText;
        return name;
    }).filter(u => u && u !== username);
    if(users.length === 0){
        alert("No other users online to challenge.");
        return;
    }
    const opponent = prompt(`Choose opponent:\n${users.join('\n')}`);
    if(!opponent || !users.includes(opponent)) return;
    ws.send(JSON.stringify({game:"challenge", game_type:gameType, opponent}));
}
function showGamePanel(type, state, players){
    const panel = document.getElementById("activeGamePanel");
    panel.style.display = "block";
    panel.innerHTML = `<h3>${type.toUpperCase()}</h3>`;
    if(type === "tictactoe"){
        panel.innerHTML += `<div class="board" id="tttBoard"></div><div id="gameStatus"></div>`;
        renderTicTacToe(state, players);
    } else if(type === "rps"){
        panel.innerHTML += `<div><button class="game-option" onclick="sendGameMove('rock')">🪨 Rock</button><button class="game-option" onclick="sendGameMove('paper')">📄 Paper</button><button class="game-option" onclick="sendGameMove('scissors')">✂️ Scissors</button></div><div id="gameStatus"></div>`;
        updateGamePanel(type, state);
    } else if(type === "truthdare"){
        panel.innerHTML += `<div><button class="game-option" onclick="sendGameMove('truth')">Truth</button><button class="game-option" onclick="sendGameMove('dare')">Dare</button></div>`;
    } else if(type === "numberguess"){
        panel.innerHTML += `<div><input id="guessInput" type="number" min="1" max="100"><button class="game-option" onclick="submitGuess()">Guess</button></div><div id="guessFeedback"></div>`;
    }
}
function renderTicTacToe(state, players){
    const boardDiv = document.getElementById("tttBoard");
    if(!boardDiv) return;
    boardDiv.innerHTML = "";
    state.board.forEach((cell, i) => {
        const cellDiv = document.createElement("div");
        cellDiv.className = "cell";
        cellDiv.innerText = cell;
        cellDiv.onclick = () => { if(state.turn === username && !state.winner) ws.send(JSON.stringify({game:"move", game_id:activeGameId, index:i})); };
        boardDiv.appendChild(cellDiv);
    });
    const status = document.getElementById("gameStatus");
    if(state.winner){
        status.innerText = state.winner === "draw" ? "It's a draw!" : `${state.winner} wins!`;
    } else {
        status.innerText = `Turn: ${state.turn}`;
    }
}
function updateGamePanel(type, state){
    if(type === "tictactoe"){
        renderTicTacToe(state);
    } else if(type === "rps"){
        const status = document.getElementById("gameStatus");
        if(state.winner){
            status.innerText = state.winner === "draw" ? "Draw!" : `${state.winner} wins!`;
        } else {
            status.innerText = `Waiting for moves...`;
        }
    } else if(type === "numberguess"){
        const fb = document.getElementById("guessFeedback");
        const guesses = state.guesses[username];
        if(guesses){
            fb.innerText = `Your guess ${guesses.guess}: ${guesses.hint}`;
        }
        if(state.winner){
            fb.innerText = `${state.winner} guessed the number! It was ${state.secret}.`;
        }
    }
}
function sendGameMove(move){
    ws.send(JSON.stringify({game:"move", game_id:activeGameId, move, choice:move}));
}
function submitGuess(){
    const inp = document.getElementById("guessInput");
    const val = parseInt(inp.value);
    if(isNaN(val)) return;
    ws.send(JSON.stringify({game:"move", game_id:activeGameId, guess:val}));
    inp.value = "";
}

// Event listeners
document.getElementById("sendPrivateMsgBtn").onclick = sendPrivateMessage;
document.getElementById("privateMessageInput").addEventListener("keydown", e => { if(e.key==="Enter") sendPrivateMessage(); });
document.getElementById("sendBtn").onclick = sendMessage;
document.getElementById("messageInput").addEventListener("keydown", e => { if(e.key==="Enter") sendMessage(); });
let typingTimer = null;
document.getElementById("messageInput").addEventListener("input", () => {
    if(ws && ws.readyState === WebSocket.OPEN){
        ws.send(JSON.stringify({typing:true, channel:currentChannel}));
        clearTimeout(typingTimer);
        typingTimer = setTimeout(() => { if(ws) ws.send(JSON.stringify({typing:false, channel:currentChannel})); }, 1200);
    }
});
document.querySelectorAll(".tab[data-view]").forEach(btn => {
    btn.addEventListener("click", () => {
        document.querySelectorAll(".tab[data-view]").forEach(b => b.classList.remove("active"));
        btn.classList.add("active");
        document.querySelectorAll(".view").forEach(v => v.classList.remove("active"));
        document.getElementById(btn.dataset.view).classList.add("active");
    });
});
document.querySelectorAll(".subtab[data-channel]").forEach(btn => {
    btn.addEventListener("click", () => {
        document.querySelectorAll(".subtab[data-channel]").forEach(b => b.classList.remove("active"));
        btn.classList.add("active");
        currentChannel = btn.dataset.channel;
        renderMessages();
        if(ws) ws.send(JSON.stringify({switch_channel:true, channel:currentChannel}));
    });
});
document.getElementById("createPollBtn").onclick = async () => {
    const fd = new FormData();
    fd.append("question", document.getElementById("pollQuestion").value.trim());
    fd.append("options", document.getElementById("pollOptions").value.trim());
    const res = await fetch("/create_poll", {method:"POST", body:fd});
    const data = await res.json();
    if(!data.success) showSystem("Poll", data.message || "Unable to create poll");
    else{ document.getElementById("pollQuestion").value = ""; document.getElementById("pollOptions").value = ""; }
};
document.getElementById("fileUploadBtn").onclick = () => document.getElementById("fileInput").click();
document.getElementById("fileInput").addEventListener("change", async (e) => {
    const file = e.target.files[0];
    if(!file) return;
    const fd = new FormData();
    fd.append("file", file);
    const res = await fetch("/upload", {method:"POST", body:fd});
    const data = await res.json();
    if(!data.success) showSystem("Upload", "Upload failed");
    e.target.value = "";
});
const heroDiv = document.querySelector('.hero .pill');
if(heroDiv){
    const soundBtn = document.createElement('button');
    soundBtn.className = 'tab';
    soundBtn.style.marginLeft = '8px';
    soundBtn.innerHTML = '🔊';
    soundBtn.onclick = toggleSound;
    heroDiv.parentNode.appendChild(soundBtn);
}
connectWS();
</script>
</body>
</html>
"""

if __name__ == "__main__":
    uvicorn.run("microlab_nexus:app", host="0.0.0.0", port=8000, reload=True)