#!/usr/bin/env python3  # -*- coding: utf-8 -*-  
"""  
PM Compass – Flask アプリ（修正版：Cosmos優先マージ保存 + current_session_id安定化 + 初回は必ず新規チャット起動）  
- 画像/PDFアップロード対応（input_image / input_file）  
- セッション終了時の自動削除（sendBeacon + /cleanup_session_files）  
- Azure Cognitive Search ハイブリッド検索（RRF融合）  
- Azure OpenAI Responses API（ストリーミング対応）  
- PDF は file_data(Base64) + filename 方式で添付（file_url は使用しない）  
- アプリ側のステップ制御を廃止、LLM判断に委譲  
  
追加修正（最後のメッセージしか残らない対策）  
- ensure_messages_from_cosmos: セッションが途切れても Cosmos から履歴復元  
- merge_messages: 既存履歴と新履歴を重複除去しながらマージ  
- persist_assistant_message: Cosmosをソースに読み戻してから append→upsert→セッション/サイドバー反映  
- send_message/stream_message 開始時に Cosmos 復元  
- JSONフォールバックのレスポンスに session_id を同梱  
- フロント: SSE final 受信時に #currentSession の data-session-id を更新  
  
追加修正（アプリ起動時は常に新規チャット）  
- index 初回 GET（セッション未初期化）で必ず start_new_chat() を実行し、前回のチャットを自動で開かない  
"""  
  
import os  
import re  
import io  
import json  
import uuid  
import base64  
import threading  
import datetime  
import traceback  
import time  
import queue  
from urllib.parse import quote, unquote, urlparse  
  
from flask import (  
    Flask,  
    request,  
    render_template,  
    redirect,  
    url_for,  
    session,  
    flash,  
    send_file,  
    stream_with_context,  
    jsonify,  
)  
from flask import copy_current_request_context  
from flask_session import Session  
from werkzeug.utils import secure_filename  
  
import certifi  
from azure.core.credentials import AzureKeyCredential  
from azure.core.pipeline.transport import RequestsTransport  
from azure.core.pipeline.policies import RetryPolicy  
from azure.search.documents import SearchClient  
from azure.cosmos import CosmosClient  
from azure.storage.blob import BlobServiceClient, generate_blob_sas, BlobSasPermissions  
from openai import AzureOpenAI  
import markdown2  
  
# RAGAS（任意依存）  
try:  
    from datasets import Dataset  
    from ragas import evaluate  
    from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_relevancy  
    from ragas.integrations.langchain import LangchainLLMWrapper, LangchainEmbeddingsWrapper  
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings  
    RAGAS_AVAILABLE = True  
except Exception:  
    RAGAS_AVAILABLE = False  
  
# ------------------------------- Azure OpenAI クライアント設定 -------------------------------  
client = AzureOpenAI(  
    api_key=os.getenv("AZURE_OPENAI_KEY"),  
    base_url=os.getenv("AZURE_OPENAI_ENDPOINT"),  
    api_version="preview",  
    default_headers={"x-ms-include-response-reasoning-summary": "true"},  
)  
  
embed_endpoint = os.getenv("AZURE_OPENAI_EMBED_ENDPOINT") or os.getenv("AZURE_OPENAI_ENDPOINT")  
embed_client = AzureOpenAI(  
    api_key=os.getenv("AZURE_OPENAI_KEY"),  
    azure_endpoint=embed_endpoint,  
    api_version="2025-04-01-preview",  
)  
  
# ------------------------------- モデル・検索設定 -------------------------------  
RESPONSES_MODEL = os.getenv("AZURE_OPENAI_RESPONSES_MODEL", "gpt-4o")  
REASONING_ENABLED_MODELS = set(  
    m.strip() for m in os.getenv("REASONING_ENABLED_MODELS", "o3,o4-mini,gpt-5,o4").split(",") if m.strip()  
)  
REASONING_EFFORT = os.getenv("REASONING_EFFORT", "high")  
  
EMBEDDING_MODEL = os.getenv("AZURE_OPENAI_EMBED_DEPLOYMENT", "text-embedding-3-large")  
VECTOR_FIELD = "contentVector"  
  
MAX_HISTORY_TO_SEND = int(os.getenv("MAX_HISTORY_TO_SEND", "50"))  
DEFAULT_DOC_COUNT = int(os.getenv("DEFAULT_DOC_COUNT", "10"))  
  
RRF_FETCH_MULTIPLIER = float(os.getenv("RRF_FETCH_MULTIPLIER", "3.0"))  
RRF_FETCH_MAX_TOP = int(os.getenv("RRF_FETCH_MAX_TOP", "300"))  
  
MAX_CHUNKS_PER_PARENT = int(os.getenv("MAX_CHUNKS_PER_PARENT", "0"))  
REWRITE_MODEL = "gpt-4o"  
MAX_REWRITE_TURNS = max(1, min(8, int(os.getenv("MAX_REWRITE_TURNS", "4"))))  
  
ENABLE_HYDE = os.getenv("ENABLE_HYDE", "1") not in ("0", "false", "False")  
ENABLE_PRF = os.getenv("ENABLE_PRF", "1") not in ("0", "false", "False")  
RECALL_PARENT_THRESHOLD_FRACTION = float(os.getenv("RECALL_PARENT_THRESHOLD_FRACTION", "0.4"))  
MIN_UNIQUE_PARENTS_ABS = int(os.getenv("MIN_UNIQUE_PARENTS_ABS", "3"))  
  
# 添付の最大バイト（超過時は添付スキップ）  
MAX_ATTACHMENT_BYTES = int(os.getenv("MAX_ATTACHMENT_BYTES", str(5 * 1024 * 1024)))  # 5MB  
  
# RAGAS フラグ  
ENABLE_RAGAS = os.getenv("ENABLE_RAGAS", "1") not in ("0", "false", "False")  
  
# ------------------------------- Flask アプリ設定 -------------------------------  
app = Flask(__name__)  
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'your-default-secret-key')  
app.config['SESSION_TYPE'] = 'filesystem'  
app.config['SESSION_FILE_DIR'] = '/tmp/flask_session'  
app.config['SESSION_PERMANENT'] = False  
app.config['MAX_CONTENT_LENGTH'] = int(os.getenv("MAX_UPLOAD_MB", "100")) * 1024 * 1024  # 100MB  
os.makedirs(app.config['SESSION_FILE_DIR'], exist_ok=True)  
Session(app)  
  
# ------------------------------- 共通 RequestsTransport/Retry 設定 -------------------------------  
def _build_requests_transport():  
    proxies = {}  
    http_proxy = os.getenv("HTTP_PROXY") or os.getenv("http_proxy")  
    https_proxy = os.getenv("HTTPS_PROXY") or os.getenv("https_proxy")  
    if http_proxy:  
        proxies["http"] = http_proxy  
    if https_proxy:  
        proxies["https"] = https_proxy  
    verify_path = os.getenv("REQUESTS_CA_BUNDLE") or os.getenv("AZURE_CA_BUNDLE") or certifi.where()  
    conn_timeout = int(os.getenv("AZURE_HTTP_CONN_TIMEOUT", "20"))  
    read_timeout = int(os.getenv("AZURE_HTTP_READ_TIMEOUT", "120"))  
    return RequestsTransport(  
        connection_verify=verify_path,  
        proxies=proxies or None,  
        connection_timeout=conn_timeout,  
        read_timeout=read_timeout,  
    )  
  
transport = _build_requests_transport()  
retry_policy = RetryPolicy(  
    retry_total=int(os.getenv("AZURE_RETRY_TOTAL", "5")),  
    retry_connect=3,  
    retry_read=3,  
    retry_status=3,  
    retry_backoff_factor=float(os.getenv("AZURE_RETRY_BACKOFF", "0.8")),  
)  
  
# ------------------------------- Azure サービスクライアント -------------------------------  
search_service_endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")  
search_service_key = os.getenv("AZURE_SEARCH_KEY")  
  
cosmos_endpoint = os.getenv("AZURE_COSMOS_ENDPOINT")  
cosmos_key = os.getenv("AZURE_COSMOS_KEY")  
database_name = 'chatdb'  
container_name = 'personalchats'  
cosmos_client = CosmosClient(cosmos_endpoint, credential=cosmos_key) if cosmos_endpoint and cosmos_key else None  
if cosmos_client:  
    database = cosmos_client.get_database_client(database_name)  
    container = database.get_container_client(container_name)  
else:  
    database = None  
    container = None  
  
blob_connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")  
blob_service_client = BlobServiceClient.from_connection_string(  
    blob_connection_string,  
    transport=transport) if blob_connection_string else None  
  
image_container_name = 'chatgpt-image'  
file_container_name = 'chatgpt-files'  
  
# コンテナ自動作成  
image_container_client = None  
file_container_client = None  
if blob_service_client:  
    try:  
        blob_service_client.create_container(image_container_name)  
    except Exception:  
        pass  
    try:  
        blob_service_client.create_container(file_container_name)  
    except Exception:  
        pass  
    image_container_client = blob_service_client.get_container_client(image_container_name)  
    file_container_client = blob_service_client.get_container_client(file_container_name)  
  
INDEX_OPTIONS = [  
    ("通常データ", "filetest11-large"),  
    ("SANUQIメール", "filetest13"),  
    ("L8データ", "filetest14"),  
    ("L8＋製膜データ", "filetest15"),  
    ("予備１", "filetest16"),  
    ("予備２", "filetest17"),  
    ("予備３", "filetest18"),  
    ("品質保証", "quality-assurance"),  
]  
INDEX_VALUES = {v for (_, v) in INDEX_OPTIONS}  
DEFAULT_SEARCH_INDEX = INDEX_OPTIONS[0][1]  
  
INDEX_TO_BLOB_CONTAINER = {  
    "filetest11-large": "filetest11",  
    "filetest13": "filetest13",  
    "filetest14": "filetest14",  
    "filetest15": "filetest15",  
    "filetest16": "filetest16",  
    "filetest17": "filetest17",  
    "filetest18": "filetest18",  
    "quality-assurance": "quality-assurance",  
}  
DEFAULT_BLOB_CONTAINER_FOR_SEARCH = INDEX_TO_BLOB_CONTAINER.get(DEFAULT_SEARCH_INDEX, "filetest11")  
  
INDEX_LANG = {  
    "filetest11-large": "ja",  
    "filetest13": "ja",  
    "filetest14": "ja",  
    "filetest15": "ja",  
    "filetest16": "ja",  
    "filetest17": "ja",  
    "filetest18": "ja",  
    "quality-assurance": "ja",  
}  
  
def to_query_language(lang: str) -> str:  
    return "ja-JP" if (lang or "").lower().startswith("ja") else "en-US"  
  
lock = threading.Lock()  
  
# ------------------------------- ユーティリティ -------------------------------  
def extract_account_key(connection_string: str) -> str:  
    if not connection_string:  
        return ""  
    pairs = [s.split("=", 1) for s in connection_string.split(";") if "=" in s]  
    return dict(pairs).get("AccountKey")  
  
def generate_sas_url(blob_client, blob_name):  
    connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")  
    account_key = extract_account_key(connection_string)  
    start = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(minutes=5)  
    expiry = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=1)  
    sas_token = generate_blob_sas(  
        account_name=blob_client.account_name,  
        container_name=blob_client.container_name,  
        blob_name=blob_name,  
        account_key=account_key,  
        permission=BlobSasPermissions(read=True),  
        expiry=expiry,  
        start=start  
    )  
    return f"{blob_client.url}?{sas_token}"  
  
def make_blob_url(container_name: str, blobname: str) -> str:  
    if not blob_service_client:  
        return ""  
    bc = blob_service_client.get_blob_client(container=container_name, blob=blobname)  
    try:  
        return generate_sas_url(bc, blobname)  
    except Exception:  
        return bc.url  
  
def encode_image_from_blob(blob_client):  
    downloader = blob_client.download_blob()  
    image_bytes = downloader.readall()  
    return base64.b64encode(image_bytes).decode('utf-8')  
  
def encode_pdf_from_blob(blob_client):  
    downloader = blob_client.download_blob()  
    pdf_bytes = downloader.readall()  
    return base64.b64encode(pdf_bytes).decode('utf-8')  
  
def strip_html_tags(html: str) -> str:  
    if not html:  
        return ""  
    html = re.sub(r'<br\s*/?>', '\n', html, flags=re.I)  
    html = re.sub(r'</p\s*>', '\n', html, flags=re.I)  
    text = re.sub(r'<[^>]+>', '', html)  
    return text  
  
def compute_first_assistant_title(messages, limit=30) -> str:  
    try:  
        for m in (messages or []):  
            if m.get("role") == "assistant":  
                t = (m.get("text") or strip_html_tags(m.get("content", ""))).strip()  
                if t:  
                    return t[:limit]  
        for m in (messages or []):  
            if m.get("role") == "user":  
                t = (m.get("content") or "").strip()  
                if t:  
                    return t[:limit]  
    except Exception:  
        pass  
    return ""  
  
def extract_reasoning_summary(resp) -> str:  
    try:  
        for out in getattr(resp, "output", []):  
            if getattr(out, "type", "") == "reasoning":  
                summary_items = getattr(out, "summary", None)  
                if summary_items:  
                    for s in summary_items:  
                        text = getattr(s, "text", None)  
                        if text:  
                            return text  
            if isinstance(out, dict) and out.get("type") == "reasoning":  
                for s in out.get("summary", []):  
                    if isinstance(s, dict) and s.get("text"):  
                        return s["text"]  
        reasoning = getattr(resp, "reasoning", None)  
        if reasoning:  
            summ = getattr(reasoning, "summary", None)  
            if isinstance(summ, str):  
                return summ  
            if hasattr(summ, "text"):  
                return summ.text  
            if isinstance(summ, dict) and "text" in summ:  
                return summ["text"]  
    except Exception as e:  
        print("Reasoning summary 取得エラー:", e)  
    return ""  
  
def extract_output_text(resp) -> str:  
    try:  
        if hasattr(resp, "output_text") and resp.output_text:  
            return resp.output_text  
        if isinstance(resp, dict) and resp.get("output_text"):  
            return resp["output_text"]  
        text_parts = []  
        out_arr = getattr(resp, "output", None)  
        if out_arr is None and isinstance(resp, dict):  
            out_arr = resp.get("output", [])  
        if out_arr:  
            for out in out_arr:  
                otype = getattr(out, "type", None) or (out.get("type") if isinstance(out, dict) else None)  
                if otype == "message":  
                    contents = getattr(out, "content", None) or (out.get("content") if isinstance(out, dict) else [])  
                    for c in contents or []:  
                        ctype = getattr(c, "type", None) or (c.get("type") if isinstance(c, dict) else None)  
                        if ctype in ("output_text", "text"):  
                            t = getattr(c, "text", None) or (c.get("text") if isinstance(c, dict) else None)  
                            if t:  
                                text_parts.append(t)  
        return "".join(text_parts)  
    except Exception as e:  
        print("output_text 抽出エラー:", e)  
        traceback.print_exc()  
        return ""  
  
def is_azure_blob_url(u: str) -> bool:  
    if not u:  
        return False  
    try:  
        p = urlparse(u)  
        if p.scheme not in ("http", "https"):  
            return False  
        host = p.netloc.lower()  
        return (".blob.core.windows.net" in host) or host.startswith("127.0.0.1") or host.startswith("localhost")  
    except Exception:  
        return False  
  
def resolve_blob_from_filepath(selected_index: str, path_or_url: str):  
    if not path_or_url:  
        return None, None  
    s = path_or_url.strip().replace("\\", "/")  
    if s.startswith("http://") or s.startswith("https://"):  
        if not is_azure_blob_url(s):  
            return None, None  
        try:  
            u = urlparse(s)  
            parts = u.path.lstrip("/").split("/", 1)  
            if len(parts) == 2:  
                return parts[0], parts[1]  
        except Exception:  
            return None, None  
    container = INDEX_TO_BLOB_CONTAINER.get(selected_index, DEFAULT_BLOB_CONTAINER_FOR_SEARCH)  
    blobname = s.lstrip("/")  
    if container and blobname.startswith(container + "/"):  
        blobname = blobname[len(container) + 1:]  
    return container, blobname  
  
# ------------------------------- クエリリライト -------------------------------  
def rewrite_queries_for_search_responses(messages: list, current_prompt: str, system_message: str = "") -> list:  
    max_query_history_turns = min(3, MAX_REWRITE_TURNS)  
  
    filtered = [m for m in (messages or []) if m.get("role") in ("user", "assistant")]  
  
    def _to_plain(m):  
        if m.get("role") == "assistant":  
            return m.get("text") or strip_html_tags(m.get("content", "")) or ""  
        return m.get("content", "") or ""  
    recent = filtered[-(max_query_history_turns * 3):]  
    history_lines = []  
    if system_message:  
        history_lines.append(f"system:{system_message}")  
    for m in recent:  
        history_lines.append(f"{m.get('role')}:{_to_plain(m)}")  
    context = "\n".join(history_lines)  
  
    system_prompt = (  
        "あなたは社内文書検索クエリ生成AIです。\n"  
        "出力仕様: {\"queries\":[\"...\", \"...\"]} の JSON を 1 行だけ返す。\n"  
        "・各クエリは30文字以内、日本語主体、必要なら英語同義語併記。\n"  
        "・最大10件生成。\n"  
        "・説明・前後の余分な文字・改行は禁止。"  
    )  
    user_prompt = (  
        f"history{{{context}}}\n"  
        f"endtask ユーザの意図を最もよく表す検索クエリを最大10件生成してください。\n"  
        f"最新ユーザ質問: {current_prompt}"  
    )  
  
    input_items = [  
        {"role": "system", "content": [{"type": "input_text", "text": system_prompt}]},  
        {"role": "user", "content": [{"type": "input_text", "text": user_prompt}]},  
    ]  
    resp = client.responses.create(  
        model=REWRITE_MODEL,  
        input=input_items,  
        temperature=0.05,  
        max_output_tokens=256,  
        store=False  
    )  
    raw = (extract_output_text(resp) or "").strip()  
  
    try:  
        start = raw.find("{")  
        end = raw.rfind("}")  
        if start != -1 and end != -1 and end > start:  
            raw = raw[start:end + 1]  
        obj = json.loads(raw)  
        qs = obj.get("queries", [])  
        if not isinstance(qs, list):  
            qs = []  
    except Exception:  
        qs = []  
  
    qs = [q.strip()[:30] for q in qs if isinstance(q, str) and q.strip()]  
    qs = list(dict.fromkeys(qs))[:10]  
    if not qs:  
        qs = [current_prompt.strip()[:30] or "検索"]  
    return qs  
  
# ------------------------------- 認証/履歴等 -------------------------------  
def get_authenticated_user():  
    if "user_id" in session and "user_name" in session:  
        return session["user_id"]  
    client_principal = request.headers.get("X-MS-CLIENT-PRINCIPAL")  
    if client_principal:  
        try:  
            decoded = base64.b64decode(client_principal).decode("utf-8")  
            user_data = json.loads(decoded)  
            user_id = user_name = None  
            for claim in user_data.get("claims", []):  
                if claim.get("typ") == "http://schemas.microsoft.com/identity/claims/objectidentifier":  
                    user_id = claim.get("val")  
                if claim.get("typ") == "name":  
                    user_name = claim.get("val")  
            if user_id:  
                session["user_id"] = user_id  
            if user_name:  
                session["user_name"] = user_name  
            return user_id  
        except Exception as e:  
            print("Easy Auth ユーザー情報の取得エラー:", e)  
    session["user_id"] = "anonymous@example.com"  
    session["user_name"] = "anonymous"  
    return session["user_id"]  
  
def compute_has_assistant(msgs: list) -> bool:  
    return any(  
        (m.get("role") == "assistant") and  
        ((m.get("text") or strip_html_tags(m.get("content", ""))).strip())  
        for m in (msgs or [])  
    )  
  
def ensure_messages_from_cosmos(active_sid: str) -> list:  
    """CosmosDBから当該セッションのメッセージを取得（存在しない/エラー時は[]）"""  
    if not (container and active_sid):  
        return []  
    with lock:  
        try:  
            user_id = get_authenticated_user()  
            item = container.read_item(item=active_sid, partition_key=user_id)  
            return item.get("messages", []) or []  
        except Exception:  
            return []  
  
def merge_messages(existing: list, new: list) -> list:  
    """既存と新規の配列を「role + (text or content)」で重複除去して結合（順序はexisting→new）"""  
    existing = existing or []  
    new = new or []  
    if not existing:  
        return new  
    if not new:  
        return existing  
  
    def fp(m):  
        role = m.get("role")  
        text = m.get("text") or strip_html_tags(m.get("content", "")) or ""  
        return f"{role}|{text}".strip()  
  
    seen = set()  
    merged = []  
    for m in existing + new:  
        key = fp(m)  
        if not key:  
            continue  
        if key not in seen:  
            seen.add(key)  
            merged.append(m)  
    return merged  
  
def persist_assistant_message(active_sid: str, assistant_html: str, full_text: str, ragas_scores: dict, system_message: str):  
    """Cosmosを基点にアシスタント応答を安全に保存し、セッションとサイドバーを同期"""  
    if not active_sid:  
        return  
  
    existing_msgs = ensure_messages_from_cosmos(active_sid)  
    session_msgs = session.get("main_chat_messages", [])  
    candidate_msgs = (session_msgs or []) + [{  
        "role": "assistant",  
        "content": assistant_html,  
        "type": "html",  
        "text": full_text,  
        "ragas": ragas_scores or {}  
    }]  
    merged_msgs = merge_messages(existing_msgs, candidate_msgs)  
  
    if not compute_has_assistant(merged_msgs):  
        return  
  
    with lock:  
        try:  
            user_id = get_authenticated_user()  
            user_name = session.get("user_name", "anonymous")  
  
            sb = session.get("sidebar_messages", [])  
            sys_msg = system_message or session.get("default_system_message", "あなたは親切なAIアシスタントです…")  
            if active_sid and sb:  
                match = next((c for c in sb if c.get("session_id") == active_sid), None)  
                if match and match.get("system_message"):  
                    sys_msg = match.get("system_message")  
  
            fam = compute_first_assistant_title(merged_msgs) or ""  
            item = {  
                'id': active_sid,  
                'user_id': user_id,  
                'user_name': user_name,  
                'session_id': active_sid,  
                'messages': merged_msgs,  
                'system_message': sys_msg,  
                'first_assistant_message': fam,  
                'timestamp': datetime.datetime.now(datetime.timezone.utc).isoformat()  
            }  
            if container:  
                container.upsert_item(item)  
  
            session["main_chat_messages"] = merged_msgs  
            sb = session.get("sidebar_messages", [])  
            updated = False  
            if active_sid:  
                for i, chat in enumerate(sb):  
                    if chat.get("session_id") == active_sid:  
                        sb[i]["messages"] = merged_msgs  
                        sb[i]["first_assistant_message"] = compute_first_assistant_title(merged_msgs)  
                        sb[i]["system_message"] = sys_msg  
                        session["sidebar_messages"] = sb  
                        updated = True  
                        break  
            if not updated:  
                sb.insert(0, {  
                    "session_id": active_sid,  
                    "messages": merged_msgs,  
                    "first_assistant_message": compute_first_assistant_title(merged_msgs),  
                    "system_message": sys_msg  
                })  
                session["sidebar_messages"] = sb  
            session.modified = True  
        except Exception as e:  
            print("persist_assistant_message 保存エラー:", e)  
            traceback.print_exc()  
  
def save_chat_history():  
    """  
    主会話(main_chat_messages)ベースで Cosmos に保存。  
    - アシスタント応答がある場合のみ保存  
    - 既存履歴とマージして短縮上書きを防ぐ  
    - system_message はサイドバーにあればそれを優先  
    """  
    if not container:  
        return  
    with lock:  
        try:  
            active_sid = session.get("current_session_id")  
            msgs = session.get("main_chat_messages", []) or []  
            if not active_sid or not msgs:  
                return  
  
            user_id = get_authenticated_user()  
            existing = []  
            try:  
                existing_item = container.read_item(item=active_sid, partition_key=user_id)  
                existing = existing_item.get("messages", []) or []  
            except Exception:  
                pass  
            msgs = merge_messages(existing, msgs)  
  
            if not compute_has_assistant(msgs):  
                return  
  
            system_message = session.get("default_system_message", "あなたは親切なAIアシスタントです…")  
            sb = session.get("sidebar_messages", [])  
            if active_sid and sb:  
                match = next((c for c in sb if c.get("session_id") == active_sid), None)  
                if match and match.get("system_message"):  
                    system_message = match.get("system_message")  
  
            fam = compute_first_assistant_title(msgs) or ""  
            user_name = session.get("user_name", "anonymous")  
            item = {  
                'id': active_sid,  
                'user_id': user_id,  
                'user_name': user_name,  
                'session_id': active_sid,  
                'messages': msgs,  
                'system_message': system_message,  
                'first_assistant_message': fam,  
                'timestamp': datetime.datetime.now(datetime.timezone.utc).isoformat()  
            }  
            container.upsert_item(item)  
  
            session["main_chat_messages"] = msgs  
            sb = session.get("sidebar_messages", [])  
            updated = False  
            for i, chat in enumerate(sb):  
                if chat.get("session_id") == active_sid:  
                    sb[i]["messages"] = msgs  
                    sb[i]["first_assistant_message"] = fam  
                    sb[i]["system_message"] = system_message  
                    session["sidebar_messages"] = sb  
                    updated = True  
                    break  
            if not updated:  
                sb.insert(0, {  
                    "session_id": active_sid,  
                    "messages": msgs,  
                    "first_assistant_message": fam,  
                    "system_message": system_message  
                })  
                session["sidebar_messages"] = sb  
            session.modified = True  
        except Exception as e:  
            print("チャット履歴保存エラー:", e)  
            traceback.print_exc()  
  
def load_chat_history():  
    if not container:  
        return []  
    with lock:  
        user_id = get_authenticated_user()  
        sidebar_messages = []  
        try:  
            one_week_ago = (datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=7)).isoformat()  
            query = """  
            SELECT * FROM c  
            WHERE c.user_id = @user_id AND c.timestamp >= @one_week_ago  
            ORDER BY c.timestamp DESC  
            """  
            parameters = [  
                {"name": "@user_id", "value": user_id},  
                {"name": "@one_week_ago", "value": one_week_ago},  
            ]  
            items = container.query_items(query=query, parameters=parameters, enable_cross_partition_query=True)  
            for item in items:  
                if 'session_id' in item:  
                    msgs = item.get("messages", []) or []  
                    if not compute_has_assistant(msgs):  
                        continue  
                    fam = item.get("first_assistant_message") or compute_first_assistant_title(msgs) or ""  
                    chat = {  
                        "session_id": item['session_id'],  
                        "messages": msgs,  
                        "system_message": item.get(  
                            "system_message", session.get('default_system_message', "あなたは親切なAIアシスタントです…")  
                        ),  
                        "first_assistant_message": fam  
                    }  
                    sidebar_messages.append(chat)  
        except Exception as e:  
            print("チャット履歴読み込みエラー:", e)  
            traceback.print_exc()  
        return sidebar_messages  
  
def save_system_prompt_item(title: str, content: str):  
    if not container:  
        return None  
    with lock:  
        try:  
            user_id = get_authenticated_user()  
            user_name = session.get('user_name', 'anonymous')  
            item = {  
                'id': str(uuid.uuid4()),  
                'doc_type': 'system_prompt',  
                'user_id': user_id,  
                'user_name': user_name,  
                'title': title,  
                'content': content,  
                'timestamp': datetime.datetime.now(datetime.timezone.utc).isoformat()  
            }  
            container.upsert_item(item)  
            return item['id']  
        except Exception as e:  
            print("システムプロンプト保存エラー:", e)  
            traceback.print_exc()  
            return None  
  
def load_system_prompts():  
    if not container:  
        return []  
    with lock:  
        get_authenticated_user()  
        prompts = []  
        try:  
            query = """  
            SELECT c.id, c.title, c.content, c.timestamp  
            FROM c  
            WHERE c.user_id = @user_id AND c.doc_type = 'system_prompt'  
            ORDER BY c.timestamp DESC  
            """  
            parameters = [{"name": "@user_id", "value": session["user_id"]}]  
            items = container.query_items(query=query, parameters=parameters, enable_cross_partition_query=True)  
            for it in items:  
                prompts.append({  
                    "id": it.get("id"),  
                    "title": it.get("title"),  
                    "content": it.get("content"),  
                    "timestamp": it.get("timestamp")  
                })  
        except Exception as e:  
            print("システムプロンプト読込エラー:", e)  
            traceback.print_exc()  
        return prompts  
  
def delete_system_prompt(prompt_id: str):  
    if not container:  
        return False  
    with lock:  
        try:  
            user_id = get_authenticated_user()  
            container.delete_item(item=prompt_id, partition_key=user_id)  
            return True  
        except Exception as e:  
            print("システムプロンプト削除エラー:", e)  
            traceback.print_exc()  
            return False  
  
def start_new_chat():  
    """  
    新規チャット開始:  
    - 画像/PDFをクリーンアップ  
    - 新しい session_id を発行  
    - main_chat_messages は空にする  
    - サイドバー（sidebar_messages）は挿入しない（空セッション非表示）  
    - current_chat_index は変更しない（上書き防止）  
    """  
    image_filenames = session.get("image_filenames", [])  
    for img_name in image_filenames:  
        if image_container_client:  
            try:  
                image_container_client.get_blob_client(img_name).delete_blob()  
            except Exception as e:  
                print("画像削除エラー:", e)  
    session["image_filenames"] = []  
  
    file_filenames = session.get("file_filenames", [])  
    for file_name in file_filenames:  
        if file_container_client:  
            try:  
                file_container_client.get_blob_client(file_name).delete_blob()  
            except Exception as e:  
                print("ファイル削除エラー:", e)  
    session["file_filenames"] = []  
  
    new_session_id = str(uuid.uuid4())  
    session["current_session_id"] = new_session_id  
    session["main_chat_messages"] = []  
    session.modified = True  
  
# ------------------------------- Azure Cognitive Search -------------------------------  
def get_search_client(index_name):  
    return SearchClient(  
        endpoint=search_service_endpoint,  
        index_name=index_name,  
        credential=AzureKeyCredential(search_service_key),  
        transport=transport,  
        retry_policy=retry_policy,  
        api_version="2024-07-01",  
    )  
  
def keyword_search(query, topNDocuments, index_name):  
    sc = get_search_client(index_name)  
    results = sc.search(  
        search_text=query,  
        search_fields=["title", "content"],  
        select="chunk_id, parent_id, title, content, filepath, url",  
        query_type="simple",  
        search_mode="all",  
        top=topNDocuments,  
    )  
    return list(results)  
  
def keyword_semantic_search(query, topNDocuments, index_name, strictness=0.0):  
    sc = get_search_client(index_name)  
    try:  
        top = max(1, min(300, int(topNDocuments)))  
    except Exception:  
        top = 50  
    results = sc.search(  
        search_text=query,  
        search_fields=["title", "content"],  
        select="chunk_id, parent_id, title, content, filepath, url",  
        query_type="semantic",  
        semantic_configuration_name="default",  
        query_caption="extractive",  
        query_answer="extractive",  
        top=top,  
    )  
    filtered = []  
    for r in results:  
        score = r.get("@search.rerankerScore", r.get("@search.score", 0.0))  
        if score >= strictness:  
            filtered.append(r)  
    filtered.sort(key=lambda x: x.get("@search.rerankerScore", x.get("@search.score", 0.0)), reverse=True)  
    return filtered  
  
def get_query_embedding(query):  
    try:  
        resp = embed_client.embeddings.create(model=EMBEDDING_MODEL, input=query, dimensions=1536)  
        return resp.data[0].embedding  
    except Exception as e:  
        print("Embedding 生成エラー:", e)  
        traceback.print_exc()  
        return []  
  
def keyword_vector_search(query, topNDocuments, index_name):  
    try:  
        sc = get_search_client(index_name)  
        embedding = get_query_embedding(query)  
        if not embedding:  
            return []  
        vector_query = {  
            "kind": "vector",  
            "vector": embedding,  
            "exhaustive": True,  
            "fields": VECTOR_FIELD,  
            "weight": 0.5,  
            "k": topNDocuments,  
        }  
        results = sc.search(  
            search_text="*",  
            vector_queries=[vector_query],  
            select="chunk_id, parent_id, title, content, filepath, url",  
            top=topNDocuments,  
        )  
        results_list = list(results)  
        if results_list and "@search.score" in results_list[0]:  
            results_list.sort(key=lambda x: x.get("@search.score", 0), reverse=True)  
        return results_list  
    except Exception as e:  
        print("ベクター検索エラー:", e)  
        traceback.print_exc()  
        return []  
  
def hybrid_search_multiqueries(queries, topNDocuments, index_name, strictness=0.0):  
    rrf_k = 60  
    fusion_scores = {}  
    fusion_docs = {}  
    try:  
        req_top = int(topNDocuments)  
    except Exception:  
        req_top = 10  
    fetch_top = int(min(RRF_FETCH_MAX_TOP, max(req_top, req_top * RRF_FETCH_MULTIPLIER)))  
    parent_counts = {}  
  
    for qtext in queries:  
        lists = [  
            keyword_search(qtext, fetch_top, index_name),  
            keyword_semantic_search(qtext, fetch_top, index_name, strictness),  
            keyword_vector_search(qtext, fetch_top, index_name),  
        ]  
        for result_list in lists:  
            for idx, r in enumerate(result_list):  
                dedup_key = r.get("chunk_id") or r.get("filepath") or r.get("id") or r.get("title")  
                if not dedup_key:  
                    continue  
                parent_id = r.get("parent_id")  
                if MAX_CHUNKS_PER_PARENT > 0 and parent_id and dedup_key not in fusion_docs:  
                    if parent_counts.get(parent_id, 0) >= MAX_CHUNKS_PER_PARENT:  
                        continue  
                contribution = 1 / (rrf_k + (idx + 1))  
                prev = fusion_scores.get(dedup_key)  
                fusion_scores[dedup_key] = (prev or 0) + contribution  
                if dedup_key not in fusion_docs:  
                    fusion_docs[dedup_key] = r  
                if MAX_CHUNKS_PER_PARENT > 0 and parent_id:  
                    parent_counts[parent_id] = parent_counts.get(parent_id, 0) + 1  
  
    sorted_keys = sorted(fusion_scores, key=lambda d: fusion_scores[d], reverse=True)  
    fused_results = []  
    for k in sorted_keys[:req_top]:  
        r = fusion_docs[k]  
        r["fusion_score"] = fusion_scores[k]  
        fused_results.append(r)  
    return fused_results  
  
def rrf_fuse_ranked_lists(lists_of_results, topNDocuments):  
    rrf_k = 60  
    fusion_scores = {}  
    fusion_docs = {}  
    try:  
        req_top = int(topNDocuments)  
    except Exception:  
        req_top = 10  
    for result_list in lists_of_results:  
        for idx, r in enumerate(result_list):  
            dedup_key = r.get("chunk_id") or r.get("filepath") or r.get("id") or r.get("title")  
            if not dedup_key:  
                continue  
            contribution = 1 / (rrf_k + (idx + 1))  
            prev = fusion_scores.get(dedup_key)  
            fusion_scores[dedup_key] = (prev or 0) + contribution  
            if dedup_key not in fusion_docs:  
                fusion_docs[dedup_key] = r  
    sorted_keys = sorted(fusion_scores, key=lambda d: fusion_scores[d], reverse=True)  
    fused_results = []  
    for k in sorted_keys[:req_top]:  
        r = fusion_docs[k]  
        r["fusion_score"] = fusion_scores[k]  
        fused_results.append(r)  
    return fused_results  
  
# ------------------------------- HyDE / PRF -------------------------------  
def hyde_paragraph(user_input: str) -> str:  
    sys_msg = "あなたは検索のための仮想文書(HyDE)を作成します。事実の付け足しは避け、中立で。"  
    p = f"""以下の質問に関して、中立で百科事典調の根拠段落を日本語で5-7文で作成してください。事実断定は避け、関連する用語を豊富に含めてください。これは検索用の仮想文書であり回答ではありません。質問: {user_input}"""  
    input_items = [  
        {"role": "system", "content": [{"type": "input_text", "text": sys_msg}]},  
        {"role": "user", "content": [{"type": "input_text", "text": p}]},  
    ]  
    resp = client.responses.create(  
        model=RESPONSES_MODEL,  
        input=input_items,  
        temperature=0.2,  
        max_output_tokens=512,  
        store=False  
    )  
    return (extract_output_text(resp) or "").strip()  
  
def refine_query_with_prf(initial_query: str, titles: list) -> str:  
    t = "\n".join(f"- {x}" for x in (titles or [])[:8])  
    prompt = f"""初回検索の上位文書のタイトル一覧です。これを参考に、より適合度の高い検索クエリを日本語で1本だけ生成。不要語は削除し、重要語は維持。出力はクエリ文字列のみ。タイトル:{t}  初回クエリ: {initial_query}"""  
    input_items = [  
        {"role": "system", "content": [{"type": "input_text", "text": "あなたは検索クエリの改良を行うプロフェッショナルです。"}]},  
        {"role": "user", "content": [{"type": "input_text", "text": prompt}]},  
    ]  
    resp = client.responses.create(  
        model=RESPONSES_MODEL,  
        input=input_items,  
        temperature=0,  
        max_output_tokens=256,  
        store=False  
    )  
    return (extract_output_text(resp) or "").strip().strip('「」\' ')  
  
def unique_parents(results):  
    return len({(r.get("parent_id") or r.get("filepath") or r.get("title")) for r in (results or [])})  
  
# ------------------------------- RAGAS -------------------------------  
RAGAS_LLM_WRAPPER = None  
RAGAS_EMB_WRAPPER = None  
  
def init_ragas_clients():  
    global RAGAS_LLM_WRAPPER, RAGAS_EMB_WRAPPER  
    if RAGAS_LLM_WRAPPER and RAGAS_EMB_WRAPPER:  
        return RAGAS_LLM_WRAPPER, RAGAS_EMB_WRAPPER  
    if not RAGAS_AVAILABLE:  
        return None, None  
  
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")  
    api_key = os.getenv("AZURE_OPENAI_KEY")  
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-06-01")  
  
    judge_deploy = os.getenv("AZURE_OPENAI_RAGAS_JUDGE_DEPLOYMENT") or os.getenv("AZURE_OPENAI_RESPONSES_MODEL", "gpt-4o")  
    embed_deploy = os.getenv("AZURE_OPENAI_EMBED_DEPLOYMENT", EMBEDDING_MODEL)  
  
    llm = ChatOpenAI(  
        azure_endpoint=azure_endpoint,  
        api_key=api_key,  
        azure_deployment=judge_deploy,  
        api_version=api_version,  
        temperature=0  
    )  
    emb = OpenAIEmbeddings(  
        azure_endpoint=embed_endpoint or azure_endpoint,  
        api_key=api_key,  
        azure_deployment=embed_deploy,  
        api_version=os.getenv("AZURE_OPENAI_EMBED_API_VERSION", "2024-06-01")  
    )  
  
    RAGAS_LLM_WRAPPER = LangchainLLMWrapper(llm)  
    RAGAS_EMB_WRAPPER = LangchainEmbeddingsWrapper(emb)  
    return RAGAS_LLM_WRAPPER, RAGAS_EMB_WRAPPER  
  
def compute_ragas_metrics(question: str, contexts: list, answer: str) -> dict:  
    try:  
        if not ENABLE_RAGAS or not RAGAS_AVAILABLE:  
            return {}  
        if not answer or not question:  
            return {}  
        llm, emb = init_ragas_clients()  
        if not llm or not emb:  
            return {}  
  
        ctxs = [c for c in (contexts or []) if isinstance(c, str) and c.strip()]  
        ds = Dataset.from_dict({  
            "question": [question],  
            "answer": [answer],  
            "contexts": [ctxs]  
        })  
  
        metrics = [answer_relevancy]  
        if ctxs:  
            metrics += [faithfulness, context_precision, context_relevancy]  
  
        result = evaluate(ds, metrics=metrics, llm=llm, embeddings=emb)  
        scores = {}  
        try:  
            df = result.to_pandas()  
            row = df.iloc[0]  
            for m in metrics:  
                name = getattr(m, "name", None) or m.__class__.__name__  
                if name in df.columns and isinstance(row[name], (int, float)):  
                    scores[name] = round(float(row[name]), 4)  
        except Exception:  
            if hasattr(result, "scores"):  
                for m in metrics:  
                    name = getattr(m, "name", None) or m.__class__.__name__  
                    val = result.scores.get(name)  
                    if isinstance(val, list):  
                        val = val[0]  
                    if isinstance(val, (int, float)):  
                        scores[name] = round(float(val), 4)  
        return scores  
    except Exception as e:  
        print("RAGAS evaluate error:", e)  
        traceback.print_exc()  
        return {}  
  
# ------------------------------- Responses 入力構築 -------------------------------  
def build_responses_input_with_history(  
    all_messages,  
    system_message,  
    context_text,  
    max_history_to_send=None,  
    wrap_context_with_markers=True,  
    context_title="参考コンテキスト（RAG）",  
):  
    input_items = []  
  
    if system_message:  
        input_items.append({  
            "role": "system",  
            "content": [{"type": "input_text", "text": system_message}]  
        })  
  
    if max_history_to_send is None:  
        max_history_to_send = MAX_HISTORY_TO_SEND  
    try:  
        max_history_to_send = int(max_history_to_send)  
    except Exception:  
        max_history_to_send = MAX_HISTORY_TO_SEND  
    max_history_to_send = max(1, min(50, max_history_to_send))  
  
    hist = (all_messages or [])[-max_history_to_send:]  
    for m in hist:  
        role = m.get("role", "user")  
        if role not in ["user", "assistant"]:  
            continue  
  
        if role == "assistant":  
            text = m.get("text") or strip_html_tags(m.get("content", ""))  
            ctype = "output_text"  
        else:  
            text = m.get("content", "")  
            ctype = "input_text"  
  
        if not text:  
            continue  
        input_items.append({  
            "role": role,  
            "content": [{"type": ctype, "text": text}]  
        })  
  
    last_user_index = None  
    for i in range(len(input_items) - 1, -1, -1):  
        if input_items[i].get("role") == "user":  
            last_user_index = i  
            break  
  
    target_user_index = last_user_index  
    if context_text:  
        ctx = (  
            f"{context_title}:\nBEGIN_CONTEXT\n{context_text}\nEND_CONTEXT"  
            if wrap_context_with_markers else context_text  
        )  
        if last_user_index is not None:  
            input_items[last_user_index]["content"].append({  
                "type": "input_text",  
                "text": ctx  
            })  
            target_user_index = last_user_index  
        else:  
            input_items.append({  
                "role": "user",  
                "content": [{"type": "input_text", "text": ctx}]  
            })  
            target_user_index = len(input_items) - 1  
  
    return input_items, target_user_index  
  
# ------------------------------- セッション内ファイル一括削除 -------------------------------  
def _delete_uploaded_files_for_session():  
    deleted = {"images": [], "files": []}  
    try:  
        image_filenames = list(session.get("image_filenames", []))  
        if image_container_client:  
            for name in image_filenames:  
                try:  
                    bc = image_container_client.get_blob_client(name)  
                    if bc.exists():  
                        bc.delete_blob(delete_snapshots="include")  
                        deleted["images"].append(name)  
                except Exception as e:  
                    print("画像削除失敗:", name, e)  
        file_filenames = list(session.get("file_filenames", []))  
        if file_container_client:  
            for name in file_filenames:  
                try:  
                    bc = file_container_client.get_blob_client(name)  
                    if bc.exists():  
                        bc.delete_blob(delete_snapshots="include")  
                        deleted["files"].append(name)  
                except Exception as e:  
                    print("PDF削除失敗:", name, e)  
        # ここではセッションを書き換えない（競合・上書き回避）  
    except Exception as e:  
        print("セッションファイル一括削除エラー:", e)  
    return deleted  
  
@app.route('/cleanup_session_files', methods=['POST'])  
def cleanup_session_files():  
    result = _delete_uploaded_files_for_session()  
    return jsonify({"ok": True, "deleted": result})  
  
# ------------------------------- 設定更新（AJAX） -------------------------------  
@app.route('/update_settings', methods=['POST'])  
def update_settings():  
    get_authenticated_user()  
    data = request.get_json(silent=True) or {}  
    changed = {}  
  
    if "selected_model" in data:  
        selected = (data.get("selected_model") or "").strip()  
        allowed_models = {"gpt-4o", "gpt-4.1", "o3", "o4-mini", "gpt-5"}  
        if selected in allowed_models:  
            session["selected_model"] = selected  
            changed["selected_model"] = selected  
    if "reasoning_effort" in data:  
        effort = (data.get("reasoning_effort") or "").strip().lower()  
        allowed_efforts = {"low", "medium", "high"}  
        if session.get("selected_model") in REASONING_ENABLED_MODELS and effort in allowed_efforts:  
            session["reasoning_effort"] = effort  
            changed["reasoning_effort"] = effort  
    if "selected_search_index" in data:  
        sel_index = (data.get("selected_search_index") or "").strip()  
        if sel_index in INDEX_VALUES:  
            session["selected_search_index"] = sel_index  
            changed["selected_search_index"] = sel_index  
    if "doc_count" in data:  
        try:  
            val = int(data.get("doc_count"))  
        except Exception:  
            val = DEFAULT_DOC_COUNT  
        session["doc_count"] = max(1, min(300, val))  
        changed["doc_count"] = session["doc_count"]  
    if "history_to_send" in data:  
        try:  
            val = int(data.get("history_to_send"))  
        except Exception:  
            val = MAX_HISTORY_TO_SEND  
        session["history_to_send"] = max(1, min(50, val))  
        changed["history_to_send"] = session["history_to_send"]  
    if "default_system_message" in data:  
        sys_msg = (data.get("default_system_message") or "").strip()  
        session["default_system_message"] = sys_msg  
        active_sid = session.get("current_session_id")  
        sb = session.get("sidebar_messages", [])  
        if active_sid:  
            for i, chat in enumerate(sb):  
                if chat.get("session_id") == active_sid:  
                    sb[i]["system_message"] = sys_msg  
                    session["sidebar_messages"] = sb  
                    break  
        changed["default_system_message"] = sys_msg  
    if "rag_enabled" in data:  
        raw = data.get("rag_enabled")  
        val = bool(raw)  
        session["rag_enabled"] = val  
        changed["rag_enabled"] = val  
    session.modified = True  
    return jsonify({"ok": True, "changed": changed})  
  
# ------------------------------- ルーティング -------------------------------  
@app.route('/', methods=['GET', 'POST'])  
def index():  
    get_authenticated_user()  
  
    # 初期設定  
    if "selected_model" not in session:  
        session["selected_model"] = RESPONSES_MODEL  
    if "reasoning_effort" not in session:  
        session["reasoning_effort"] = REASONING_EFFORT  
    if "selected_search_index" not in session:  
        session["selected_search_index"] = DEFAULT_SEARCH_INDEX  
    if "doc_count" not in session:  
        session["doc_count"] = max(1, min(300, int(DEFAULT_DOC_COUNT)))  
    if "history_to_send" not in session:  
        session["history_to_send"] = max(1, min(50, int(MAX_HISTORY_TO_SEND)))  
    if "default_system_message" not in session:  
        session["default_system_message"] = (  
            "あなたは親切なAIアシスタントです。ユーザーの質問が不明確な場合は、"  
            "「こういうことですか？」と内容を確認してください。質問が明確な場合は、"  
            "簡潔かつ正確に答えてください。"  
        )  
  
    # 毎回 Cosmos から履歴を再同期（空セッション非表示）  
    session["sidebar_messages"] = load_chat_history() or []  
  
    # アプリ起動時（このセッションの初回表示）は必ず新規チャットを開始  
    if not session.get("initial_chat_opened"):  
        start_new_chat()  
        session["initial_chat_opened"] = True  
  
    # main_chat_messages が未設定なら空で初期化（初回は必ず空の新規チャット）  
    if "main_chat_messages" not in session:  
        session["main_chat_messages"] = []  
  
    # current_session_id の初期補完（万一欠落時）  
    if "current_session_id" not in session or not session.get("current_session_id"):  
        start_new_chat()  
  
    if "image_filenames" not in session:  
        session["image_filenames"] = []  
    if "file_filenames" not in session:  
        session["file_filenames"] = []  
    if "saved_prompts" not in session:  
        session["saved_prompts"] = load_system_prompts()  
    if "show_all_history" not in session:  
        session["show_all_history"] = False  
    if "upload_prefix" not in session:  
        session["upload_prefix"] = str(uuid.uuid4())  
    if "rag_enabled" not in session:  
        session["rag_enabled"] = True  
    session.modified = True  
  
    # --- POST ハンドラ ---  
    if request.method == 'POST':  
        # 新しいチャット  
        if request.form.get('new_chat'):  
            start_new_chat()  
            session.modified = True  
            return redirect(url_for('index'))  
  
        # 既存チャットを選択  
        if 'select_chat' in request.form:  
            sid = request.form.get('select_chat')  
            sb = session.get("sidebar_messages", [])  
            for i, chat in enumerate(sb):  
                if chat.get("session_id") == sid:  
                    session["current_chat_index"] = i  
                    session["current_session_id"] = sid  
                    msgs = ensure_messages_from_cosmos(sid) or chat.get("messages", [])  
                    session["main_chat_messages"] = msgs  
                    session.modified = True  
                    break  
            return redirect(url_for('index'))  
  
        # 履歴の表示件数トグル  
        if 'toggle_history' in request.form:  
            session["show_all_history"] = not bool(session.get("show_all_history", False))  
            session.modified = True  
            return redirect(url_for('index'))  
  
        # system_message の直接更新（session_id ベース）  
        if 'set_system_message' in request.form:  
            sys_msg = (request.form.get('system_message') or '').strip()  
            session["default_system_message"] = sys_msg  
            active_sid = session.get("current_session_id")  
            sb = session.get("sidebar_messages", [])  
            if active_sid:  
                for i, chat in enumerate(sb):  
                    if chat.get("session_id") == active_sid:  
                        sb[i]["system_message"] = sys_msg  
                        session["sidebar_messages"] = sb  
                        break  
            session.modified = True  
            return redirect(url_for('index'))  
  
        # 登録済みプロンプトを適用（session_id ベース）  
        if 'apply_system_prompt' in request.form:  
            prompt_id = request.form.get('select_prompt_id')  
            saved = session.get("saved_prompts", [])  
            match = next((p for p in saved if p.get("id") == prompt_id), None)  
            if match:  
                session["default_system_message"] = match.get("content", "")  
                active_sid = session.get("current_session_id")  
                sb = session.get("sidebar_messages", [])  
                if active_sid:  
                    for i, chat in enumerate(sb):  
                        if chat.get("session_id") == active_sid:  
                            sb[i]["system_message"] = session["default_system_message"]  
                            session["sidebar_messages"] = sb  
                            break  
            session.modified = True  
            return redirect(url_for('index'))  
  
        # 新しい指示を保存  
        if 'add_system_prompt' in request.form:  
            title = (request.form.get('prompt_title') or '').strip()  
            content = (request.form.get('prompt_content') or '').strip()  
            if title and content:  
                pid = save_system_prompt_item(title, content)  
                if pid:  
                    session["saved_prompts"] = load_system_prompts()  
            session.modified = True  
            return redirect(url_for('index'))  
  
        # （フォーム直送のためのフォールバック設定。AJAXが効いていない環境用）  
        if 'set_model' in request.form:  
            selected = (request.form.get('model') or '').strip()  
            allowed_models = {"gpt-4o", "gpt-4.1", "o3", "o4-mini", "gpt-5"}  
            if selected in allowed_models:  
                session["selected_model"] = selected  
            session.modified = True  
            return redirect(url_for('index'))  
  
        if 'set_index' in request.form:  
            sel_index = (request.form.get('search_index') or '').strip()  
            if sel_index in INDEX_VALUES:  
                session["selected_search_index"] = sel_index  
            session.modified = True  
            return redirect(url_for('index'))  
  
        if 'set_doc_count' in request.form:  
            try:  
                val = int(request.form.get('doc_count') or DEFAULT_DOC_COUNT)  
            except Exception:  
                val = DEFAULT_DOC_COUNT  
            session["doc_count"] = max(1, min(300, val))  
            session.modified = True  
            return redirect(url_for('index'))  
  
        if 'set_history_to_send' in request.form:  
            try:  
                val = int(request.form.get('history_to_send') or MAX_HISTORY_TO_SEND)  
            except Exception:  
                val = MAX_HISTORY_TO_SEND  
            session["history_to_send"] = max(1, min(50, val))  
            session.modified = True  
            return redirect(url_for('index'))  
  
        # 画像/PDFアップロード  
        if 'upload_files' in request.form:  
            if not blob_service_client or not file_container_client or not image_container_client:  
                flash("ストレージ未設定です。環境変数 AZURE_STORAGE_CONNECTION_STRING を設定してください。", "error")  
                return redirect(url_for('index'))  
  
            files_list = request.files.getlist('files')  
            if not files_list:  
                flash("ファイルが選択されていません。", "warning")  
                return redirect(url_for('index'))  
  
            upload_prefix = session.get("upload_prefix", str(uuid.uuid4()))  
            uploaded_images = session.get("image_filenames", [])  
            uploaded_pdfs = session.get("file_filenames", [])  
  
            for f in files_list:  
                if not f or not f.filename:  
                    continue  
                fname = secure_filename(f.filename)  
                ext = os.path.splitext(fname)[1].lower()  
                blobname = f"{upload_prefix}/{uuid.uuid4().hex}__{fname}"  
  
                try:  
                    if ext == '.pdf':  
                        bc = file_container_client.get_blob_client(blobname)  
                        bc.upload_blob(f.stream, overwrite=True, content_type="application/pdf")  
                        uploaded_pdfs.append(blobname)  
                        print("Uploaded PDF:", blobname)  
                    elif ext in ['.png', '.jpg', '.jpeg', '.gif']:  
                        bc = image_container_client.get_blob_client(blobname)  
                        mime = "image/png" if ext == '.png' else ("image/gif" if ext == '.gif' else "image/jpeg")  
                        bc.upload_blob(f.stream, overwrite=True, content_type=mime)  
                        uploaded_images.append(blobname)  
                        print("Uploaded image:", blobname)  
                    else:  
                        flash(f"未対応の拡張子: {fname}", "warning")  
                except Exception as e:  
                    print("Upload error:", e)  
                    traceback.print_exc()  
                    flash(f"アップロードに失敗: {fname} ({e})", "error")  
  
            session["image_filenames"] = uploaded_images  
            session["file_filenames"] = uploaded_pdfs  
            session.modified = True  
            return redirect(url_for('index'))  
  
        # 画像削除  
        if 'delete_image' in request.form:  
            name = request.form.get('delete_image')  
            if image_container_client and name:  
                try:  
                    image_container_client.get_blob_client(name).delete_blob(delete_snapshots="include")  
                except Exception as e:  
                    print("画像削除エラー:", e)  
                    flash(f"画像削除失敗: {e}", "error")  
            session["image_filenames"] = [x for x in session.get("image_filenames", []) if x != name]  
            session.modified = True  
            return redirect(url_for('index'))  
  
        # PDF削除  
        if 'delete_file' in request.form:  
            name = request.form.get('delete_file')  
            if file_container_client and name:  
                try:  
                    file_container_client.get_blob_client(name).delete_blob(delete_snapshots="include")  
                except Exception as e:  
                    print("PDF削除エラー:", e)  
                    flash(f"PDF削除失敗: {e}", "error")  
            session["file_filenames"] = [x for x in session.get("file_filenames", []) if x != name]  
            session.modified = True  
            return redirect(url_for('index'))  
  
        # その他POST  
        return redirect(url_for('index'))  
  
    # --- GET: 描画 ---  
    images = []  
    if image_container_client:  
        for blobname in session.get("image_filenames", []):  
            base = blobname.split("/")[-1]  
            display = base.split("__", 1)[1] if "__" in base else base  
            url = make_blob_url(image_container_name, blobname)  
            images.append({'name': display, 'blob': blobname, 'url': url})  
  
    files = []  
    if file_container_client:  
        for blobname in session.get("file_filenames", []):  
            base = blobname.split("/")[-1]  
            display = base.split("__", 1)[1] if "__" in base else base  
            url = make_blob_url(file_container_name, blobname)  
            files.append({'name': display, 'blob': blobname, 'url': url})  
  
    chat_history = session.get("main_chat_messages", [])  
    sidebar_messages = session.get("sidebar_messages", [])  
    saved_prompts = session.get("saved_prompts", [])  
    max_displayed_history = 6  
    max_total_history = 50  
    show_all_history = session.get("show_all_history", False)  
  
    return render_template(  
        'index.html',  
        chat_history=chat_history,  
        chat_sessions=sidebar_messages,  
        images=images,  
        files=files,  
        show_all_history=show_all_history,  
        max_displayed_history=max_displayed_history,  
        max_total_history=max_total_history,  
        session=session,  
        index_options=INDEX_OPTIONS,  
        saved_prompts=saved_prompts  
    )  
  
# ------------------------------- 準備/送信/SSE -------------------------------  
@app.route('/prepare_stream', methods=['POST'])  
def prepare_stream():  
    data = request.get_json(silent=True) or {}  
    prompt = (data.get('prompt') or '').strip()  
    if not prompt:  
        return jsonify({"error": "missing prompt"}), 400  
    prepared = session.get('prepared_prompts', {})  
    mid = str(uuid.uuid4())  
    prepared[mid] = prompt  
    session['prepared_prompts'] = prepared  
    session.modified = True  
    return jsonify({"message_id": mid})  
  
@app.route('/send_message', methods=['POST'])  
def send_message():  
    data = request.get_json()  
    prompt = (data.get('prompt') or '').strip()  
    if not prompt:  
        return (  
            json.dumps({'response': '', 'search_files': [], 'reasoning_summary': '', 'rewritten_queries': [], 'ragas': {}, 'session_id': session.get("current_session_id", "")}),  
            400,  
            {'Content-Type': 'application/json'}  
        )  
  
    # 開始時点のアクティブ会話IDを安定取得  
    active_sid = session.get("current_session_id")  
    if not active_sid:  
        sb = session.get("sidebar_messages", [])  
        if sb:  
            idx = session.get("current_chat_index", 0)  
            active_sid = sb[idx].get("session_id")  
            session["current_session_id"] = active_sid  
        else:  
            start_new_chat()  
            active_sid = session.get("current_session_id")  
  
    # Cosmos から復元（セッションが空の場合）  
    messages = session.get("main_chat_messages", [])  
    if not messages and container and active_sid:  
        messages = ensure_messages_from_cosmos(active_sid)  
        session["main_chat_messages"] = messages  
        session.modified = True  
  
    # ユーザメッセージを追加  
    messages.append({"role": "user", "content": prompt, "type": "text"})  
    session["main_chat_messages"] = messages  
    session.modified = True  
    save_chat_history()  
  
    try:  
        selected_index = session.get("selected_search_index", DEFAULT_SEARCH_INDEX)  
        doc_count = max(1, min(300, int(session.get("doc_count", DEFAULT_DOC_COUNT))))  
        rag_enabled = bool(session.get("rag_enabled", True))  
  
        # system_message は current_session_id から優先取得  
        system_message = session.get("default_system_message", "")  
        sb_for_sys = session.get("sidebar_messages", [])  
        if active_sid:  
            match_chat = next((c for c in sb_for_sys if c.get("session_id") == active_sid), None)  
            if match_chat:  
                system_message = match_chat.get("system_message", system_message)  
        else:  
            idx = session.get("current_chat_index", 0)  
            if sb_for_sys and 0 <= idx < len(sb_for_sys):  
                system_message = sb_for_sys[idx].get("system_message", system_message)  
  
        queries = []  
        search_files = []  
        context = ""  
  
        if rag_enabled:  
            queries = rewrite_queries_for_search_responses(messages, prompt, system_message)  
  
            strictness = 0.0  
            results_list = hybrid_search_multiqueries(queries, doc_count, selected_index, strictness)  
  
            threshold = max(MIN_UNIQUE_PARENTS_ABS, int(doc_count * RECALL_PARENT_THRESHOLD_FRACTION))  
            if ENABLE_HYDE and unique_parents(results_list) < threshold:  
                hyde_doc = hyde_paragraph(prompt)  
                hyde_vec_results = keyword_vector_search(hyde_doc, doc_count, selected_index)  
                results_list = rrf_fuse_ranked_lists([results_list, hyde_vec_results], doc_count)  
  
            if ENABLE_PRF and unique_parents(results_list) < threshold and queries:  
                top_titles = [r.get("title", "") for r in results_list[:8]]  
                prf_q = refine_query_with_prf(queries[0], top_titles)  
                queries2 = list(dict.fromkeys([prf_q] + queries))[:2]  
                results_list = hybrid_search_multiqueries(queries2, doc_count, selected_index, strictness)  
                queries = queries2  
  
            context = "\n".join([  
                f"ファイル名: {r.get('title', '不明')}\n内容: {r.get('content','')}"  
                for r in results_list  
            ])[:50000]  
  
            for r in results_list:  
                title = r.get('title', '不明')  
                content = r.get('content', '')  
                fp = (r.get('filepath') or '').replace('\\', '/')  
                container_name, blobname = (None, None)  
                if fp:  
                    container_name, blobname = resolve_blob_from_filepath(selected_index, fp)  
                else:  
                    u = r.get('url') or ''  
                    if is_azure_blob_url(u):  
                        container_name, blobname = resolve_blob_from_filepath(selected_index, u)  
                url = ''  
                if container_name and blobname:  
                    try:  
                        if blobname.lower().endswith('.txt'):  
                            url = url_for('download_txt', container=container_name, blobname=quote(blobname))  
                        else:  
                            blob_client = blob_service_client.get_blob_client(container=container_name, blob=blobname)  
                            url = generate_sas_url(blob_client, blobname)  
                    except Exception as e:  
                        print("SAS/ダウンロードURL生成エラー:", e)  
                path_display = fp or (f"{container_name}/{blobname}" if (container_name and blobname) else (r.get('url') or ''))  
                search_files.append({'title': title, 'content': content, 'url': url, 'filepath': path_display})  
        else:  
            queries = []  
            search_files = []  
            context = ""  
  
        history_to_send = session.get("history_to_send", MAX_HISTORY_TO_SEND)  
        input_items, target_user_index = build_responses_input_with_history(  
            all_messages=messages,  
            system_message=system_message,  
            context_text=context,  
            max_history_to_send=history_to_send,  
            wrap_context_with_markers=True  
        )  
  
        if target_user_index is None:  
            for i in range(len(input_items) - 1, -1, -1):  
                if input_items[i].get("role") == "user":  
                    target_user_index = i  
                    break  
  
        if target_user_index is not None:  
            image_filenames = session.get("image_filenames", [])  
            for img_blob in image_filenames:  
                if not image_container_client:  
                    continue  
                try:  
                    blob_client = image_container_client.get_blob_client(img_blob)  
                    props = blob_client.get_blob_properties()  
                    if props.size and props.size > MAX_ATTACHMENT_BYTES:  
                        print(f"画像が大きすぎるため添付スキップ: {img_blob} ({props.size} bytes)")  
                        continue  
                    sas_url = generate_sas_url(blob_client, img_blob)  
                    input_items[target_user_index]["content"].append({  
                        "type": "input_image",  
                        "image_url": sas_url  
                    })  
                except Exception as e:  
                    print("画像添付URL生成エラー:", e)  
                    traceback.print_exc()  
  
            file_filenames = session.get("file_filenames", [])  
            for pdf_blob in file_filenames:  
                if not pdf_blob.lower().endswith('.pdf') or not file_container_client:  
                    continue  
                try:  
                    blob_client = file_container_client.get_blob_client(pdf_blob)  
                    props = blob_client.get_blob_properties()  
                    if props.size and props.size > MAX_ATTACHMENT_BYTES:  
                        print(f"PDFが大きすぎるため添付スキップ: {pdf_blob} ({props.size} bytes)")  
                        continue  
                    pdf_b64 = encode_pdf_from_blob(blob_client)  
                    filename = pdf_blob.split("/")[-1]  
                    display = filename.split("__", 1)[1] if "__" in filename else filename  
                    input_items[target_user_index]["content"].append({  
                        "type": "input_file",  
                        "filename": display,  
                        "file_data": f"data:application/pdf;base64,{pdf_b64}",  
                    })  
                except Exception as e:  
                    print("PDF添付Base64生成エラー:", e)  
                    traceback.print_exc()  
  
        model_to_use = session.get("selected_model", RESPONSES_MODEL)  
        request_kwargs = dict(model=model_to_use, input=input_items, store=False)  
        if model_to_use in REASONING_ENABLED_MODELS:  
            effort = session.get("reasoning_effort", REASONING_EFFORT)  
            if effort not in {"low", "medium", "high"}:  
                effort = REASONING_EFFORT  
            request_kwargs["reasoning"] = {"effort": effort}  
  
        response = client.responses.create(**request_kwargs)  
        output_text = extract_output_text(response)  
        assistant_html = markdown2.markdown(  
            output_text or "",  
            extras=["tables", "fenced-code-blocks", "code-friendly", "break-on-newline", "cuddled-lists"]  
        ) or "<p>（応答テキストが空でした。もう一度お試しください）</p>"  
        reasoning_summary = extract_reasoning_summary(response)  
  
        ragas_scores = {}  
        try:  
            contexts_for_eval = [sf.get("content", "") for sf in (search_files or []) if isinstance(sf.get("content"), str)]  
            ragas_scores = compute_ragas_metrics(prompt, contexts_for_eval, output_text)  
        except Exception as e:  
            print("RAGAS compute error:", e)  
  
        persist_assistant_message(  
            active_sid=active_sid,  
            assistant_html=assistant_html,  
            full_text=output_text,  
            ragas_scores=ragas_scores,  
            system_message=system_message  
        )  
  
        return (  
            json.dumps({  
                'response': assistant_html,  
                'search_files': search_files,  
                'reasoning_summary': reasoning_summary,  
                'rewritten_queries': queries,  
                'ragas': ragas_scores,  
                'session_id': active_sid  
            }),  
            200,  
            {'Content-Type': 'application/json'}  
        )  
    except Exception as e:  
        print("チャット応答エラー:", e)  
        traceback.print_exc()  
        flash(f"エラーが発生しました: {e}", "error")  
        return (  
            json.dumps({  
                'response': f"エラーが発生しました: {e}",  
                'search_files': [],  
                'reasoning_summary': '',  
                'rewritten_queries': [],  
                'ragas': {},  
                'session_id': active_sid  
            }),  
            500,  
            {'Content-Type': 'application/json'}  
        )  
  
def _sse_event(event_name: str, data_obj) -> str:  
    return f"event: {event_name}\ndata: {json.dumps(data_obj, ensure_ascii=False)}\n\n"  
  
@app.route('/stream_message', methods=['GET'])  
def stream_message():  
    mid = (request.args.get('mid') or request.args.get('message_id') or '').strip()  
    prompt = (request.args.get('prompt') or '').strip()  
    if not prompt and mid:  
        prepared = session.get('prepared_prompts', {})  
        prompt = (prepared.get(mid) or '').strip()  
        if mid in prepared:  
            del prepared[mid]  
            session['prepared_prompts'] = prepared  
            session.modified = True  
    if not prompt:  
        return ("missing prompt", 400, {"Content-Type": "text/plain; charset=utf-8"})  
  
    # 開始時点のアクティブ会話IDを安定取得  
    active_sid = session.get("current_session_id")  
    if not active_sid:  
        sb = session.get("sidebar_messages", [])  
        if sb:  
            idx = session.get("current_chat_index", 0)  
            active_sid = sb[idx].get("session_id")  
            session["current_session_id"] = active_sid  
        else:  
            start_new_chat()  
            active_sid = session.get("current_session_id")  
  
    # Cosmos から復元（セッションが空の場合）  
    messages = session.get("main_chat_messages", [])  
    if not messages and container and active_sid:  
        messages = ensure_messages_from_cosmos(active_sid)  
        session["main_chat_messages"] = messages  
        session.modified = True  
  
    messages.append({"role": "user", "content": prompt, "type": "text"})  
    session["main_chat_messages"] = messages  
    session.modified = True  
    save_chat_history()  
  
    selected_index = session.get("selected_search_index", DEFAULT_SEARCH_INDEX)  
    doc_count = max(1, min(300, int(session.get("doc_count", DEFAULT_DOC_COUNT))))  
    history_to_send = session.get("history_to_send", MAX_HISTORY_TO_SEND)  
  
    system_message = session.get("default_system_message", "")  
    sb_for_sys = session.get("sidebar_messages", [])  
    if active_sid:  
        match_chat = next((c for c in sb_for_sys if c.get("session_id") == active_sid), None)  
        if match_chat:  
            system_message = match_chat.get("system_message", system_message)  
    else:  
        idx = session.get("current_chat_index", 0)  
        if sb_for_sys and 0 <= idx < len(sb_for_sys):  
            system_message = sb_for_sys[idx].get("system_message", system_message)  
  
    image_filenames = list(session.get("image_filenames", []))  
    file_filenames = list(session.get("file_filenames", []))  
  
    model_to_use = session.get("selected_model", RESPONSES_MODEL)  
    effort = session.get("reasoning_effort", REASONING_EFFORT)  
    if effort not in {"low", "medium", "high"}:  
        effort = REASONING_EFFORT  
    enable_reasoning = model_to_use in REASONING_ENABLED_MODELS  
    rag_enabled = bool(session.get("rag_enabled", True))  
  
    q = queue.Queue(maxsize=100)  
    done_event = threading.Event()  
    result_holder = {  
        "full_text": "",  
        "assistant_html": "",  
        "reasoning_summary": "",  
        "search_files": [],  
        "error": None,  
        "ragas_scores": {},  
        "persisted": False  
    }  
  
    @copy_current_request_context  
    def producer():  
        try:  
            search_files = []  
            context = ""  
            queries = []  
  
            if rag_enabled:  
                queries = rewrite_queries_for_search_responses(messages, prompt, system_message)  
                q.put(_sse_event("rewritten_queries", {"queries": queries}))  
  
                strictness = 0.0  
                results_list = hybrid_search_multiqueries(queries, doc_count, selected_index, strictness)  
  
                threshold = max(MIN_UNIQUE_PARENTS_ABS, int(doc_count * RECALL_PARENT_THRESHOLD_FRACTION))  
                if ENABLE_HYDE and unique_parents(results_list) < threshold:  
                    hyde_doc = hyde_paragraph(prompt)  
                    hyde_vec_results = keyword_vector_search(hyde_doc, doc_count, selected_index)  
                    results_list = rrf_fuse_ranked_lists([results_list, hyde_vec_results], doc_count)  
  
                if ENABLE_PRF and unique_parents(results_list) < threshold and queries:  
                    top_titles = [r.get("title", "") for r in results_list[:8]]  
                    prf_q = refine_query_with_prf(queries[0], top_titles)  
                    queries2 = list(dict.fromkeys([prf_q] + queries))[:2]  
                    results_list = hybrid_search_multiqueries(queries2, doc_count, selected_index, strictness)  
                    queries = queries2  
                    q.put(_sse_event("rewritten_queries", {"queries": queries}))  
  
                context = "\n".join([  
                    f"ファイル名: {r.get('title', '不明')}\n内容: {r.get('content','')}"  
                    for r in results_list  
                ])[:50000]  
  
                for r in results_list:  
                    title = r.get('title', '不明')  
                    content = r.get('content', '')  
                    fp = (r.get('filepath') or '').replace('\\', '/')  
                    container_name, blobname = (None, None)  
                    if fp:  
                        container_name, blobname = resolve_blob_from_filepath(selected_index, fp)  
                    else:  
                        u = r.get('url') or ''  
                        if is_azure_blob_url(u):  
                            container_name, blobname = resolve_blob_from_filepath(selected_index, u)  
                    url = ''  
                    if container_name and blobname:  
                        try:  
                            if blobname.lower().endswith('.txt'):  
                                url = url_for('download_txt', container=container_name, blobname=quote(blobname))  
                            else:  
                                blob_client = blob_service_client.get_blob_client(container=container_name, blob=blobname)  
                                url = generate_sas_url(blob_client, blobname)  
                        except Exception as e:  
                            print("SAS/ダウンロードURL生成エラー:", e)  
                    path_display = fp or (f"{container_name}/{blobname}" if (container_name and blobname) else (r.get('url') or ''))  
                    search_files.append({'title': title, 'content': content, 'url': url, 'filepath': path_display})  
                result_holder["search_files"] = search_files  
                q.put(_sse_event("search_files", search_files))  
            else:  
                queries = []  
                search_files = []  
                context = ""  
  
            input_items, target_user_index = build_responses_input_with_history(  
                all_messages=messages,  
                system_message=system_message,  
                context_text=context,  
                max_history_to_send=history_to_send,  
                wrap_context_with_markers=True  
            )  
  
            if target_user_index is None:  
                for i in range(len(input_items) - 1, -1, -1):  
                    if input_items[i].get("role") == "user":  
                        target_user_index = i  
                        break  
  
            if target_user_index is not None:  
                for img_blob in image_filenames:  
                    if not image_container_client:  
                        continue  
                    try:  
                        blob_client = image_container_client.get_blob_client(img_blob)  
                        props = blob_client.get_blob_properties()  
                        if props.size and props.size > MAX_ATTACHMENT_BYTES:  
                            print(f"画像が大きすぎるため添付スキップ: {img_blob} ({props.size} bytes)")  
                            continue  
                        sas_url = generate_sas_url(blob_client, img_blob)  
                        input_items[target_user_index]["content"].append({  
                            "type": "input_image",  
                            "image_url": sas_url  
                        })  
                    except Exception as e:  
                        print("画像添付URL生成エラー:", e)  
                        traceback.print_exc()  
  
                for pdf_blob in file_filenames:  
                    if not pdf_blob.lower().endswith('.pdf') or not file_container_client:  
                        continue  
                    try:  
                        blob_client = file_container_client.get_blob_client(pdf_blob)  
                        props = blob_client.get_blob_properties()  
                        if props.size and props.size > MAX_ATTACHMENT_BYTES:  
                            print(f"PDFが大きすぎるため添付スキップ: {pdf_blob} ({props.size} bytes)")  
                            continue  
                        pdf_b64 = encode_pdf_from_blob(blob_client)  
                        filename = pdf_blob.split("/")[-1]  
                        display = filename.split("__", 1)[1] if "__" in filename else filename  
                        input_items[target_user_index]["content"].append({  
                            "type": "input_file",  
                            "filename": display,  
                            "file_data": f"data:application/pdf;base64,{pdf_b64}",  
                        })  
                    except Exception as e:  
                        print("PDF添付Base64生成エラー:", e)  
                        traceback.print_exc()  
  
            request_kwargs = dict(model=model_to_use, input=input_items, store=False)  
            if enable_reasoning:  
                request_kwargs["reasoning"] = {"effort": effort}  
  
            with client.responses.stream(**request_kwargs) as stream:  
                for event in stream:  
                    etype = getattr(event, "type", "")  
                    if etype == "response.output_text.delta":  
                        delta = getattr(event, "delta", "") or ""  
                        if isinstance(delta, str) and delta:  
                            result_holder["full_text"] += delta  
                            q.put(_sse_event("delta", {"text": delta}))  
                    elif etype.endswith(".delta"):  
                        delta = getattr(event, "delta", "")  
                        if isinstance(delta, str) and delta:  
                            result_holder["full_text"] += delta  
                            q.put(_sse_event("delta", {"text": delta}))  
                    elif etype == "response.error":  
                        err = getattr(event, "error", None)  
                        msg = str(err) if err else "unknown error"  
                        result_holder["error"] = msg  
                        q.put(_sse_event("error", {"message": msg}))  
                final_response = stream.get_final_response()  
  
            if not result_holder["full_text"]:  
                try:  
                    final_text = extract_output_text(final_response) or ""  
                    if final_text:  
                        result_holder["full_text"] = final_text  
                        q.put(_sse_event("delta", {"text": final_text}))  
                except Exception as e:  
                    print("final_response からの出力抽出失敗:", e)  
  
            result_holder["reasoning_summary"] = extract_reasoning_summary(final_response)  
            full_text = result_holder["full_text"]  
            assistant_html = markdown2.markdown(  
                full_text or "",  
                extras=["tables", "fenced-code-blocks", "code-friendly", "break-on-newline", "cuddled-lists"]  
            ) or "<p>（応答テキストが空でした。もう一度お試しください）</p>"  
            result_holder["assistant_html"] = assistant_html  
  
            if result_holder["reasoning_summary"]:  
                q.put(_sse_event("reasoning_summary", {"summary": result_holder["reasoning_summary"]}))  
  
            if ENABLE_RAGAS and RAGAS_AVAILABLE and result_holder["full_text"]:  
                try:  
                    contexts = [sf.get("content", "") for sf in (result_holder.get("search_files") or []) if isinstance(sf.get("content"), str)]  
                    ragas_scores = compute_ragas_metrics(prompt, contexts, result_holder["full_text"])  
                    result_holder["ragas_scores"] = ragas_scores  
                    if ragas_scores:  
                        q.put(_sse_event("ragas_metrics", ragas_scores))  
                except Exception as e:  
                    print("RAGAS SSE error:", e)  
  
            q.put(_sse_event("final", {"html": assistant_html, "session_id": active_sid}))  
  
            try:  
                persist_assistant_message(  
                    active_sid=active_sid,  
                    assistant_html=assistant_html,  
                    full_text=result_holder["full_text"],  
                    ragas_scores=result_holder.get("ragas_scores", {}),  
                    system_message=system_message  
                )  
                result_holder["persisted"] = True  
            except Exception as e:  
                print("producer 内の履歴保存エラー（保険）:", e)  
  
        except Exception as e:  
            print("SSE producer エラー:", e)  
            traceback.print_exc()  
            result_holder["error"] = str(e)  
            q.put(_sse_event("error", {"message": str(e)}))  
        finally:  
            q.put(_sse_event("done", {}))  
            q.put(None)  
            done_event.set()  
  
    def heartbeat():  
        while not done_event.is_set():  
            try:  
                q.put(":keepalive\n\n")  
            except Exception:  
                break  
            time.sleep(12)  
  
    threading.Thread(target=producer, daemon=True).start()  
    threading.Thread(target=heartbeat, daemon=True).start()  
  
    def generate():  
        yield ":connected\n\n"  
        while True:  
            chunk = q.get()  
            if chunk is None:  
                break  
            yield chunk  
        try:  
            if (result_holder["assistant_html"] or result_holder["full_text"]) and not result_holder.get("persisted"):  
                persist_assistant_message(  
                    active_sid=active_sid,  
                    assistant_html=result_holder["assistant_html"],  
                    full_text=result_holder["full_text"],  
                    ragas_scores=result_holder.get("ragas_scores", {}),  
                    system_message=system_message  
                )  
        except Exception as e:  
            print("SSE後処理の履歴保存エラー:", e)  
            traceback.print_exc()  
  
    headers = {  
        "Content-Type": "text/event-stream; charset=utf-8",  
        "Cache-Control": "no-cache, no-transform",  
        "X-Accel-Buffering": "no",  
        "Connection": "keep-alive"  
    }  
    return app.response_class(stream_with_context(generate()), headers=headers)  
  
# ------------------------------- テキスト Blob ダウンロード -------------------------------  
@app.route("/download_txt/<container>/<path:blobname>")  
def download_txt(container, blobname):  
    if not blob_service_client:  
        return ("Blob service not configured", 500)  
    blobname = unquote(blobname)  
    blob_client = blob_service_client.get_blob_client(container=container, blob=blobname)  
    txt_bytes = blob_client.download_blob().readall()  
    try:  
        txt_str = txt_bytes.decode("utf-8")  
    except UnicodeDecodeError:  
        txt_str = txt_bytes.decode("cp932", errors="ignore")  
    bom = b'\xef\xbb\xbf'  
    buf = io.BytesIO(bom + txt_str.encode("utf-8"))  
    filename = os.path.basename(blobname)  
    ascii_filename = "download.txt"  
    response = send_file(  
        buf,  
        as_attachment=True,  
        download_name=ascii_filename,  
        mimetype="text/plain; charset=utf-8"  
    )  
    response.headers["Content-Disposition"] = (  
        f'attachment; filename="{ascii_filename}"; filename*=UTF-8\'\'{quote(filename)}'  
    )  
    return response  
  
# ------------------------------- エントリポイント -------------------------------  
if __name__ == '__main__':  
    app.run(debug=True, host='0.0.0.0')  