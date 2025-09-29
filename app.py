#!/usr/bin/env python3  # -*- coding: utf-8 -*-  
"""  
PM Compass – Flask アプリ（PDFアップロード＆分析対応版, 埋め込み分離 + RRFオーバーフェッチ対応 + SSEストリーミング対応, シンプルクエリリライト版）  
- 画像/PDFアップロード対応（input_image / input_file）  
- セッション終了時の自動削除（sendBeacon + /cleanup_session_files）  
- Azure Cognitive Search ハイブリッド検索（RRF融合）  
- Azure OpenAI Responses API（ストリーミング対応）  
- PDF は file_data(Base64) + filename 方式で添付（file_url は使用しない）  
- アプリ側のステップ制御（Step1/Step2判定）を廃止し、LLM側のプロンプト判断に委ねる  
  
変更点  
- サイドバー設定のAJAX更新 /update_settings を追加  
- RAG ON/OFF をセッションで管理（rag_enabled）  
- RAG OFF時は検索/リライト/HyDE/PRFをスキップして会話＋添付のみ  
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
  
REWRITE_MODEL = "gpt-4o"  # 固定  
MAX_REWRITE_TURNS = max(1, min(8, int(os.getenv("MAX_REWRITE_TURNS", "4"))))  
  
ENABLE_HYDE = os.getenv("ENABLE_HYDE", "1") not in ("0", "false", "False")  
ENABLE_PRF = os.getenv("ENABLE_PRF", "1") not in ("0", "false", "False")  
RECALL_PARENT_THRESHOLD_FRACTION = float(os.getenv("RECALL_PARENT_THRESHOLD_FRACTION", "0.4"))  
MIN_UNIQUE_PARENTS_ABS = int(os.getenv("MIN_UNIQUE_PARENTS_ABS", "3"))  
  
# 添付の最大バイト（超過時は添付スキップ）  
MAX_ATTACHMENT_BYTES = int(os.getenv("MAX_ATTACHMENT_BYTES", str(5 * 1024 * 1024)))  # 5MB  
  
# ------------------------------- Flask アプリ設定 -------------------------------  
app = Flask(__name__)  
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'your-default-secret-key')  
app.config['SESSION_TYPE'] = 'filesystem'  
app.config['SESSION_FILE_DIR'] = '/tmp/flask_session'  
app.config['SESSION_PERMANENT'] = False  
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
        connection_verify=verify_path,  # verify ではなく connection_verify  
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
    transport=transport  
) if blob_connection_string else None  
  
image_container_name = 'chatgpt-image'  
file_container_name = 'chatgpt-files'  
  
# コンテナ自動作成（既存ならスキップ）  
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
def rewrite_query(user_input: str, recent_messages: list) -> str:  
    prompt = (  
        "あなたは社内情報検索のためのアシスタントです。\n"  
        "以下の会話履歴とユーザーの最新質問をもとに、検索エンジンに適した簡潔で明確な日本語クエリを生成してください。\n"  
        "不要な会話表現や雑談は除外し、検索意図を正確に反映したキーワードやフレーズを含めてください。\n"  
        "【会話履歴】\n"  
    )  
    for msg in (recent_messages or []):  
        prompt += f"- {msg}\n"  
    prompt += f"【ユーザー質問】\n{user_input}\n"  
    prompt += "【検索用クエリ】"  
  
    input_items = [{"role": "user", "content": [{"type": "input_text", "text": prompt}]}]  
    resp = client.responses.create(  
        model=REWRITE_MODEL,  
        input=input_items,  
        temperature=0.5,  
        max_output_tokens=256,  
        store=False  # サーバー側に履歴を保存しない  
    )  
    rewritten = (extract_output_text(resp) or "").strip()  
    rewritten = rewritten.strip().strip('「」"\' 　')  
    return rewritten  
  
# ------------------------------- 認証/履歴/指示テンプレート -------------------------------  
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
  
def save_chat_history():  
    if not container:  
        return  
    with lock:  
        try:  
            sidebar = session.get("sidebar_messages", [])  
            idx = session.get("current_chat_index", 0)  
            if idx < len(sidebar):  
                current = sidebar[idx]  
                user_id = get_authenticated_user()  
                user_name = session.get("user_name", "anonymous")  
                session_id = current.get("session_id")  
                item = {  
                    'id': session_id,  
                    'user_id': user_id,  
                    'user_name': user_name,  
                    'session_id': session_id,  
                    'messages': current.get("messages", []),  
                    'system_message': current.get(  
                        "system_message", session.get("default_system_message", "あなたは親切なAIアシスタントです…")  
                    ),  
                    'first_assistant_message': current.get("first_assistant_message", ""),  
                    'timestamp': datetime.datetime.now(datetime.timezone.utc).isoformat()  
                }  
                container.upsert_item(item)  
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
                    chat = {  
                        "session_id": item['session_id'],  
                        "messages": item.get("messages", []),  
                        "system_message": item.get(  
                            "system_message", session.get('default_system_message', "あなたは親切なAIアシスタントです…")  
                        ),  
                        "first_assistant_message": item.get("first_assistant_message", "")  
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
        user_id = get_authenticated_user()  
        prompts = []  
        try:  
            query = """  
            SELECT c.id, c.title, c.content, c.timestamp  
            FROM c  
            WHERE c.user_id = @user_id AND c.doc_type = 'system_prompt'  
            ORDER BY c.timestamp DESC  
            """  
            parameters = [{"name": "@user_id", "value": user_id}]  
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
    # 画像削除  
    image_filenames = session.get("image_filenames", [])  
    for img_name in image_filenames:  
        if image_container_client:  
            try:  
                image_container_client.get_blob_client(img_name).delete_blob()  
            except Exception as e:  
                print("画像削除エラー:", e)  
    session["image_filenames"] = []  
  
    # PDF削除  
    file_filenames = session.get("file_filenames", [])  
    for file_name in file_filenames:  
        if file_container_client:  
            try:  
                file_container_client.get_blob_client(file_name).delete_blob()  
            except Exception as e:  
                print("ファイル削除エラー:", e)  
    session["file_filenames"] = []  
  
    new_session_id = str(uuid.uuid4())  
    new_chat = {  
        "session_id": new_session_id,  
        "messages": [],  
        "first_assistant_message": "",  
        "system_message": session.get('default_system_message', "あなたは親切なAIアシスタントです…"),  
    }  
    sidebar = session.get("sidebar_messages", [])  
    sidebar.insert(0, new_chat)  
    session["sidebar_messages"] = sidebar  
    session["current_chat_index"] = 0  
    session["main_chat_messages"] = []  
  
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
  
# ------------------------------- フォールバック: HyDE / PRF（Responses API） -------------------------------  
def hyde_paragraph(user_input: str) -> str:  
    sys_msg = "あなたは検索のための仮想文書(HyDE)を作成します。事実の付け足しは避け、中立で。"  
    p = f"""以下の質問に関して、中立で百科事典調の根拠段落を日本語で5-7文で作成してください。事実断定は避け、関連する用語を豊富に含めてください。これは検索用の仮想文書であり回答ではありません。  質問: {user_input}"""  
    input_items = [  
        {"role": "system", "content": [{"type": "input_text", "text": sys_msg}]},  
        {"role": "user", "content": [{"type": "input_text", "text": p}]},  
    ]  
    resp = client.responses.create(  
        model=RESPONSES_MODEL,  
        input=input_items,  
        temperature=0.2,  
        max_output_tokens=512,  
        store=False  # サーバー側に履歴を保存しない  
    )  
    return (extract_output_text(resp) or "").strip()  
  
def refine_query_with_prf(initial_query: str, titles: list) -> str:  
    t = "\n".join(f"- {x}" for x in (titles or [])[:8])  
    prompt = f"""初回検索の上位文書のタイトル一覧です。これを参考に、より適合度の高い検索クエリを日本語で1本だけ生成。不要語は削除し、重要語は維持。出力はクエリ文字列のみ。  タイトル:  {t}    初回クエリ: {initial_query}"""  
    input_items = [  
        {"role": "system", "content": [{"type": "input_text", "text": "あなたは検索クエリの改良を行うプロフェッショナルです。"}]},  
        {"role": "user", "content": [{"type": "input_text", "text": prompt}]},  
    ]  
    resp = client.responses.create(  
        model=RESPONSES_MODEL,  
        input=input_items,  
        temperature=0,  
        max_output_tokens=256,  
        store=False  # サーバー側に履歴を保存しない  
    )  
    return (extract_output_text(resp) or "").strip().strip('「」\' ')  
  
def unique_parents(results):  
    return len({(r.get("parent_id") or r.get("filepath") or r.get("title")) for r in (results or [])})  
  
# ------------------------------- Responses API 入力を構築（system→history→直近userにコンテキスト追記） -------------------------------  
def build_responses_input_with_history(  
    all_messages,  
    system_message,  
    context_text,  
    max_history_to_send=None,  
    wrap_context_with_markers=True,  
    context_title="参考コンテキスト（RAG）",  
):  
    """  
    入力構築ルール:  
    - system を最初に  
    - 履歴（user/assistant）を時系列でそのまま詰める（改変しない）  
    - context_text があれば、末尾に『コンテキスト専用の user メッセージ』は作らず、  
      「直近の実ユーザー発話」に input_text として追記する（承認/指示を直近として維持）  
    - BEGIN_CONTEXT/END_CONTEXT で囲む（任意）  
    - 添付（input_image/input_file）はこの『直近の実ユーザー発話』に append する  
    戻り値:  
    (input_items, target_user_index)  # target_user_index は『直近の実ユーザー発話』。存在しない場合は None。  
    """  
    input_items = []  
  
    # 1) system  
    if system_message:  
        input_items.append({  
            "role": "system",  
            "content": [{"type": "input_text", "text": system_message}]  
        })  
  
    # 2) 履歴（時系列のまま）  
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
  
    # 3) 直近の『実ユーザー発話』を特定  
    last_user_index = None  
    for i in range(len(input_items) - 1, -1, -1):  
        if input_items[i].get("role") == "user":  
            last_user_index = i  
            break  
  
    # 4) コンテキスト追記（直近のユーザー発話に append。なければレアケースとして新規userを作成）  
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
            # 直近ユーザー発話が存在しない場合のみ、新規userとして追加（稀ケース）  
            input_items.append({  
                "role": "user",  
                "content": [{"type": "input_text", "text": ctx}]  
            })  
            target_user_index = len(input_items) - 1  
  
    return input_items, target_user_index  
  
# ------------------------------- セッション内アップロードの一括削除 + エンドポイント -------------------------------  
def _delete_uploaded_files_for_session():  
    deleted = {"images": [], "files": []}  
    try:  
        # 画像  
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
        # PDF  
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
        session["image_filenames"] = []  
        session["file_filenames"] = []  
        session.modified = True  
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
  
    # モデル  
    if "selected_model" in data:  
        selected = (data.get("selected_model") or "").strip()  
        allowed_models = {"gpt-4o", "gpt-4.1", "o3", "o4-mini", "gpt-5"}  
        if selected in allowed_models:  
            session["selected_model"] = selected  
            changed["selected_model"] = selected  
  
    # reasoning_effort（対応モデル時のみ）  
    if "reasoning_effort" in data:  
        effort = (data.get("reasoning_effort") or "").strip().lower()  
        allowed_efforts = {"low", "medium", "high"}  
        if session.get("selected_model") in REASONING_ENABLED_MODELS and effort in allowed_efforts:  
            session["reasoning_effort"] = effort  
            changed["reasoning_effort"] = effort  
  
    # 検索インデックス  
    if "selected_search_index" in data:  
        sel_index = (data.get("selected_search_index") or "").strip()  
        if sel_index in INDEX_VALUES:  
            session["selected_search_index"] = sel_index  
            changed["selected_search_index"] = sel_index  
  
    # ドキュメント数  
    if "doc_count" in data:  
        try:  
            val = int(data.get("doc_count"))  
        except Exception:  
            val = DEFAULT_DOC_COUNT  
        session["doc_count"] = max(1, min(300, val))  
        changed["doc_count"] = session["doc_count"]  
  
    # 履歴数  
    if "history_to_send" in data:  
        try:  
            val = int(data.get("history_to_send"))  
        except Exception:  
            val = MAX_HISTORY_TO_SEND  
        session["history_to_send"] = max(1, min(50, val))  
        changed["history_to_send"] = session["history_to_send"]  
  
    # システムメッセージ  
    if "default_system_message" in data:  
        sys_msg = (data.get("default_system_message") or "").strip()  
        session["default_system_message"] = sys_msg  
        idx = session.get("current_chat_index", 0)  
        sidebar = session.get("sidebar_messages", [])  
        if sidebar and idx < len(sidebar):  
            sidebar[idx]["system_message"] = sys_msg  
            session["sidebar_messages"] = sidebar  
        changed["default_system_message"] = sys_msg  
  
    # RAG ON/OFF  
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
    if "sidebar_messages" not in session:  
        session["sidebar_messages"] = load_chat_history() or []  
    if "current_chat_index" not in session:  
        start_new_chat()  
    if "main_chat_messages" not in session:  
        idx0 = session.get("current_chat_index", 0)  
        sb = session.get("sidebar_messages", [])  
        session["main_chat_messages"] = sb[idx0].get("messages", []) if (sb and idx0 < len(sb)) else []  
    if "image_filenames" not in session:  
        session["image_filenames"] = []  
    if "file_filenames" not in session:  
        session["file_filenames"] = []  
    if "saved_prompts" not in session:  
        session["saved_prompts"] = load_system_prompts()  
    if "show_all_history" not in session:  
        session["show_all_history"] = False  
    if "upload_prefix" not in session:  
        session["upload_prefix"] = str(uuid.uuid4())  # セッション専用プレフィックス  
    if "rag_enabled" not in session:  
        session["rag_enabled"] = True  
    session.modified = True  
  
    # POST ハンドラ（既存のフォーム動作は維持）  
    if request.method == 'POST':  
        if 'set_model' in request.form:  
            selected = (request.form.get("model") or "").strip()  
            allowed_models = {"gpt-4o", "gpt-4.1", "o3", "o4-mini", "gpt-5"}  
            if selected in allowed_models:  
                session["selected_model"] = selected  
            effort = (request.form.get("reasoning_effort") or "").strip().lower()  
            allowed_efforts = {"low", "medium", "high"}  
            if session.get("selected_model") in REASONING_ENABLED_MODELS and effort in allowed_efforts:  
                session["reasoning_effort"] = effort  
            session.modified = True  
            return redirect(url_for('index'))  
  
        if 'set_index' in request.form:  
            sel_index = (request.form.get("search_index") or "").strip()  
            if sel_index in INDEX_VALUES:  
                session["selected_search_index"] = sel_index  
            session.modified = True  
            return redirect(url_for('index'))  
  
        if 'set_doc_count' in request.form:  
            raw = (request.form.get("doc_count") or "").strip()  
            try:  
                val = int(raw)  
            except Exception:  
                val = DEFAULT_DOC_COUNT  
            session["doc_count"] = max(1, min(300, val))  
            session.modified = True  
            return redirect(url_for('index'))  
  
        if 'set_history_to_send' in request.form:  
            raw = (request.form.get("history_to_send") or "").strip()  
            try:  
                val = int(raw)  
            except Exception:  
                val = MAX_HISTORY_TO_SEND  
            session["history_to_send"] = max(1, min(50, val))  
            session.modified = True  
            return redirect(url_for('index'))  
  
        if 'set_system_message' in request.form:  
            sys_msg = (request.form.get("system_message") or "").strip()  
            session["default_system_message"] = sys_msg  
            idx = session.get("current_chat_index", 0)  
            sidebar = session.get("sidebar_messages", [])  
            if sidebar and idx < len(sidebar):  
                sidebar[idx]["system_message"] = sys_msg  
                session["sidebar_messages"] = sidebar  
            session.modified = True  
            return redirect(url_for('index'))  
  
        if 'add_system_prompt' in request.form:  
            title = (request.form.get("prompt_title") or "").strip()  
            content = (request.form.get("prompt_content") or "").strip()  
            if title and content:  
                save_system_prompt_item(title, content)  
                session["saved_prompts"] = load_system_prompts()  
                flash("指示を登録しました。", "info")  
            else:  
                flash("タイトルと指示内容を入力してください。", "error")  
            session.modified = True  
            return redirect(url_for('index'))  
  
        if 'apply_system_prompt' in request.form:  
            pid = request.form.get("select_prompt_id", "")  
            prompts = session.get("saved_prompts", [])  
            selected = next((p for p in prompts if p.get("id") == pid), None)  
            if selected:  
                sys_msg = selected.get("content", "")  
                session["default_system_message"] = sys_msg  
                idx = session.get("current_chat_index", 0)  
                sidebar = session.get("sidebar_messages", [])  
                if sidebar and idx < len(sidebar):  
                    sidebar[idx]["system_message"] = sys_msg  
                    session["sidebar_messages"] = sidebar  
                flash(f"『{selected.get('title','無題')}』を適用しました。", "info")  
            else:  
                flash("指示が見つかりません。", "error")  
            session.modified = True  
            return redirect(url_for('index'))  
  
        if 'delete_system_prompt' in request.form:  
            pid = request.form.get("delete_system_prompt", "")  
            if pid:  
                delete_system_prompt(pid)  
                session["saved_prompts"] = load_system_prompts()  
                flash("指示を削除しました。", "info")  
            session.modified = True  
            return redirect(url_for('index'))  
  
        if 'new_chat' in request.form:  
            start_new_chat()  
            session["show_all_history"] = False  
            session.modified = True  
            return redirect(url_for('index'))  
  
        if 'select_chat' in request.form:  
            selected_session = request.form.get("select_chat")  
            sidebar = session.get("sidebar_messages", [])  
            for idx, chat in enumerate(sidebar):  
                if chat.get("session_id") == selected_session:  
                    session["current_chat_index"] = idx  
                    session["main_chat_messages"] = chat.get("messages", [])  
                    session["default_system_message"] = chat.get("system_message", session.get("default_system_message"))  
                    break  
            session.modified = True  
            return redirect(url_for('index'))  
  
        if 'toggle_history' in request.form:  
            session["show_all_history"] = not session.get("show_all_history", False)  
            session.modified = True  
            return redirect(url_for('index'))  
  
        if 'upload_files' in request.form:  
            if 'files' in request.files:  
                files = request.files.getlist("files")  
                image_filenames = session.get("image_filenames", [])  
                file_filenames = session.get("file_filenames", [])  
                upload_prefix = session.get("upload_prefix") or str(uuid.uuid4())  
                session["upload_prefix"] = upload_prefix  
                for f in files:  
                    if f and f.filename != '':  
                        original = secure_filename(f.filename)  
                        ext = original.rsplit('.', 1)[-1].lower() if '.' in original else ''  
                        blob_name = f"{upload_prefix}/{uuid.uuid4()}__{original}"  
                        if ext in ['png', 'jpeg', 'jpg', 'gif']:  
                            if image_container_client:  
                                blob_client = image_container_client.get_blob_client(blob_name)  
                                try:  
                                    f.stream.seek(0)  
                                    blob_client.upload_blob(f.stream, overwrite=True)  
                                    if blob_name not in image_filenames:  
                                        image_filenames.append(blob_name)  
                                except Exception as e:  
                                    print("画像アップロードエラー:", e)  
                                    flash("画像アップロードに失敗しました。", "error")  
                        elif ext == 'pdf':  
                            if file_container_client:  
                                blob_client = file_container_client.get_blob_client(blob_name)  
                                try:  
                                    f.stream.seek(0)  
                                    blob_client.upload_blob(f.stream, overwrite=True)  
                                    if blob_name not in file_filenames:  
                                        file_filenames.append(blob_name)  
                                except Exception as e:  
                                    print("PDFアップロードエラー:", e)  
                                    flash("PDFアップロードに失敗しました。", "error")  
                session["image_filenames"] = image_filenames  
                session["file_filenames"] = file_filenames  
                session.modified = True  
            return redirect(url_for('index'))  
  
        if 'delete_image' in request.form:  
            delete_blobname = request.form.get("delete_image")  
            image_filenames = [n for n in session.get("image_filenames", []) if n != delete_blobname]  
            if image_container_client and delete_blobname:  
                try:  
                    bc = image_container_client.get_blob_client(delete_blobname)  
                    if bc.exists():  
                        bc.delete_blob(delete_snapshots="include")  
                except Exception as e:  
                    print("画像削除エラー(無視可):", e)  
            session["image_filenames"] = image_filenames  
            session.modified = True  
            return redirect(url_for('index'))  
  
        if 'delete_file' in request.form:  
            delete_blobname = request.form.get("delete_file")  
            file_filenames = [n for n in session.get("file_filenames", []) if n != delete_blobname]  
            if file_container_client and delete_blobname:  
                try:  
                    bc = file_container_client.get_blob_client(delete_blobname)  
                    if bc.exists():  
                        bc.delete_blob(delete_snapshots="include")  
                except Exception as e:  
                    print("ファイル削除エラー(無視可):", e)  
            session["file_filenames"] = file_filenames  
            session.modified = True  
            return redirect(url_for('index'))  
  
    # 描画データ作成（SAS URLを発行）  
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
  
# ------------------------------- 長文を事前登録する準備エンドポイント -------------------------------  
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
  
# ------------------------------- メッセージ送信（非SSE） -------------------------------  
@app.route('/send_message', methods=['POST'])  
def send_message():  
    data = request.get_json()  
    prompt = (data.get('prompt') or '').strip()  
    if not prompt:  
        return (  
            json.dumps({'response': '', 'search_files': [], 'reasoning_summary': '', 'rewritten_queries': []}),  
            400,  
            {'Content-Type': 'application/json'}  
        )  
  
    messages = session.get("main_chat_messages", [])  
    messages.append({"role": "user", "content": prompt, "type": "text"})  
    session["main_chat_messages"] = messages  
    session.modified = True  
    save_chat_history()  
  
    try:  
        selected_index = session.get("selected_search_index", DEFAULT_SEARCH_INDEX)  
        doc_count = max(1, min(300, int(session.get("doc_count", DEFAULT_DOC_COUNT))))  
        rag_enabled = bool(session.get("rag_enabled", True))  
  
        system_message = session.get("default_system_message", "")  
        idx = session.get("current_chat_index", 0)  
        sidebar = session.get("sidebar_messages", [])  
        if sidebar and 0 <= idx < len(sidebar):  
            system_message = sidebar[idx].get("system_message", system_message)  
  
        # RAG分岐  
        queries = []  
        search_files = []  
        context = ""  
  
        def _to_text(m):  
            if m.get("role") == "assistant":  
                return m.get("text") or strip_html_tags(m.get("content", ""))  
            return m.get("content", "")  
  
        if rag_enabled:  
            turns = max(1, min(MAX_REWRITE_TURNS, len(messages)))  
            recent_texts = [_to_text(m) for m in messages[-turns:]]  
            rq = rewrite_query(prompt, recent_texts)  
            queries = [rq or prompt]  
  
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
  
            # 検索ファイル一覧を生成  
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
                search_files.append({'title': title, 'content': content, 'url': url})  
        else:  
            # RAG OFF: 検索もリライトも行わず、会話＋添付のみ  
            queries = []  
            search_files = []  
            context = ""  
  
        history_to_send = session.get("history_to_send", MAX_HISTORY_TO_SEND)  
  
        # Responses API 入力構築（system→history→直近userにコンテキスト追記）  
        input_items, target_user_index = build_responses_input_with_history(  
            all_messages=messages,  
            system_message=system_message,  
            context_text=context,  
            max_history_to_send=history_to_send,  
            wrap_context_with_markers=True  
        )  
  
        # 添付は『直近の実ユーザー発話』へ（context_text が空ならフォールバックで最後の通常 user）  
        if target_user_index is None:  
            for i in range(len(input_items) - 1, -1, -1):  
                if input_items[i].get("role") == "user":  
                    target_user_index = i  
                    break  
  
        if target_user_index is not None:  
            # 画像  
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
  
            # PDF  
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
  
        messages.append({"role": "assistant", "content": assistant_html, "type": "html", "text": output_text})  
        session["main_chat_messages"] = messages  
        session.modified = True  
  
        idx = session.get("current_chat_index", 0)  
        sidebar = session.get("sidebar_messages", [])  
        if idx < len(sidebar):  
            sidebar[idx]["messages"] = messages  
            if not sidebar[idx].get("first_assistant_message"):  
                sidebar[idx]["first_assistant_message"] = output_text  
            session["sidebar_messages"] = sidebar  
            session.modified = True  
  
        save_chat_history()  
  
        return (  
            json.dumps({  
                'response': assistant_html,  
                'search_files': search_files,  
                'reasoning_summary': reasoning_summary,  
                'rewritten_queries': queries  
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
                'rewritten_queries': []  
            }),  
            500,  
            {'Content-Type': 'application/json'}  
        )  
  
# ------------------------------- SSE ストリーミング ヘルパー -------------------------------  
def _sse_event(event_name: str, data_obj) -> str:  
    return f"event: {event_name}\ndata: {json.dumps(data_obj, ensure_ascii=False)}\n\n"  
  
# ------------------------------- SSE ストリーミング エンドポイント -------------------------------  
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
  
    messages = session.get("main_chat_messages", [])  
    messages.append({"role": "user", "content": prompt, "type": "text"})  
    session["main_chat_messages"] = messages  
    session.modified = True  
    save_chat_history()  
  
    selected_index = session.get("selected_search_index", DEFAULT_SEARCH_INDEX)  
    doc_count = max(1, min(300, int(session.get("doc_count", DEFAULT_DOC_COUNT))))  
    history_to_send = session.get("history_to_send", MAX_HISTORY_TO_SEND)  
  
    system_message = session.get("default_system_message", "")  
    idx = session.get("current_chat_index", 0)  
    sidebar = session.get("sidebar_messages", [])  
    if sidebar and 0 <= idx < len(sidebar):  
        system_message = sidebar[idx].get("system_message", system_message)  
  
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
    result_holder = {"full_text": "", "assistant_html": "", "reasoning_summary": "", "search_files": [], "error": None}  
  
    @copy_current_request_context  
    def producer():  
        try:  
            search_files = []  
            context = ""  
            queries = []  
  
            if rag_enabled:  
                def _to_text(m):  
                    if m.get("role") == "assistant":  
                        return m.get("text") or strip_html_tags(m.get("content", ""))  
                    return m.get("content", "")  
  
                turns = max(1, min(MAX_REWRITE_TURNS, len(messages)))  
                recent_texts = [_to_text(m) for m in messages[-turns:]]  
                rq = rewrite_query(prompt, recent_texts)  
                queries = [rq or prompt]  
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
  
                # 検索ファイル一覧  
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
                    search_files.append({'title': title, 'content': content, 'url': url})  
                result_holder["search_files"] = search_files  
                q.put(_sse_event("search_files", search_files))  
            else:  
                # RAG OFF: 検索もリライトも行わず、会話＋添付のみ  
                queries = []  
                search_files = []  
                context = ""  
  
            # 入力構築（system→history→直近userにコンテキスト追記）  
            input_items, target_user_index = build_responses_input_with_history(  
                all_messages=messages,  
                system_message=system_message,  
                context_text=context,  
                max_history_to_send=history_to_send,  
                wrap_context_with_markers=True  
            )  
  
            # 添付付与（直近ユーザーへ。context_text が空なら最後の通常 user にフォールバック）  
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
                        if delta:  
                            result_holder["full_text"] += delta  
                            q.put(_sse_event("delta", {"text": delta}))  
                    elif etype == "response.error":  
                        err = getattr(event, "error", None)  
                        msg = str(err) if err else "unknown error"  
                        result_holder["error"] = msg  
                        q.put(_sse_event("error", {"message": msg}))  
                final_response = stream.get_final_response()  
  
            result_holder["reasoning_summary"] = extract_reasoning_summary(final_response)  
            full_text = result_holder["full_text"]  
            assistant_html = markdown2.markdown(  
                full_text or "",  
                extras=["tables", "fenced-code-blocks", "code-friendly", "break-on-newline", "cuddled-lists"]  
            ) or "<p>（応答テキストが空でした。もう一度お試しください）</p>"  
            result_holder["assistant_html"] = assistant_html  
  
            if result_holder["reasoning_summary"]:  
                q.put(_sse_event("reasoning_summary", {"summary": result_holder["reasoning_summary"]}))  
            q.put(_sse_event("final", {"html": assistant_html}))  
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
            if result_holder["assistant_html"] or result_holder["full_text"]:  
                msgs = session.get("main_chat_messages", [])  
                msgs.append({  
                    "role": "assistant",  
                    "content": result_holder["assistant_html"],  
                    "type": "html",  
                    "text": result_holder["full_text"]  
                })  
                session["main_chat_messages"] = msgs  
                session.modified = True  
                idx2 = session.get("current_chat_index", 0)  
                sidebar2 = session.get("sidebar_messages", [])  
                if idx2 < len(sidebar2):  
                    sidebar2[idx2]["messages"] = msgs  
                    if not sidebar2[idx2].get("first_assistant_message"):  
                        sidebar2[idx2]["first_assistant_message"] = result_holder["full_text"]  
                    session["sidebar_messages"] = sidebar2  
                    session.modified = True  
                save_chat_history()  
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