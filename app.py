#!/usr/bin/env python3  
# -*- coding: utf-8 -*-  
"""  
PM Compass PRO – Flask アプリ（Entra ID 認証版）  
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
import mimetypes  
import math  
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
    abort,  
)  
from flask import copy_current_request_context  
from flask_session import Session  
from werkzeug.utils import secure_filename  
  
from azure.core.exceptions import ResourceNotFoundError, ClientAuthenticationError, HttpResponseError  
from azure.search.documents import SearchClient  
from azure.cosmos import CosmosClient  
from azure.storage.blob import BlobServiceClient, generate_blob_sas, BlobSasPermissions  
from openai import AzureOpenAI  
import markdown2  
from azure.identity import AzureCliCredential, ManagedIdentityCredential  
  
# tiktoken（トークン数見積もり用）  
try:  
    import tiktoken  
except ImportError:  
    tiktoken = None  

  
# ------------------------------- アプリ環境/Flask -------------------------------  
APP_ENV = os.getenv("APP_ENV", "prod").lower()  
IS_LOCAL = APP_ENV == "local"  
  
app = Flask(__name__)  
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'your-default-secret-key')  
app.config['SESSION_TYPE'] = 'filesystem'  
app.config['SESSION_FILE_DIR'] = '/tmp/flask_session'  
app.config['SESSION_PERMANENT'] = False  
app.config['MAX_CONTENT_LENGTH'] = int(os.getenv("MAX_UPLOAD_MB", "100")) * 1024 * 1024  # 100MB  
os.makedirs(app.config['SESSION_FILE_DIR'], exist_ok=True)  
Session(app)  
  
# ローカルではSASリンクを既定ON、本番では既定OFF（必要に応じて上書き）  
# この版ではSASは使用せず、常に中継を使用するため、この設定値に関わらず中継を使います。  
USE_SAS_LINKS = os.getenv("USE_SAS_LINKS", "true" if IS_LOCAL else "false").lower() == "true"  
  
# ------------------------------- Azure Identity 認証 -------------------------------  
def build_credential():  
    if IS_LOCAL:  
        return AzureCliCredential()  
    mi_client_id = os.getenv("AZURE_CLIENT_ID")  
    return ManagedIdentityCredential(client_id=mi_client_id) if mi_client_id else ManagedIdentityCredential()  
  
credential = build_credential()  
  
def build_azure_ad_token_provider(credential, scope):  
    def _provider():  
        return credential.get_token(scope).token  
    return _provider  
  
aad_token_provider = build_azure_ad_token_provider(credential, "https://cognitiveservices.azure.com/.default")  
  
# ------------------------------- Azure OpenAI クライアント設定（AAD） -------------------------------  
# メインの Responses API クライアント（Entra ID 認証）  
client = AzureOpenAI(  
    base_url=os.getenv("AZURE_OPENAI_ENDPOINT"),  # 例: https://<resource-name>.openai.azure.com/openai/v1  
    api_version="v1",  
    azure_ad_token_provider=aad_token_provider,  
    default_headers={"x-ms-include-response-reasoning-summary": "true"},  
)  
  
# 埋め込み用クライアント  
embed_endpoint = os.getenv("AZURE_OPENAI_EMBED_ENDPOINT") or os.getenv("AZURE_OPENAI_ENDPOINT")  
embed_client = AzureOpenAI(  
    azure_endpoint=embed_endpoint,                # 例: https://<resource-name>.openai.azure.com  
    api_version="2025-04-01-preview",  
    azure_ad_token_provider=aad_token_provider,  
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
REWRITE_MODEL = "gpt-4.1"  
MAX_REWRITE_TURNS = max(1, min(8, int(os.getenv("MAX_REWRITE_TURNS", "4"))))  
  
ENABLE_HYDE = os.getenv("ENABLE_HYDE", "1") not in ("0", "false", "False")  
ENABLE_PRF = os.getenv("ENABLE_PRF", "1") not in ("0", "false", "False")  
RECALL_PARENT_THRESHOLD_FRACTION = float(os.getenv("RECALL_PARENT_THRESHOLD_FRACTION", "0.4"))  
MIN_UNIQUE_PARENTS_ABS = int(os.getenv("MIN_UNIQUE_PARENTS_ABS", "3"))  
  
# 添付の最大バイト（超過時は添付スキップ）  
MAX_ATTACHMENT_BYTES = int(os.getenv("MAX_ATTACHMENT_BYTES", str(30 * 1024 * 1024)))  # 30MB  
  
# ------------------------------- Azure サービスクライアント（AAD） -------------------------------  
search_service_endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")  
  
cosmos_endpoint = os.getenv("AZURE_COSMOS_ENDPOINT")  
database_name = 'chatdb'  
container_name = 'personalchats'       # チャット履歴用  
system_prompt_container_name = 'systemprompts_1'  # ★ システムプロンプト用  
  
cosmos_client = CosmosClient(cosmos_endpoint, credential=credential) if cosmos_endpoint else None  
if cosmos_client:  
    database = cosmos_client.get_database_client(database_name)  
    container = database.get_container_client(container_name)  
    system_prompt_container = database.get_container_client(system_prompt_container_name)  
else:  
    database = None  
    container = None  
    system_prompt_container = None  
  
blob_account_url = os.getenv("AZURE_STORAGE_ACCOUNT_URL")  
blob_service_client = BlobServiceClient(  
    account_url=blob_account_url,  
    credential=credential  
) if blob_account_url else None  
  
image_container_name = 'chatgpt-image'  
file_container_name = 'chatgpt-files'  
  
# コンテナ自動作成（権限不足ならスキップ）  
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
  
# ====== ダウンロード用の検証（緩和版） ======  
ALLOWED_CONTAINERS_FOR_DOWNLOAD = set(INDEX_TO_BLOB_CONTAINER.values()) | {image_container_name, file_container_name}  
  
def validate_blob_path(blobname: str):  
    # パストラバーサル/絶対パス/制御文字/バックスラッシュは拒否。その他（日本語・括弧など）は許容。  
    if ".." in blobname or blobname.startswith(("/", "\\")):  
        abort(400)  
    if re.search(r'[\x00-\x1f\x7f]', blobname) or "\\" in blobname:  
        abort(400)  
  
def validate_container_and_path(container: str, blobname: str):  
    if container not in ALLOWED_CONTAINERS_FOR_DOWNLOAD:  
        abort(403)  
    validate_blob_path(blobname)  
  
# 追加: blobname 正規化（最大2回のunquote）  
def normalize_blobname(s: str) -> str:  
    try:  
        once = unquote(s)  
        twice = unquote(once)  
        return twice if twice != once else once  
    except Exception:  
        return s  
  
# ------------------------------- ユーティリティ -------------------------------  
def generate_sas_url(blob_client, blob_name):  
    """  
    SASは本版では使用しませんが、将来用のヘルパーとして残しています。  
    """  
    if not blob_service_client or not blob_client:  
        return ""  
    try:  
        start = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(minutes=15)  
        expiry = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=1)  
        udk = blob_service_client.get_user_delegation_key(start, expiry)  
        sas_token = generate_blob_sas(  
            account_name=blob_client.account_name,  
            container_name=blob_client.container_name,  
            blob_name=blob_name,  
            user_delegation_key=udk,  
            permission=BlobSasPermissions(read=True),  
            start=start,  
            expiry=expiry,  
            protocol="https",  
            version="2020-12-06",  
        )  
        return f"{blob_client.url}?{sas_token}"  
    except Exception as e:  
        print("SAS/UDK 発行失敗:", e)  
        return ""  
  
def build_download_url(container_name: str, blobname: str, is_text: bool) -> str:  
    """  
    一元的なダウンロードURL生成（常にアプリ中継）。  
    - TXT は /download_txt  
    - 非TXT は /download_blob  
    - url_for へ渡す blobname は quote（スラッシュ維持）  
    """  
    if is_text:  
        return url_for('download_txt', container=container_name, blobname=quote(blobname))  
    return url_for('download_blob', container=container_name, blobname=quote(blobname))  
  
def make_blob_url(container_name: str, blobname: str) -> str:  
    """  
    画像/PDFの表示用 URL を生成（常にアプリ中継の inline 表示）。  
    """  
    try:  
        return url_for('view_blob', container=container_name, blobname=quote(blobname))  
    except Exception:  
        return ""  
  
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
  
# ------------------------------- トークナイズ関連 -------------------------------  
def _get_encoding_for_model(model_name: str):  
    """  
    モデル名から適切そうな tiktoken encoding を取得する。  
    Azure のモデル名でも動くように、fallback を多めにしています。  
    """  
    if tiktoken is None:  
        return None  
    try:  
        return tiktoken.encoding_for_model(model_name)  
    except Exception:  
        pass  
    # o 系モデル用  
    try:  
        return tiktoken.get_encoding("o200k_base")  
    except Exception:  
        pass  
    # それでもダメな場合の最後の手段  
    try:  
        return tiktoken.get_encoding("cl100k_base")  
    except Exception:  
        return None  
  
  
def estimate_input_tokens(model_name: str, input_items: list):  
    """  
    Responses API に渡す input_items から「ざっくり」の入力トークン数を見積もる。  
  
    - text 系コンテンツ (input_text / output_text / text) だけを対象  
    - 画像 (input_image) や PDF (input_file) はトークン数に含めない  
      → その分、モデルの usage.input_tokens より少なめに出る  
    """  
    if not input_items:  
        return 0  
  
    enc = _get_encoding_for_model(model_name)  
    if enc is None:  
        return None  
  
    parts = []  
    try:  
        for item in input_items:  
            role = item.get("role", "user")  
            contents = item.get("content") or []  
            buf = [f"{role}: "]  
            for c in contents:  
                ctype = c.get("type")  
                if ctype in ("input_text", "output_text", "text"):  
                    t = c.get("text") or ""  
                    buf.append(t)  
            parts.append("\n".join(buf))  
        full_text = "\n\n".join(parts)  
        return len(enc.encode(full_text))  
    except Exception as e:  
        print("estimate_input_tokens エラー:", e)  
        return None  
  
  
def extract_usage_input_tokens(resp_obj, resp_payload):  
    """  
    Responses オブジェクトまたはその dict から usage.input_tokens を安全に取り出す。  
    """  
    # 1) オブジェクトから取る  
    try:  
        usage = getattr(resp_obj, "usage", None)  
        if usage:  
            val = getattr(usage, "input_tokens", None)  
            if isinstance(val, int):  
                return val  
            # 念のため別名も試す  
            val2 = getattr(usage, "input", None)  
            if isinstance(val2, int):  
                return val2  
    except Exception:  
        pass  
  
    # 2) dict から取る  
    try:  
        if isinstance(resp_payload, dict):  
            usage2 = resp_payload.get("usage") or {}  
            if isinstance(usage2, dict):  
                val = usage2.get("input_tokens") or usage2.get("input")  
                if isinstance(val, int):  
                    return val  
    except Exception:  
        pass  
  
    return None  
  
# ------------------------------- その他ユーティリティ -------------------------------  
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
    """  
    Search結果の filepath or url から (container, blobname) を復元  
    - URL の場合は path を unquote 済みで分解  
    - 相対パスも unquote 済みに正規化  
    """  
    if not path_or_url:  
        return None, None  
    s = (path_or_url or "").strip().replace("\\", "/")  
    # URL の場合  
    if s.startswith("http://") or s.startswith("https://"):  
        if not is_azure_blob_url(s):  
            return None, None  
        try:  
            u = urlparse(s)  
            path = unquote(u.path)  
            parts = path.lstrip("/").split("/", 1)  
            if len(parts) == 2:  
                return parts[0], parts[1]  
        except Exception:  
            return None, None  
    # 相対パス  
    container_name2 = INDEX_TO_BLOB_CONTAINER.get(selected_index, DEFAULT_BLOB_CONTAINER_FOR_SEARCH)  
    blobname = unquote(s.lstrip("/"))  
    if container_name2 and blobname.startswith(container_name2 + "/"):  
        blobname = blobname[len(container_name2) + 1:]  
    return container_name2, blobname  
  
def resolve_container_blob_from_result(selected_index: str, result):  
    """  
    Search結果から (container, blobname) を頑健に復元する。  
    - filepath に '/' が含まれなければ url/metadata_storage_path から復元  
    """  
    fp = (result.get('filepath') or '').replace('\\', '/')  
    u = result.get('url') or result.get('metadata_storage_path') or ''  
    if fp and ('/' in fp):  
        return resolve_blob_from_filepath(selected_index, fp)  
    if u:  
        if is_azure_blob_url(u):  
            return resolve_blob_from_filepath(selected_index, u)  
    if fp:  
        return resolve_blob_from_filepath(selected_index, fp)  
    return (None, None)  
  
# 追加: フォルダ名抽出（タイトルの右側に表示するため）  
def extract_folder_from_blobname(blobname: str) -> str:  
    if not blobname:  
        return ""  
    p = blobname.replace("\\", "/").strip("/")  
    if "/" not in p:  
        return ""  
    return p.rsplit("/", 1)[0]  
  
def extract_folder_from_result(selected_index: str, result) -> str:  
    try:  
        container2, blobname = resolve_container_blob_from_result(selected_index, result)  
        if blobname:  
            return extract_folder_from_blobname(blobname)  
    except Exception:  
        pass  
    # フォールバック: filepath/url から推定  
    try:  
        fp = (result.get('filepath') or '').replace('\\', '/').strip('/')  
        if '/' in fp:  
            return fp.rsplit('/', 1)[0]  
        u = result.get('url') or result.get('metadata_storage_path') or ''  
        if u and is_azure_blob_url(u):  
            path = unquote(urlparse(u).path).lstrip('/')  
            parts = path.split('/', 1)  
            rel = parts[1] if len(parts) == 2 else parts[0]  
            if '/' in rel:  
                return rel.rsplit('/', 1)[0]  
    except Exception:  
        pass  
    return ""  
  
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
    "・最大4件生成。\n"  
    "・複数クエリを出すときは、互いにできるだけ異なる観点やキーワード構成にしてください。\n"  
    "・同じ単語を並べ替えただけ、助詞だけを変えただけなど、意味がほぼ同じクエリは生成しないでください。\n"  
    "・意味が大きく変わる候補が2〜3件しか思いつかない場合は、その件数だけで構いません。\n"  
    "・説明・前後の余分な文字・改行は禁止。"  
    )  
    user_prompt = (  
    f"history{{{context}}}\n"  
    "endtask ユーザの意図をよく表す文書検索用クエリを最大4件生成してください。\n"  
    "可能であれば、表現や観点が少しずつ異なるクエリを含めてください。\n"  
    f"最新ユーザ質問: {current_prompt}"  
    )  
  
    input_items = [  
        {"role": "system", "content": [{"type": "input_text", "text": system_prompt}]},  
        {"role": "user", "content": [{"type": "input_text", "text": user_prompt}]},  
    ]  
    resp = client.responses.create(  
        model=REWRITE_MODEL,  
        input=input_items,  
        temperature=0.8,  
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
    qs = list(dict.fromkeys(qs))[:4]  
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
    # ローカル/未統合用のデフォルト  
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
  
def persist_assistant_message(active_sid: str, assistant_html: str, full_text: str, system_message: str):  
    if not active_sid:  
        return  
  
    existing_msgs = ensure_messages_from_cosmos(active_sid)  
    session_msgs = session.get("main_chat_messages", [])  
  
    candidate_msgs = (session_msgs or []) + [{  
        "role": "assistant",  
        "content": assistant_html,  
        "type": "html",  
        "text": full_text,  
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
  
            # セッションは表示用キャッシュとして更新（SSEの場合は保存されないが問題ない）  
            session["main_chat_messages"] = merged_msgs  
            sb = session.get("sidebar_messages", [])  
            updated = False  
            if active_sid and sb:  
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
    互換性維持のために残していますが、  
    現在は send_message / stream_message からは呼ばないようにしています。  
    Cosmos 側が正本となる設計にしたため、ここから Cosmos を上書きしない方が安全です。  
    """  
    if not container:  
        return  
    # 必要なら将来用に実装を追加  
  
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
  
# ★★★★★ ここから systemprompts_1 用のコード ★★★★★  
  
def save_system_prompt_item(title: str, content: str):  
    """  
    システムプロンプトの保存先を systemprompts_1 コンテナに変更  
    """  
    if not system_prompt_container:  
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
            system_prompt_container.upsert_item(item)  
            return item['id']  
        except Exception as e:  
            print("システムプロンプト保存エラー:", e)  
            traceback.print_exc()  
            return None  
  
def load_system_prompts():  
    """  
    systemprompts_1 コンテナからシステムプロンプトを読み込む  
    """  
    if not system_prompt_container:  
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
            items = system_prompt_container.query_items(  
                query=query,  
                parameters=parameters,  
                enable_cross_partition_query=True  
            )  
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
    """  
    systemprompts_1 コンテナからシステムプロンプトを削除  
    """  
    if not system_prompt_container:  
        return False  
    with lock:  
        try:  
            user_id = get_authenticated_user()  
            system_prompt_container.delete_item(item=prompt_id, partition_key=user_id)  
            return True  
        except Exception as e:  
            print("システムプロンプト削除エラー:", e)  
            traceback.print_exc()  
            return False  
  
# ★★★★★ systemprompts_1 関連ここまで ★★★★★  
  
def start_new_chat():  
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
        credential=credential,  
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
                dedup_key = r.get("chunk_id") or r.get("filepath") or r.get("url") or r.get("id") or r.get("title")  
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
            dedup_key = r.get("chunk_id") or r.get("filepath") or r.get("url") or r.get("id") or r.get("title")  
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
  
    if context_text:  
        ctx = f"{context_title}:\nBEGIN_CONTEXT\n{context_text}\nEND_CONTEXT" if wrap_context_with_markers else context_text  
        if last_user_index is not None:  
            input_items[last_user_index]["content"].append({"type": "input_text", "text": ctx})  
        else:  
            input_items.append({"role": "user", "content": [{"type": "input_text", "text": ctx}]})  
  
    return input_items, last_user_index  
  
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
  
    if "main_chat_messages" not in session:  
        session["main_chat_messages"] = []  
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
        if request.form.get('new_chat'):  
            start_new_chat()  
            session.modified = True  
            return redirect(url_for('index'))  
  
        if 'select_chat' in request.form:  
            sid = request.form.get('select_chat')  
            sb = session.get("sidebar_messages", [])  
            for i, chat in enumerate(sb):  
                if chat.get("session_id") == sid:  
                    session["current_chat_index"] = i  
                    session["current_session_id"] = sid  
                    # 常に Cosmos を優先  
                    msgs = ensure_messages_from_cosmos(sid) or chat.get("messages", [])  
                    session["main_chat_messages"] = msgs  
                    session.modified = True  
                    break  
            return redirect(url_for('index'))  
  
        if 'toggle_history' in request.form:  
            session["show_all_history"] = not bool(session.get("show_all_history", False))  
            session.modified = True  
            return redirect(url_for('index'))  
  
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
  
        if 'apply_system_prompt' in request.form:  
            prompt_id = request.form.get("select_prompt_id")  
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
  
        if 'add_system_prompt' in request.form:  
            title = (request.form.get('prompt_title') or '').strip()  
            content = (request.form.get('prompt_content') or '').strip()  
            if title and content:  
                pid = save_system_prompt_item(title, content)  
                if pid:  
                    session["saved_prompts"] = load_system_prompts()  
            session.modified = True  
            return redirect(url_for('index'))  
  
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
                flash("ストレージ未設定です。環境変数 AZURE_STORAGE_ACCOUNT_URL を設定してください。", "error")  
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
  
        return redirect(url_for('index'))  
  
    # --- GET: 描画 ---  
    images = []  
    if image_container_client:  
        for blobname in session.get("image_filenames", []):  
            base = blobname.split("/")[-1]  
            display = base.split("__", 1)[1] if "__" in base else base  
            url_img = make_blob_url(image_container_name, blobname)  # /view_blob に統一  
            images.append({'name': display, 'blob': blobname, 'url': url_img})  
  
    files = []  
    if file_container_client:  
        for blobname in session.get("file_filenames", []):  
            base = blobname.split("/")[-1]  
            display = base.split("__", 1)[1] if "__" in base else base  
            url_file = make_blob_url(file_container_name, blobname)  # /view_blob に統一  
            files.append({'name': display, 'blob': blobname, 'url': url_file})  
  
    # ★ ここを追加：アクティブなチャットIDの履歴を必ず Cosmos から読み直す  
    active_sid = session.get("current_session_id")  
    if container and active_sid:  
        chat_history = ensure_messages_from_cosmos(active_sid) or []  
        session["main_chat_messages"] = chat_history  
        session.modified = True  
    else:  
        chat_history = session.get("main_chat_messages", []) or []  
  
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
  
    # ★ クライアントから渡された session_id があれば保持しておく  
    client_sid = (data.get("session_id") or "").strip()  
    if client_sid:  
        session["current_session_id"] = client_sid  
  
    prepared = session.get('prepared_prompts', {})  
    mid = str(uuid.uuid4())  
    prepared[mid] = prompt  
    session['prepared_prompts'] = prepared  
    session.modified = True  
    return jsonify({"message_id": mid})  
  
@app.route('/send_message', methods=['POST'])  
def send_message():  
    data = request.get_json() or {}  
    prompt = (data.get('prompt') or '').strip()  
    if not prompt:  
        return (  
            json.dumps({  
                'response': '',  
                'search_files': [],  
                'reasoning_summary': '',  
                'rewritten_queries': [],  
                'session_id': session.get("current_session_id", ""),  
                'debug': None,  
            }),  
            400,  
            {'Content-Type': 'application/json'}  
        )  
  
    # ★ クライアントから渡された session_id を最優先  
    client_sid = (data.get("session_id") or "").strip()  
    if client_sid:  
        active_sid = client_sid  
        session["current_session_id"] = client_sid  
    else:  
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
  
    # ★ ここで毎回 Cosmos から最新履歴を読み込む（なければ空リスト）  
    if container and active_sid:  
        messages = ensure_messages_from_cosmos(active_sid) or []  
    else:  
        messages = session.get("main_chat_messages", []) or []  
  
    # 表示用キャッシュとして session にコピー  
    session["main_chat_messages"] = messages  
    session.modified = True  
  
    # ユーザメッセージを追加  
    messages.append({"role": "user", "content": prompt, "type": "text"})  
    session["main_chat_messages"] = messages  
    session.modified = True  
  
    try:  
        selected_index = session.get("selected_search_index", DEFAULT_SEARCH_INDEX)  
        doc_count = max(1, min(300, int(session.get("doc_count", DEFAULT_DOC_COUNT))))  
        rag_enabled = bool(session.get("rag_enabled", True))  
  
        # system_message の決定  
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
  
        # ------------ RAG 部分 ------------  
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
                container_name2, blobname = resolve_container_blob_from_result(selected_index, r)  
                url = ''  
                if container_name2 and blobname:  
                    try:  
                        is_text = blobname.lower().endswith('.txt')  
                        url = build_download_url(container_name2, blobname, is_text)  
                    except Exception as e:  
                        print("ダウンロードURL生成エラー:", e)  
                        url = url_for('download_blob', container=container_name2, blobname=quote(blobname))  
                path_display = r.get('filepath') or (  
                    f"{container_name2}/{blobname}" if (container_name2 and blobname)  
                    else (r.get('url') or r.get('metadata_storage_path') or '')  
                )  
                folder = extract_folder_from_result(selected_index, r)  
                search_files.append({  
                    'title': title,  
                    'content': content,  
                    'url': url,  
                    'filepath': path_display,  
                    'folder': folder  
                })  
        else:  
            queries = []  
            search_files = []  
            context = ""  
        # ------------ RAG 部分ここまで ------------  
  
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
  
        # 添付ファイル一覧（デバッグ用にも使う）  
        image_filenames = session.get("image_filenames", [])  
        file_filenames = session.get("file_filenames", [])  
  
        # 画像 / PDF 添付  
        if target_user_index is not None:  
            # 画像  
            for img_blob in image_filenames:  
                if not image_container_client:  
                    continue  
                try:  
                    blob_client = image_container_client.get_blob_client(img_blob)  
                    props = blob_client.get_blob_properties()  
                    if props.size and props.size > MAX_ATTACHMENT_BYTES:  
                        print(f"画像が大きすぎるため添付スキップ: {img_blob} ({props.size} bytes)")  
                        continue  
                    img_b64 = encode_image_from_blob(blob_client)  
                    ext = os.path.splitext(img_blob)[1].lower()  
                    mime = "image/png" if ext == ".png" else ("image/gif" if ext == ".gif" else "image/jpeg")  
                    data_url = f"data:{mime};base64,{img_b64}"  
                    input_items[target_user_index]["content"].append({  
                        "type": "input_image",  
                        "image_url": data_url  
                    })  
                except Exception as e:  
                    print("画像添付Base64生成エラー:", e)  
                    traceback.print_exc()  
  
            # PDF  
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
  
        # モデル / reasoning 設定  
        model_to_use = session.get("selected_model", RESPONSES_MODEL)  
        effort = session.get("reasoning_effort", REASONING_EFFORT)  
        if effort not in {"low", "medium", "high"}:  
            effort = REASONING_EFFORT  
        enable_reasoning = model_to_use in REASONING_ENABLED_MODELS  
  
        # OpenAI 呼び出し用パラメータ（デバッグにも使う）  
        request_kwargs = dict(model=model_to_use, input=input_items, store=False)  
        if enable_reasoning:  
            request_kwargs["reasoning"] = {"effort": effort}  
  
        api_request_payload = {  
            "model": model_to_use,  
            "input": input_items,  
            "store": False,  
        }  
        if enable_reasoning:  
            api_request_payload["reasoning"] = {"effort": effort}  
  
        # OpenAI 呼び出し  
        response = client.responses.create(**request_kwargs)  
  
        # レスポンスを dict に変換（そのままデバッグ表示用）  
        try:  
            if hasattr(response, "model_dump_json"):  
                api_response_payload = json.loads(response.model_dump_json())  
            else:  
                api_response_payload = json.loads(str(response))  
        except Exception:  
            try:  
                api_response_payload = json.loads(str(response))  
            except Exception:  
                api_response_payload = str(response)  
  
        # トークン数チェック（ローカル計算 vs usage.input_tokens）  
        local_input_tokens = estimate_input_tokens(model_to_use, input_items)  
        api_input_tokens = extract_usage_input_tokens(response, api_response_payload)  
        token_diff = None  
        if isinstance(local_input_tokens, int) and isinstance(api_input_tokens, int):  
            token_diff = api_input_tokens - local_input_tokens  
  
        output_text = extract_output_text(response)  
        assistant_html = markdown2.markdown(  
            output_text or "",  
            extras=["tables", "fenced-code-blocks", "code-friendly", "break-on-newline", "cuddled-lists"]  
        ) or "<p>（応答テキストが空でした。もう一度お試しください）</p>"  
        reasoning_summary = extract_reasoning_summary(response)  
  
        # ------------ デバッグ情報作成 ------------  
        input_summary = []  
        try:  
            for item in input_items:  
                role = item.get("role")  
                contents = item.get("content") or []  
                texts = []  
                for c in contents:  
                    ctype = c.get("type")  
                    if ctype in ("input_text", "output_text", "text"):  
                        t = (c.get("text") or "")  
                        if len(t) > 120:  
                            t = t[:120] + "…"  
                        texts.append(t)  
                    elif ctype == "input_image":  
                        texts.append("[画像添付]")  
                    elif ctype == "input_file":  
                        filename = c.get("filename") or "（ファイル名不明）"  
                        texts.append(f"[ファイル添付: {filename}]")  
                input_summary.append({"role": role, "contents": texts})  
        except Exception as e:  
            print("debug input_summary 生成エラー(send_message):", e)  
            traceback.print_exc()  
  
        debug_info = {  
            "session_id": active_sid,  
            "selected_model": model_to_use,  
            "reasoning_effort": effort if enable_reasoning else None,  
            "rag_enabled": rag_enabled,  
            "selected_index": selected_index,  
            "doc_count": doc_count,  
            "history_to_send": history_to_send,  
            "system_message": (system_message or "")[:500],  
            "queries": queries,  
            "context_length": len(context or ""),  
            "context_preview": (context or "")[:400],  
            "message_count_before": len(messages),  
            "attachment_images": image_filenames,  
            "attachment_files": file_filenames,  
            "input_summary": input_summary,  
            "api_request": api_request_payload,  
            "api_response": api_response_payload,  
            "token_check": {  
                "local_input_tokens": local_input_tokens,  
                "api_input_tokens": api_input_tokens,  
                "token_diff": token_diff,  
                "note": "local_input_tokens は text 部分のみを tiktoken で数えた概算値で、画像/PDF やフォーマット上のオーバーヘッドは含んでいません。"  
            },  
        }  
        # ------------ デバッグ情報ここまで ------------  
  
        # ★ assistant を Cosmos に保存（ここで session と Cosmos の整合を取る）  
        persist_assistant_message(  
            active_sid=active_sid,  
            assistant_html=assistant_html,  
            full_text=output_text,  
            system_message=system_message  
        )  
  
        return (  
            json.dumps({  
                'response': assistant_html,  
                'search_files': search_files,  
                'reasoning_summary': reasoning_summary,  
                'rewritten_queries': queries,  
                'session_id': active_sid,  
                'debug': debug_info,  
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
                'session_id': active_sid,  
                'debug': None,  
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
  
    # ★ クライアントから渡された sid を最優先  
    sid_param = (request.args.get("sid") or "").strip()  
    if sid_param:  
        active_sid = sid_param  
        session["current_session_id"] = sid_param  
    else:  
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
  
    # ★ ここでも毎回 Cosmos 側から最新履歴を取得する  
    if container and active_sid:  
        messages = ensure_messages_from_cosmos(active_sid) or []  
    else:  
        messages = session.get("main_chat_messages", []) or []  
  
    # セッションの表示用キャッシュを更新  
    session["main_chat_messages"] = messages  
    session.modified = True  
  
    # ユーザメッセージ追加  
    messages.append({"role": "user", "content": prompt, "type": "text"})  
    session["main_chat_messages"] = messages  
    session.modified = True  
  
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
        "error": None  
    }  
  
    @copy_current_request_context  
    def producer():  
        try:  
            search_files = []  
            context = ""  
            queries = []  
  
            # ------------ RAG 部分 ------------  
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
                    container_name2, blobname = resolve_container_blob_from_result(selected_index, r)  
                    url = ''  
                    if container_name2 and blobname:  
                        try:  
                            is_text = blobname.lower().endswith('.txt')  
                            url = build_download_url(container_name2, blobname, is_text)  
                        except Exception as e:  
                            print("ダウンロードURL生成エラー:", e)  
                            url = url_for('download_blob', container=container_name2, blobname=quote(blobname))  
                    path_display = r.get('filepath') or (  
                        f"{container_name2}/{blobname}" if (container_name2 and blobname)  
                        else (r.get('url') or r.get('metadata_storage_path') or '')  
                    )  
                    folder = extract_folder_from_result(selected_index, r)  
                    search_files.append({  
                        'title': title,  
                        'content': content,  
                        'url': url,  
                        'filepath': path_display,  
                        'folder': folder  
                    })  
                result_holder["search_files"] = search_files  
                q.put(_sse_event("search_files", search_files))  
            else:  
                queries = []  
                search_files = []  
                context = ""  
            # ------------ RAG 部分ここまで ------------  
  
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
  
            # 添付ファイル（デバッグ情報にも使うため変数は外側のを参照）  
            if target_user_index is not None:  
                # 画像  
                for img_blob in image_filenames:  
                    if not image_container_client:  
                        continue  
                    try:  
                        blob_client = image_container_client.get_blob_client(img_blob)  
                        props = blob_client.get_blob_properties()  
                        if props.size and props.size > MAX_ATTACHMENT_BYTES:  
                            print(f"画像が大きすぎるため添付スキップ: {img_blob} ({props.size} bytes)")  
                            continue  
                        img_b64 = encode_image_from_blob(blob_client)  
                        ext = os.path.splitext(img_blob)[1].lower()  
                        mime = "image/png" if ext == ".png" else ("image/gif" if ext == ".gif" else "image/jpeg")  
                        data_url = f"data:{mime};base64,{img_b64}"  
                        input_items[target_user_index]["content"].append({  
                            "type": "input_image",  
                            "image_url": data_url  
                        })  
                    except Exception as e:  
                        print("画像添付Base64生成エラー:", e)  
                        traceback.print_exc()  
  
                # PDF  
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
  
            # OpenAI Responses Streaming 用リクエスト  
            request_kwargs = dict(model=model_to_use, input=input_items, store=False)  
            if enable_reasoning:  
                request_kwargs["reasoning"] = {"effort": effort}  
  
            api_request_payload = {  
                "model": model_to_use,  
                "input": input_items,  
                "store": False,  
            }  
            if enable_reasoning:  
                api_request_payload["reasoning"] = {"effort": effort}  
  
            # OpenAI Responses Streaming  
            with client.responses.stream(**request_kwargs) as stream:  
                for event in stream:  
                    etype = getattr(event, "type", "")  
                    if etype == "response.output_text.delta":  
                        delta = getattr(event, "delta", "") or ""  
                        if isinstance(delta, str) and delta:  
                            result_holder["full_text"] += delta  
                            q.put(_sse_event("delta", {"text": delta}))  
                    elif etype.endswith(".delta"):  
                        delta = getattr(event, "delta", "") or ""  
                        if isinstance(delta, str) and delta:  
                            result_holder["full_text"] += delta  
                            q.put(_sse_event("delta", {"text": delta}))  
                    elif etype == "response.error":  
                        err = getattr(event, "error", None)  
                        msg = str(err) if err else "unknown error"  
                        result_holder["error"] = msg  
                        q.put(_sse_event("error", {"message": msg}))  
                final_response = stream.get_final_response()  
  
            # full_text が空なら final_response からテキスト抽出  
            if not result_holder["full_text"]:  
                try:  
                    final_text = extract_output_text(final_response) or ""  
                    if final_text:  
                        result_holder["full_text"] = final_text  
                        q.put(_sse_event("delta", {"text": final_text}))  
                except Exception as e:  
                    print("final_response からの出力抽出失敗:", e)  
  
            # レスポンスを dict に変換（そのままデバッグ表示用）  
            try:  
                if hasattr(final_response, "model_dump_json"):  
                    api_response_payload = json.loads(final_response.model_dump_json())  
                else:  
                    api_response_payload = json.loads(str(final_response))  
            except Exception:  
                try:  
                    api_response_payload = json.loads(str(final_response))  
                except Exception:  
                    api_response_payload = str(final_response)  
  
            # トークン数チェック  
            local_input_tokens = estimate_input_tokens(model_to_use, input_items)  
            api_input_tokens = extract_usage_input_tokens(final_response, api_response_payload)  
            token_diff = None  
            if isinstance(local_input_tokens, int) and isinstance(api_input_tokens, int):  
                token_diff = api_input_tokens - local_input_tokens  
  
            result_holder["reasoning_summary"] = extract_reasoning_summary(final_response)  
            full_text = result_holder["full_text"]  
            assistant_html = markdown2.markdown(  
                full_text or "",  
                extras=["tables", "fenced-code-blocks", "code-friendly", "break-on-newline", "cuddled-lists"]  
            ) or "<p>（応答テキストが空でした。もう一度お試しください）</p>"  
            result_holder["assistant_html"] = assistant_html  
  
            # assistant を Cosmos に保存  
            try:  
                if full_text or assistant_html:  
                    persist_assistant_message(  
                        active_sid=active_sid,  
                        assistant_html=assistant_html,  
                        full_text=full_text,  
                        system_message=system_message  
                    )  
            except Exception as e:  
                print("SSE producer 履歴保存エラー:", e)  
                traceback.print_exc()  
  
            # ------------ デバッグ情報: モデルに渡す内容＋レスポンス ------------  
            input_summary = []  
            try:  
                for item in input_items:  
                    role = item.get("role")  
                    contents = item.get("content") or []  
                    texts = []  
                    for c in contents:  
                        ctype = c.get("type")  
                        if ctype in ("input_text", "output_text", "text"):  
                            t = (c.get("text") or "")  
                            if len(t) > 120:  
                                t = t[:120] + "…"  
                            texts.append(t)  
                        elif ctype == "input_image":  
                            texts.append("[画像添付]")  
                        elif ctype == "input_file":  
                            filename = c.get("filename") or "（ファイル名不明）"  
                            texts.append(f"[ファイル添付: {filename}]")  
                    input_summary.append({"role": role, "contents": texts})  
            except Exception as e:  
                print("debug input_summary 生成エラー(stream_message):", e)  
                traceback.print_exc()  
  
            debug_info = {  
                "session_id": active_sid,  
                "selected_model": model_to_use,  
                "reasoning_effort": effort if enable_reasoning else None,  
                "rag_enabled": rag_enabled,  
                "selected_index": selected_index,  
                "doc_count": doc_count,  
                "history_to_send": history_to_send,  
                "system_message": (system_message or "")[:500],  
                "queries": queries,  
                "context_length": len(context or ""),  
                "context_preview": (context or "")[:400],  
                "message_count_before": len(messages),  
                "attachment_images": image_filenames,  
                "attachment_files": file_filenames,  
                "input_summary": input_summary,  
                "api_request": api_request_payload,  
                "api_response": api_response_payload,  
                "token_check": {  
                    "local_input_tokens": local_input_tokens,  
                    "api_input_tokens": api_input_tokens,  
                    "token_diff": token_diff,  
                    "note": "local_input_tokens は text 部分のみを tiktoken で数えた概算値で、画像/PDF やフォーマット上のオーバーヘッドは含んでいません。"  
                },  
            }  
            q.put(_sse_event("debug", debug_info))  
            # ------------ デバッグ情報ここまで ------------  
  
            if result_holder["reasoning_summary"]:  
                q.put(_sse_event("reasoning_summary", {"summary": result_holder["reasoning_summary"]}))  
  
            q.put(_sse_event("final", {"html": assistant_html, "session_id": active_sid}))  
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
  
    headers = {  
        "Content-Type": "text/event-stream; charset=utf-8",  
        "Cache-Control": "no-cache, no-transform",  
        "X-Accel-Buffering": "no",  
        "Connection": "keep-alive"  
    }  
    return app.response_class(stream_with_context(generate()), headers=headers)  
  
# ------------------------------- テキスト Blob ダウンロード（attachment） -------------------------------  
@app.route("/download_txt/<container>/<path:blobname>")  
def download_txt(container, blobname):  
    if not blob_service_client:  
        return ("Blob service not configured", 500)  
  
    # 受け側は最大2回 unquote で正規化  
    blobname = normalize_blobname(blobname)  
  
    # バリデーションは try の外で実行（abort を捕捉しない）  
    validate_container_and_path(container, blobname)  
  
    try:  
        bc = blob_service_client.get_blob_client(container=container, blob=blobname)  
        txt_bytes = bc.download_blob().readall()  
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
    except ResourceNotFoundError:  
        return ("Not Found", 404)  
    except ClientAuthenticationError:  
        return ("Forbidden", 403)  
    except HttpResponseError as e:  
        print("Blob download_txt error:", e)  
        return (f"Blob download error: {e}", 502)  
    except Exception as e:  
        print("Unexpected blob download_txt error:", e)  
        traceback.print_exc()  
        return (f"Unexpected error: {e}", 500)  
  
# ------------------------------- バイナリ Blob ダウンロード（attachment） -------------------------------  
@app.route("/download_blob/<container>/<path:blobname>")  
def download_blob(container, blobname):  
    if not blob_service_client:  
        return ("Blob service not configured", 500)  
  
    # 受け側は最大2回 unquote で正規化  
    blobname = normalize_blobname(blobname)  
  
    # バリデーションは try の外で実行（abort を捕捉しない）  
    validate_container_and_path(container, blobname)  
  
    try:  
        bc = blob_service_client.get_blob_client(container=container, blob=blobname)  
        data = bc.download_blob().readall()  
        filename = os.path.basename(blobname)  
        content_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"  
        return send_file(io.BytesIO(data), as_attachment=True, download_name=filename, mimetype=content_type)  
    except ResourceNotFoundError:  
        return ("Not Found", 404)  
    except ClientAuthenticationError:  
        return ("Forbidden", 403)  
    except HttpResponseError as e:  
        print("Blob download error:", e)  
        return (f"Blob download error: {e}", 502)  
    except Exception as e:  
        print("Unexpected blob download error:", e)  
        traceback.print_exc()  
        return (f"Unexpected error: {e}", 500)  
  
# ------------------------------- Blob inline 表示（画像/PDF等） -------------------------------  
@app.route("/view_blob/<container>/<path:blobname>")  
def view_blob(container, blobname):  
    if not blob_service_client:  
        return ("Blob service not configured", 500)  
  
    blobname = normalize_blobname(blobname)  
    validate_container_and_path(container, blobname)  
  
    try:  
        bc = blob_service_client.get_blob_client(container=container, blob=blobname)  
        data = bc.download_blob().readall()  
        content_type = mimetypes.guess_type(blobname)[0] or "application/octet-stream"  
        return send_file(io.BytesIO(data), as_attachment=False, download_name=os.path.basename(blobname), mimetype=content_type)  
    except ResourceNotFoundError:  
        return ("Not Found", 404)  
    except ClientAuthenticationError:  
        return ("Forbidden", 403)  
    except HttpResponseError as e:  
        print("Blob view error:", e)  
        return (f"Blob view error: {e}", 502)  
    except Exception as e:  
        print("Unexpected blob view error:", e)  
        traceback.print_exc()  
        return (f"Unexpected error: {e}", 500)  
  
# ------------------------------- エントリポイント -------------------------------  
if __name__ == '__main__':  
    app.run(debug=True, host='0.0.0.0')  