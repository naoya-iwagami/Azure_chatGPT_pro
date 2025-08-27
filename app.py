#!/usr/bin/env python3  
# -*- coding: utf-8 -*-  
"""  
PM Compass – Flask アプリ（PDFアップロード＆分析対応版, 埋め込み分離 + RRFオーバーフェッチ対応）  
  
主な機能:  
  
- Azure OpenAI Responses API を使用（system 役割、会話履歴、reasoning 対応）  
- Azure Cognitive Search のハイブリッド検索（キーワード＋セマンティック＋ベクター, RRF融合）  
- 取得ドキュメント数を 1～300 の範囲で変更可能（デフォルト10）  
- 会話履歴の送信件数を 1～50 の範囲で変更可能（デフォルト10）  
- モデル選択（gpt-4o / gpt-4.1 / o3 / o4-mini / gpt-5）と reasoning_effort 設定  
- 画像/PDF アップロード（画像: input_image, PDF: input_file）に対応  
- Cosmos DB にチャット履歴・指示テンプレートを保存/読込/削除  
- 検索結果（search_files）をフロントで番号付き表示し、SASリンクやテキストダウンロードも提供  
  
修正点の要旨:  
- Azure Cognitive Search クライアントに api_version="2024-07-01" を明示  
- セマンティック検索で @search.rerankerScore を利用し、strictness 既定値を 0.0 に  
- ハイブリッド融合時の重複判定キーを強化（filepath → id → title）  
- 取得ドキュメント数の上限を 300 に拡張  
- 埋め込みは別の Azure OpenAI エンドポイント/キー/APIバージョン/デプロイ名で呼び出せるよう分離  
  - AZURE_OPENAI_EMBED_ENDPOINT / AZURE_OPENAI_EMBED_KEY / AZURE_OPENAI_EMBED_API_VERSION / AZURE_OPENAI_EMBED_DEPLOYMENT を追加  
- RRF のオーバーフェッチを導入（ユニーク文書数の不足を緩和）  
  - RRF_FETCH_MULTIPLIER（既定2.5倍）と RRF_FETCH_MAX_TOP（既定300）で調整  
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
from urllib.parse import quote, unquote  
  
from flask import (  
    Flask, request, render_template, redirect, url_for, session,  
    flash, send_file  
)  
from flask_session import Session  # pip install Flask-Session  
from werkzeug.utils import secure_filename  
  
# Azure / OpenAI 関連  
import certifi  
from azure.core.credentials import AzureKeyCredential  
from azure.core.pipeline.transport import RequestsTransport  
from azure.search.documents import SearchClient  
from azure.cosmos import CosmosClient  
from azure.storage.blob import (  
    BlobServiceClient, generate_blob_sas, BlobSasPermissions  
)  
from openai import AzureOpenAI  
import markdown2  
  
# プロキシ（SyntaxError 解消: 行を分割）  
os.environ['HTTP_PROXY'] = 'http://g3.konicaminolta.jp:8080'  
os.environ['HTTPS_PROXY'] = 'http://g3.konicaminolta.jp:8080'  
  
# -------------------------------  
# Azure OpenAI クライアント設定  
# -------------------------------  
# Responses 用クライアント（従来どおり）  
client = AzureOpenAI(  
    api_key=os.getenv("AZURE_OPENAI_KEY"),  
    base_url=os.getenv("AZURE_OPENAI_ENDPOINT"),  
    api_version="preview",  
    default_headers={"x-ms-include-response-reasoning-summary": "true"}  
)  
  
# 埋め込み用クライアント（別の Azure OpenAI リソース/バージョンに切替可）  
# 未設定の場合は Responses と同じ設定を継承  
embed_client = AzureOpenAI(  
    api_key=os.getenv("AZURE_OPENAI_KEY"),  
    azure_endpoint=os.getenv("AZURE_OPENAI_EMBED_ENDPOINT"),  
    api_version="2025-04-01-preview"
)  
  
# モデル・埋め込み・検索設定  
RESPONSES_MODEL = os.getenv("AZURE_OPENAI_RESPONSES_MODEL", "gpt-4o")  
REASONING_ENABLED_MODELS = set(  
    m.strip() for m in os.getenv("REASONING_ENABLED_MODELS", "o3,o4-mini,gpt-5,o4").split(",") if m.strip()  
)  
REASONING_EFFORT = os.getenv("REASONING_EFFORT", "high")  
  
# Azure OpenAI の埋め込みは「デプロイ名」を指定  
EMBEDDING_MODEL = "text-embedding-3-large" 
VECTOR_FIELD = "contentVector"  
  
MAX_HISTORY_TO_SEND = int(os.getenv("MAX_HISTORY_TO_SEND", "10"))   # 1～50  
DEFAULT_DOC_COUNT = int(os.getenv("DEFAULT_DOC_COUNT", "10"))       # 1～300  
  
# RRF オーバーフェッチ設定  
RRF_FETCH_MULTIPLIER = float(os.getenv("RRF_FETCH_MULTIPLIER", "3.0"))  # 各方式の取得件数倍率  
RRF_FETCH_MAX_TOP = int(os.getenv("RRF_FETCH_MAX_TOP", "300"))          # サービス上限に合わせる  
  
# -------------------------------  
# Flask アプリ設定  
# -------------------------------  
app = Flask(__name__)  
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'your-default-secret-key')  
app.config['SESSION_TYPE'] = 'filesystem'  
app.config['SESSION_FILE_DIR'] = '/tmp/flask_session'  
app.config['SESSION_PERMANENT'] = False  
# セッションディレクトリが無い場合に備えて作成（任意・安全）  
os.makedirs(app.config['SESSION_FILE_DIR'], exist_ok=True)  
Session(app)  
  
# -------------------------------  
# Azure サービスクライアント  
# -------------------------------  
search_service_endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")  
search_service_key = os.getenv("AZURE_SEARCH_KEY")  
transport = RequestsTransport(verify=certifi.where())  
  
cosmos_endpoint = os.getenv("AZURE_COSMOS_ENDPOINT")  
cosmos_key = os.getenv("AZURE_COSMOS_KEY")  
database_name = 'chatdb'  
container_name = 'personalchats'  
cosmos_client = CosmosClient(cosmos_endpoint, credential=cosmos_key)  
database = cosmos_client.get_database_client(database_name)  
container = database.get_container_client(container_name)  
  
blob_connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")  
blob_service_client = BlobServiceClient.from_connection_string(blob_connection_string)  
image_container_name = 'chatgpt-image'  
image_container_client = blob_service_client.get_container_client(image_container_name)  
file_container_name = 'chatgpt-files'  
file_container_client = blob_service_client.get_container_client(file_container_name)  
  
# UI で選択するインデックス（表示名 → インデックス名）  
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
  
# スレッドロック  
lock = threading.Lock()  
  
# -------------------------------  
# ユーティリティ  
# -------------------------------  
def extract_account_key(connection_string: str) -> str:  
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
  
def encode_image_from_blob(blob_client):  
    downloader = blob_client.download_blob()  
    image_bytes = downloader.readall()  
    return base64.b64encode(image_bytes).decode('utf-8')  
  
def encode_pdf_from_blob(blob_client):  
    downloader = blob_client.download_blob()  
    pdf_bytes = downloader.readall()  
    return base64.b64encode(pdf_bytes).decode('utf-8')  
  
def strip_html_tags(html: str) -> str:  
    """簡易 HTML→テキスト"""  
    if not html:  
        return ""  
    html = re.sub(r'<br\s*/?>', '\n', html, flags=re.I)  
    html = re.sub(r'</p\s*>', '\n', html, flags=re.I)  
    text = re.sub(r'<[^>]+>', '', html)  
    return text  
  
def extract_reasoning_summary(resp) -> str:  
    """Responses API 応答から reasoning summary を抽出"""  
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
    """  
    Responses API 応答から本文テキストを堅牢に抽出する。  
    - resp.output_text があれば最優先で返す  
    - なければ output 配列から message/content を走査して text を連結  
    """  
    try:  
        # 1) 最優先: aggregated output_text  
        if hasattr(resp, "output_text") and resp.output_text:  
            return resp.output_text  
        if isinstance(resp, dict) and resp.get("output_text"):  
            return resp["output_text"]  
  
        # 2) フォールバック: output 配列から抽出  
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
  
# -------------------------------  
# 認証/履歴/指示テンプレート  
# -------------------------------  
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
                        "system_message",  
                        session.get("default_system_message", "あなたは親切なAIアシスタントです…")  
                    ),  
                    'first_assistant_message': current.get("first_assistant_message", ""),  
                    'timestamp': datetime.datetime.now(datetime.timezone.utc).isoformat()  
                }  
                container.upsert_item(item)  
        except Exception as e:  
            print("チャット履歴保存エラー:", e)  
            traceback.print_exc()  
  
def load_chat_history():  
    with lock:  
        user_id = get_authenticated_user()  
        sidebar_messages = []  
        try:  
            one_week_ago = (  
                datetime.datetime.now(datetime.timezone.utc) -  
                datetime.timedelta(days=7)  
            ).isoformat()  
            query = """  
                SELECT * FROM c  
                WHERE c.user_id = @user_id AND c.timestamp >= @one_week_ago  
                ORDER BY c.timestamp DESC  
            """  
            parameters = [  
                {"name": "@user_id", "value": user_id},  
                {"name": "@one_week_ago", "value": one_week_ago}  
            ]  
            items = container.query_items(  
                query=query,  
                parameters=parameters,  
                enable_cross_partition_query=True  
            )  
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
            items = container.query_items(  
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
    with lock:  
        try:  
            user_id = get_authenticated_user()  
            # パーティションキーが /user_id と仮定  
            container.delete_item(item=prompt_id, partition_key=user_id)  
            return True  
        except Exception as e:  
            print("システムプロンプト削除エラー:", e)  
            traceback.print_exc()  
            return False  
  
def start_new_chat():  
    # 一時アップロードの削除（画像）  
    image_filenames = session.get("image_filenames", [])  
    for img_name in image_filenames:  
        blob_client = image_container_client.get_blob_client(img_name)  
        try:  
            blob_client.delete_blob()  
        except Exception as e:  
            print("画像削除エラー:", e)  
    session["image_filenames"] = []  
  
    # 一時アップロードの削除（PDF）  
    file_filenames = session.get("file_filenames", [])  
    for file_name in file_filenames:  
        blob_client = file_container_client.get_blob_client(file_name)  
        try:  
            blob_client.delete_blob()  
        except Exception as e:  
            print("ファイル削除エラー:", e)  
    session["file_filenames"] = []  
  
    new_session_id = str(uuid.uuid4())  
    new_chat = {  
        "session_id": new_session_id,  
        "messages": [],  
        "first_assistant_message": "",  
        "system_message": session.get(  
            'default_system_message',  
            "あなたは親切なAIアシスタントです…"  
        )  
    }  
    sidebar = session.get("sidebar_messages", [])  
    sidebar.insert(0, new_chat)  
    session["sidebar_messages"] = sidebar  
    session["current_chat_index"] = 0  
    session["main_chat_messages"] = []  
  
# -------------------------------  
# Azure Cognitive Search  
# -------------------------------  
def get_search_client(index_name):  
    return SearchClient(  
        endpoint=search_service_endpoint,  
        index_name=index_name,  
        credential=AzureKeyCredential(search_service_key),  
        transport=transport,  
        api_version="2024-07-01"  
    )  
  
def keyword_search(query, topNDocuments, index_name):  
    sc = get_search_client(index_name)  
    results = sc.search(  
        search_text=query,  
        search_fields=["title", "content"],  
        select="title, content, filepath",  
        query_type="simple",  
        top=topNDocuments  
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
        select="title, content, filepath",  
        query_type="semantic",  
        semantic_configuration_name="default",  
        query_caption="extractive",  
        query_answer="extractive",  
        top=top  
    )  
    # rerankerScore を優先  
    filtered = []  
    for r in results:  
        score = r.get("@search.rerankerScore", r.get("@search.score", 0.0))  
        if score >= strictness:  
            filtered.append(r)  
    filtered.sort(key=lambda x: x.get("@search.rerankerScore", x.get("@search.score", 0.0)), reverse=True)  
    return filtered  
  
def get_query_embedding(query):  
    try:  
        # Azure OpenAI: model には「埋め込みデプロイ名」を指定  
        resp = embed_client.embeddings.create(model=EMBEDDING_MODEL, input=query,  
            dimensions=1536)  
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
  
        # 必ず辞書形式で "kind": "vector" をセット  
        vector_query = {  
            "kind": "vector",  
            "vector": embedding,  
            "exhaustive": True,  
            "fields": VECTOR_FIELD,  
            "weight": 0.5,  
            "k": topNDocuments  
        }  
        results = sc.search(  
            search_text="*",  
            vector_queries=[vector_query],  
            select="title, content, filepath",  
            top=topNDocuments  
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
    """  
    RRF によるハイブリッド融合  
    - 各方式はオーバーフェッチし、最後に topNDocuments で切り詰める  
    """  
    rrf_k = 60  
    fusion_scores = {}  
    fusion_docs = {}  
  
    # オーバーフェッチ件数を算出  
    try:  
        req_top = int(topNDocuments)  
    except Exception:  
        req_top = 10  
    # 倍率を適用してサービス上限でクリップ  
    fetch_top = int(min(RRF_FETCH_MAX_TOP, max(req_top, req_top * RRF_FETCH_MULTIPLIER)))  
  
    for q in queries:  
        # 3 種類の結果を統合（オーバーフェッチ）  
        lists = [  
            keyword_search(q, fetch_top, index_name),  
            keyword_semantic_search(q, fetch_top, index_name, strictness),  
            keyword_vector_search(q, fetch_top, index_name)  
        ]  
        for result_list in lists:  
            for idx, r in enumerate(result_list):  
                # 重複判定キー（優先順）  
                doc_id = r.get("filepath") or r.get("id") or r.get("title")  
                if not doc_id:  
                    continue  
                contribution = 1 / (rrf_k + (idx + 1))  
                fusion_scores[doc_id] = fusion_scores.get(doc_id, 0) + contribution  
                if doc_id not in fusion_docs:  
                    fusion_docs[doc_id] = r  
  
    sorted_doc_ids = sorted(fusion_scores, key=lambda d: fusion_scores[d], reverse=True)  
    fused_results = []  
    for doc_id in sorted_doc_ids[:req_top]:  
        r = fusion_docs[doc_id]  
        r["fusion_score"] = fusion_scores[doc_id]  
        fused_results.append(r)  
    return fused_results  
  
# -------------------------------  
# Responses API 入力を構築  
# -------------------------------  
def build_responses_input_with_history(all_messages, system_message, context_text, max_history_to_send=None):  
    """  
    Responses API に渡す input を組み立てる。  
  
    - system_message を role=system として先頭に追加  
    - 直近 max_history_to_send 件の履歴（user, assistant）を追加  
    - 直近の user に context_text を input_text として追記  
    - 添付は呼び出し側で最後の user の content に追加  
    """  
    input_items = []  
  
    if system_message:  
        input_items.append({  
            "role": "system",  
            "content": [  
                {"type": "input_text", "text": system_message}  
            ]  
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
            content_type = "output_text"  
        else:  
            text = m.get("content", "")  
            content_type = "input_text"  
  
        if not text:  
            continue  
  
        input_items.append({  
            "role": role,  
            "content": [  
                {"type": content_type, "text": text}  
            ]  
        })  
  
    if context_text:  
        for idx in range(len(input_items) - 1, -1, -1):  
            if input_items[idx].get("role") == "user":  
                input_items[idx]["content"].append({  
                    "type": "input_text",  
                    "text": f"以下のコンテキストを参考にしてください:\n{context_text}"  
                })  
                break  
  
    return input_items  
  
# -------------------------------  
# ルーティング  
# -------------------------------  
@app.route('/', methods=['GET', 'POST'])  
def index():  
    get_authenticated_user()  
  
    # 初期値（未設定時のみ）  
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
        session["show_all_history"] = False  
    if "main_chat_messages" not in session:  
        idx = session.get("current_chat_index", 0)  
        sidebar = session.get("sidebar_messages", [])  
        if sidebar and idx < len(sidebar):  
            session["main_chat_messages"] = sidebar[idx].get("messages", [])  
        else:  
            session["main_chat_messages"] = []  
    if "image_filenames" not in session:  
        session["image_filenames"] = []  
    if "file_filenames" not in session:  
        session["file_filenames"] = []  
    if "show_all_history" not in session:  
        session["show_all_history"] = False  
    if "saved_prompts" not in session:  
        session["saved_prompts"] = load_system_prompts()  
    session.modified = True  
  
    # POST: 設定更新・ファイル操作  
    if request.method == 'POST':  
        # モデル/effort  
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
  
        # インデックス  
        if 'set_index' in request.form:  
            sel_index = (request.form.get("search_index") or "").strip()  
            if sel_index in INDEX_VALUES:  
                session["selected_search_index"] = sel_index  
            session.modified = True  
            return redirect(url_for('index'))  
  
        # 取得ドキュメント数  
        if 'set_doc_count' in request.form:  
            raw = (request.form.get("doc_count") or "").strip()  
            try:  
                val = int(raw)  
            except Exception:  
                val = DEFAULT_DOC_COUNT  
            session["doc_count"] = max(1, min(300, val))  
            session.modified = True  
            return redirect(url_for('index'))  
  
        # 直近N発話  
        if 'set_history_to_send' in request.form:  
            raw = (request.form.get("history_to_send") or "").strip()  
            try:  
                val = int(raw)  
            except Exception:  
                val = MAX_HISTORY_TO_SEND  
            session["history_to_send"] = max(1, min(50, val))  
            session.modified = True  
            return redirect(url_for('index'))  
  
        # system_message 更新  
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
  
        # 指示テンプレート登録  
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
  
        # 指示テンプレート適用  
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
  
        # 指示テンプレート削除  
        if 'delete_system_prompt' in request.form:  
            pid = request.form.get("delete_system_prompt", "")  
            if pid:  
                delete_system_prompt(pid)  
                session["saved_prompts"] = load_system_prompts()  
                flash("指示を削除しました。", "info")  
            session.modified = True  
            return redirect(url_for('index'))  
  
        # 新規チャット  
        if 'new_chat' in request.form:  
            start_new_chat()  
            session["show_all_history"] = False  
            session.modified = True  
            return redirect(url_for('index'))  
  
        # チャット選択  
        if 'select_chat' in request.form:  
            selected_session = request.form.get("select_chat")  
            sidebar = session.get("sidebar_messages", [])  
            for idx, chat in enumerate(sidebar):  
                if chat.get("session_id") == selected_session:  
                    session["current_chat_index"] = idx  
                    session["main_chat_messages"] = chat.get("messages", [])  
                    session["default_system_message"] = chat.get(  
                        "system_message", session.get("default_system_message")  
                    )  
                    break  
            session.modified = True  
            return redirect(url_for('index'))  
  
        # 履歴表示の切替  
        if 'toggle_history' in request.form:  
            session["show_all_history"] = not session.get("show_all_history", False)  
            session.modified = True  
            return redirect(url_for('index'))  
  
        # ファイルアップロード  
        if 'upload_files' in request.form:  
            if 'files' in request.files:  
                files = request.files.getlist("files")  
                image_filenames = session.get("image_filenames", [])  
                file_filenames = session.get("file_filenames", [])  
                for f in files:  
                    if f and f.filename != '':  
                        filename = secure_filename(f.filename)  
                        ext = filename.rsplit('.', 1)[-1].lower()  
                        if ext in ['png', 'jpeg', 'jpg', 'gif']:  
                            blob_client = image_container_client.get_blob_client(filename)  
                            f.stream.seek(0)  
                            blob_client.upload_blob(f.stream, overwrite=True)  
                            if filename not in image_filenames:  
                                image_filenames.append(filename)  
                        elif ext == 'pdf':  
                            blob_client = file_container_client.get_blob_client(filename)  
                            f.stream.seek(0)  
                            blob_client.upload_blob(f.stream, overwrite=True)  
                            if filename not in file_filenames:  
                                file_filenames.append(filename)  
                session["image_filenames"] = image_filenames  
                session["file_filenames"] = file_filenames  
                session.modified = True  
            return redirect(url_for('index'))  
  
        # 画像削除  
        if 'delete_image' in request.form:  
            delete_name = request.form.get("delete_image")  
            image_filenames = session.get("image_filenames", [])  
            image_filenames = [n for n in image_filenames if n != delete_name]  
            blob_client = image_container_client.get_blob_client(delete_name)  
            try:  
                blob_client.delete_blob()  
            except Exception as e:  
                print("画像削除エラー:", e)  
                traceback.print_exc()  
            session["image_filenames"] = image_filenames  
            session.modified = True  
            return redirect(url_for('index'))  
  
        # PDF 削除  
        if 'delete_file' in request.form:  
            delete_name = request.form.get("delete_file")  
            file_filenames = session.get("file_filenames", [])  
            file_filenames = [n for n in file_filenames if n != delete_name]  
            blob_client = file_container_client.get_blob_client(delete_name)  
            try:  
                blob_client.delete_blob()  
            except Exception as e:  
                print("ファイル削除エラー:", e)  
                traceback.print_exc()  
            session["file_filenames"] = file_filenames  
            session.modified = True  
            return redirect(url_for('index'))  
  
    chat_history = session.get("main_chat_messages", [])  
    sidebar_messages = session.get("sidebar_messages", [])  
    image_filenames = session.get("image_filenames", [])  
    file_filenames = session.get("file_filenames", [])  
    saved_prompts = session.get("saved_prompts", [])  
  
    images = [  
        {'name': fn, 'url': image_container_client.get_blob_client(fn).url}  
        for fn in image_filenames  
    ]  
    files = [  
        {'name': fn, 'url': file_container_client.get_blob_client(fn).url}  
        for fn in file_filenames  
    ]  
  
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
  
# -------------------------------  
# メッセージ送信（OpenAI 呼び出し）  
# -------------------------------  
@app.route('/send_message', methods=['POST'])  
def send_message():  
    data = request.get_json()  
    prompt = (data.get('prompt') or '').strip()  
  
    if not prompt:  
        return (  
            json.dumps({  
                'response': '',  
                'search_files': [],  
                'reasoning_summary': ''  
            }),  
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
        blob_container_for_search = selected_index  # 必要に応じてマッピング変更  
  
        # 取得ドキュメント数（1～300）  
        doc_count = max(1, min(300, int(session.get("doc_count", DEFAULT_DOC_COUNT))))  
  
        # ハイブリッド検索（RRF + オーバーフェッチ）  
        queries = [prompt]  
        strictness = 0.0  # rerankerScore フィルタ下限  
        results_list = hybrid_search_multiqueries(  
            queries, doc_count, selected_index, strictness  
        )  
  
        # モデルに渡すコンテキスト（最大 50,000 文字）  
        context = "\n".join([  
            f"ファイル名: {r.get('title', '不明')}\n内容: {r.get('content','')}"  
            for r in results_list  
        ])[:50000]  
  
        # UI 用リンク（SAS / テキストダウンロード）  
        search_files = []  
        for r in results_list:  
            filepath = r.get('filepath', '')  
            title = r.get('title', '不明')  
            content = r.get('content', '')  
            if filepath and filepath.lower().endswith('.txt'):  
                url = url_for(  
                    'download_txt',  
                    container=blob_container_for_search,  
                    blobname=quote(filepath)  
                )  
            elif filepath:  
                try:  
                    blob_client = blob_service_client.get_blob_client(  
                        container=blob_container_for_search, blob=filepath  
                    )  
                    url = generate_sas_url(blob_client, filepath)  
                except Exception:  
                    url = ''  
            else:  
                url = ''  
            search_files.append({'title': title, 'content': content, 'url': url})  
  
        # system_message  
        system_message = session.get("default_system_message", "")  
        idx = session.get("current_chat_index", 0)  
        sidebar = session.get("sidebar_messages", [])  
        if sidebar and 0 <= idx < len(sidebar):  
            system_message = sidebar[idx].get("system_message", system_message)  
  
        # Responses API 入力構築（履歴＋system＋context）  
        history_to_send = session.get("history_to_send", MAX_HISTORY_TO_SEND)  
        input_items = build_responses_input_with_history(  
            all_messages=messages,  
            system_message=system_message,  
            context_text=context,  
            max_history_to_send=history_to_send  
        )  
  
        # 添付ファイル（最後の user に付与）  
        last_user_index = None  
        for i in range(len(input_items) - 1, -1, -1):  
            if input_items[i].get("role") == "user":  
                last_user_index = i  
                break  
  
        if last_user_index is not None:  
            # 画像（data URL）  
            image_filenames = session.get("image_filenames", [])  
            for img_name in image_filenames:  
                blob_client = image_container_client.get_blob_client(img_name)  
                try:  
                    encoded = encode_image_from_blob(blob_client)  
                    ext = img_name.rsplit('.', 1)[-1].lower()  
                    mime = f"image/{'jpeg' if ext == 'jpg' else ext}" if ext in ['png', 'jpeg', 'jpg', 'gif'] else 'application/octet-stream'  
                    data_url = f"data:{mime};base64,{encoded}"  
                    input_items[last_user_index]["content"].append(  
                        {"type": "input_image", "image_url": data_url}  
                    )  
                except Exception as e:  
                    print("画像エンコードエラー:", e)  
                    traceback.print_exc()  
  
            # PDF（base64 埋め込み）  
            file_filenames = session.get("file_filenames", [])  
            for pdf_name in file_filenames:  
                if pdf_name.lower().endswith('.pdf'):  
                    blob_client = file_container_client.get_blob_client(pdf_name)  
                    try:  
                        encoded = encode_pdf_from_blob(blob_client)  
                        input_items[last_user_index]["content"].append({  
                            "type": "input_file",  
                            "filename": pdf_name,  
                            "file_data": f"data:application/pdf;base64,{encoded}"  
                        })  
                    except Exception as e:  
                        print("PDFエンコードエラー:", e)  
                        traceback.print_exc()  
  
        # モデル呼び出し  
        model_to_use = session.get("selected_model", RESPONSES_MODEL)  
        request_kwargs = dict(  
            model=model_to_use,  
            input=input_items  
        )  
        if model_to_use in REASONING_ENABLED_MODELS:  
            effort = session.get("reasoning_effort", REASONING_EFFORT)  
            if effort not in {"low", "medium", "high"}:  
                effort = REASONING_EFFORT  
            request_kwargs["reasoning"] = {"effort": effort}  
  
        response = client.responses.create(**request_kwargs)  
  
        # 応答テキスト抽出（堅牢版）  
        output_text = extract_output_text(response)  
  
        # Markdown→HTML  
        assistant_html = markdown2.markdown(  
            output_text,  
            extras=["tables", "fenced-code-blocks", "code-friendly", "break-on-newline", "cuddled-lists"]  
        )  
  
        if not output_text:  
            # 応答が空のときのフォールバック（任意）  
            assistant_html = "<p>（応答テキストが空でした。もう一度お試しください）</p>"  
  
        reasoning_summary = extract_reasoning_summary(response)  
  
        # 履歴へ保存  
        messages.append({  
            "role": "assistant",  
            "content": assistant_html,  
            "type": "html",  
            "text": output_text  
        })  
        session["main_chat_messages"] = messages  
        session.modified = True  
  
        # サイドバーへ反映  
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
                'reasoning_summary': reasoning_summary  
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
                'reasoning_summary': ''  
            }),  
            500,  
            {'Content-Type': 'application/json'}  
        )  
  
# -------------------------------  
# テキスト Blob ダウンロード  
# -------------------------------  
@app.route("/download_txt/<container>/<path:blobname>")  
def download_txt(container, blobname):  
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
  
# -------------------------------  
# エントリポイント  
# -------------------------------  
if __name__ == '__main__':  
    app.run(debug=True, host='0.0.0.0')  