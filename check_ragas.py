# check_ragas.py
import os
import traceback

print("--- RAGAS 診断ツール開始 ---")

# 1. ライブラリのインポート確認
print("\n[Step 1] ライブラリをインポート中...")
try:
    from ragas import evaluate
    from datasets import Dataset
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    print("✅ ライブラリのインポートに成功しました。")
except Exception as e:
    print("❌ ライブラリのインポートに失敗しました！")
    print("エラー詳細:")
    traceback.print_exc()
    exit(1)

# 2. 環境変数の確認（簡易）
print("\n[Step 2] 環境変数の読み込み確認...")
# app.py と同じように環境変数をロードする（dotenvがあれば）
try:
    from dotenv import load_dotenv
    load_dotenv()
except:
    pass

endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
api_key = os.getenv("AZURE_OPENAI_API_KEY")
ad_token_provider = None

# ID認証かどうかチェック
from azure.identity import DefaultAzureCredential
try:
    cred = DefaultAzureCredential()
    # トークン取得テスト
    token = cred.get_token("https://cognitiveservices.azure.com/.default")
    print("✅ Azure Identity (Entra ID) 認証: OK")
except Exception as e:
    print(f"⚠️ Azure Identity 認証警告: {e}")
    print("   (ローカルで az login していないと失敗する場合があります)")

# 3. クライアント初期化テスト
print("\n[Step 3] LangChainクライアント初期化テスト...")
try:
    # 実際のモデル名は環境に合わせて調整してください
    llm_deploy = os.getenv("AZURE_OPENAI_RESPONSES_MODEL", "gpt-4o")
    emb_deploy = os.getenv("AZURE_OPENAI_EMBED_DEPLOYMENT", "text-embedding-3-large")
    
    print(f"   使用モデル(LLM): {llm_deploy}")
    print(f"   使用モデル(Embed): {emb_deploy}")

    llm = ChatOpenAI(
        azure_endpoint=endpoint,
        azure_deployment=llm_deploy,
        api_version="2024-06-01",
        # credential=cred # langchainのバージョンによってはtoken_providerが必要
    )
    emb = OpenAIEmbeddings(
        azure_endpoint=endpoint,
        azure_deployment=emb_deploy,
        api_version="2024-06-01",
        # credential=cred
    )
    print("✅ クライアントオブジェクト作成: OK")
    
except Exception as e:
    print("❌ クライアント初期化エラー")
    print("エラー詳細:")
    traceback.print_exc()

print("\n--- 診断終了 ---")