#!/usr/bin/env python3  
# -*- coding: utf-8 -*-  
  
"""  
personalchats から systemprompts_1 へ  
doc_type = "system_prompt" のドキュメントを移行（コピー＆削除）するスクリプト  
"""  
  
import os  
import traceback  
  
from azure.cosmos import CosmosClient  
from azure.identity import AzureCliCredential, ManagedIdentityCredential  
  
# 必要ならプロキシ設定（不要ならコメントアウト）  
# os.environ['HTTP_PROXY'] = 'http://g3.konicaminolta.jp:8080'  
# os.environ['HTTPS_PROXY'] = 'http://g3.konicaminolta.jp:8080'  
  
  
APP_ENV = os.getenv("APP_ENV", "prod").lower()  
IS_LOCAL = APP_ENV == "local"  
  
  
def build_credential():  
    """  
    アプリ本体と同じロジック:  
    - ローカル: AzureCliCredential  
    - 本番: ManagedIdentityCredential  
    """  
    if IS_LOCAL:  
        return AzureCliCredential()  
    mi_client_id = os.getenv("AZURE_CLIENT_ID")  
    return ManagedIdentityCredential(client_id=mi_client_id) if mi_client_id else ManagedIdentityCredential()  
  
  
def main():  
    cosmos_endpoint = os.getenv("AZURE_COSMOS_ENDPOINT")  
    if not cosmos_endpoint:  
        print("環境変数 AZURE_COSMOS_ENDPOINT が設定されていません。")  
        return  
  
    database_name = "chatdb"  
    src_container_name = "personalchats"  
    dst_container_name = "systemprompts_1"  # ★ ここが新コンテナ  
  
    credential = build_credential()  
  
    client = CosmosClient(cosmos_endpoint, credential=credential)  
    db = client.get_database_client(database_name)  
    src = db.get_container_client(src_container_name)  
    dst = db.get_container_client(dst_container_name)  
  
    # doc_type = "system_prompt" のドキュメントを全件取得  
    query = "SELECT * FROM c WHERE c.doc_type = 'system_prompt'"  
  
    migrated = 0  
    errors = 0  
  
    print("移行開始: personalchats -> systemprompts_1")  
  
    for item in src.query_items(query=query, enable_cross_partition_query=True):  
        try:  
            # 新コンテナに upsert（ID はそのまま）  
            dst.upsert_item(item)  
            # 移行が成功したら元を削除  
            user_id = item.get("user_id")  
            if not user_id:  
                print(f"警告: user_id が無い system_prompt がありました (id={item.get('id')})。削除をスキップします。")  
            else:  
                src.delete_item(item=item["id"], partition_key=user_id)  
            migrated += 1  
        except Exception as e:  
            print("移行エラー:", e)  
            traceback.print_exc()  
            errors += 1  
  
    print(f"移行完了: 移行 {migrated} 件, エラー {errors} 件")  
  
  
if __name__ == "__main__":  
    main()  