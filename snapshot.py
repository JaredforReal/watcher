"""
python snapshot.py --limit 1 --hf-api-base https://hf-mirror.com --no-proxy --write --create-missing
"""
import argparse
import os
import time
from datetime import datetime
from urllib.parse import quote

import requests
from dotenv import load_dotenv


load_dotenv()

DEFAULT_HF_API_BASE = "https://huggingface.co"
HF_MIRROR_API_BASE = "https://hf-mirror.com"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Collect a one-off Hugging Face download snapshot for all models in an org."
    )
    parser.add_argument(
        "--org",
        default=os.environ.get("SNAPSHOT_HF_ORG") or os.environ.get("HF_ORG", "zai-org"),
        help="Hugging Face org/user name. Defaults to SNAPSHOT_HF_ORG, HF_ORG, or zai-org.",
    )
    parser.add_argument(
        "--hf-api-base",
        default=os.environ.get("HF_API_BASE", DEFAULT_HF_API_BASE),
        help="Hugging Face API base URL. Defaults to https://huggingface.co.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N models, useful for smoke tests.",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Write records to the snapshot Feishu table. Without this flag, only prints a dry-run summary.",
    )
    parser.add_argument(
        "--create-missing",
        action="store_true",
        help="Create missing Feishu rows when --write is used. Default only updates rows that already exist today.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.1,
        help="Seconds to sleep between Feishu writes. Defaults to 0.1.",
    )
    parser.add_argument(
        "--no-proxy",
        action="store_true",
        help="Ignore HTTP(S)_PROXY environment variables for HF requests.",
    )
    return parser.parse_args()


def make_session(no_proxy=False):
    session = requests.Session()
    if no_proxy:
        session.trust_env = False
    return session


def request_json(session, method, url, *, timeout=30, **kwargs):
    last_exc = None
    for attempt in range(3):
        try:
            res = session.request(method, url, timeout=timeout, **kwargs)
            data = res.json()
            if res.status_code >= 400:
                raise RuntimeError(f"{method} {url} failed: {res.status_code} {data}")
            return data
        except Exception as exc:
            last_exc = exc
            if attempt < 2:
                time.sleep(1 + attempt)
    raise last_exc


def candidate_hf_bases(primary):
    primary = primary.rstrip("/")
    bases = [primary]
    if primary == DEFAULT_HF_API_BASE:
        bases.append(HF_MIRROR_API_BASE)
    return bases


def fetch_hf_models(session, org, bases):
    params = {"author": org, "limit": 1000}
    last_exc = None
    for base in bases:
        url = f"{base}/api/models"
        try:
            models = request_json(session, "GET", url, params=params)
            print(f"HF model list source: {base}")
            return sorted(
                [item["modelId"] for item in models if item.get("modelId", "").startswith(f"{org}/")],
                key=str.lower,
            ), base
        except Exception as exc:
            last_exc = exc
            print(f"HF model list failed from {base}: {exc}")
    raise last_exc


def get_hf_downloads(session, model_id, base):
    encoded_model_id = quote(model_id, safe="/")
    url = f"{base.rstrip('/')}/api/models/{encoded_model_id}"
    try:
        data = request_json(
            session,
            "GET",
            url,
            params={"expand[]": "downloadsAllTime"},
            timeout=30,
        )
    except Exception as exc:
        print(f"    WARN: {model_id} downloads failed: {exc}")
        return 0
    return data.get("downloadsAllTime") or 0


def get_feishu_token():
    app_id = os.environ["FEISHU_APP_ID"]
    app_secret = os.environ["FEISHU_APP_SECRET"]
    url = "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal"
    data = request_json(
        requests.Session(),
        "POST",
        url,
        json={"app_id": app_id, "app_secret": app_secret},
    )
    token = data.get("tenant_access_token")
    if data.get("code") != 0 or not token:
        raise RuntimeError(f"获取飞书 token 失败: {data}")
    return token


class FeishuClient:
    def __init__(self):
        self.token = get_feishu_token()

    def refresh_token(self):
        self.token = get_feishu_token()

    def request(self, method, url, **kwargs):
        for attempt in range(2):
            headers = kwargs.pop("headers", {}).copy()
            headers.update(feishu_headers(self.token))
            data = request_json(
                requests.Session(),
                method,
                url,
                headers=headers,
                **kwargs,
            )
            if data.get("code") == 99991663 and attempt == 0:
                print("飞书 token 已失效，刷新后重试...")
                self.refresh_token()
                continue
            return data
        return data


def snapshot_base_token():
    return os.environ["SNAPSHOT_FEISHU_BASE_TOKEN"]


def snapshot_downloads_table_id():
    return os.environ["SNAPSHOT_FEISHU_DOWNLOADS_TABLE_ID"]


def feishu_headers(token):
    return {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}


def get_today_snapshot_record(client, model_id):
    url = (
        "https://open.feishu.cn/open-apis/bitable/v1/apps/"
        f"{snapshot_base_token()}/tables/{snapshot_downloads_table_id()}/records/search"
    )
    payload = {
        "page_size": 1,
        "field_names": ["Model ID", "日期"],
        "filter": {
            "conjunction": "and",
            "conditions": [
                {
                    "field_name": "Model ID",
                    "operator": "is",
                    "value": [model_id],
                },
                {
                    "field_name": "日期",
                    "operator": "is",
                    "value": ["Today"],
                },
            ],
        },
    }
    data = client.request("POST", url, json=payload)
    if data.get("code") != 0:
        raise RuntimeError(f"读取 snapshot {model_id} 今天记录失败: {data}")
    items = data.get("data", {}).get("items", [])
    return items[0] if items else None


def get_today_snapshot_records(client, model_ids):
    records = []
    for model_id in sorted(set(model_ids), key=str.lower):
        record = get_today_snapshot_record(client, model_id)
        if record:
            records.append(record)
        time.sleep(0.05)

    print(f"Snapshot Downloads 表今天已有 {len(records)} 条记录")
    return records


def create_snapshot_record(client, fields):
    url = (
        "https://open.feishu.cn/open-apis/bitable/v1/apps/"
        f"{snapshot_base_token()}/tables/{snapshot_downloads_table_id()}/records"
    )
    return client.request("POST", url, json={"fields": fields})


def update_snapshot_record(client, record_id, fields):
    url = (
        "https://open.feishu.cn/open-apis/bitable/v1/apps/"
        f"{snapshot_base_token()}/tables/{snapshot_downloads_table_id()}/records/{record_id}"
    )
    return client.request("PUT", url, json={"fields": fields})


def extract_text(field_value):
    if isinstance(field_value, list) and field_value:
        return field_value[0].get("text", "")
    return str(field_value) if field_value else ""


def today_start_ms():
    today = datetime.now().date()
    return int(datetime.combine(today, datetime.min.time()).timestamp() * 1000)


def is_today(date_val, today_start):
    if not date_val:
        return False
    today_end = today_start + 24 * 60 * 60 * 1000 - 1
    return today_start <= int(date_val) <= today_end


def existing_today_records(records, today_start):
    mapping = {}
    for item in records:
        fields = item.get("fields", {})
        model_id = extract_text(fields.get("Model ID"))
        if not model_id or model_id == "Total":
            continue
        if is_today(fields.get("日期"), today_start):
            mapping[model_id] = item["record_id"]
    return mapping


def model_display_name(model_id, org):
    prefix = f"{org}/"
    return model_id[len(prefix):] if model_id.startswith(prefix) else model_id


def main():
    args = parse_args()
    hf_session = make_session(no_proxy=args.no_proxy)
    models, hf_base = fetch_hf_models(hf_session, args.org, candidate_hf_bases(args.hf_api_base))
    if args.limit is not None:
        models = models[: args.limit]

    print(f"Found {len(models)} HF models under {args.org}")
    today_start = today_start_ms()
    rows = []
    for index, full_model_id in enumerate(models, start=1):
        downloads = get_hf_downloads(hf_session, full_model_id, hf_base)
        model_id = model_display_name(full_model_id, args.org)
        rows.append(
            {
                "Model ID": model_id,
                "HF总下载量": downloads,
                "日期": today_start,
            }
        )
        print(f"  [{index}/{len(models)}] {model_id}: HF={downloads}")

    if not args.write:
        total = sum(row["HF总下载量"] for row in rows)
        print(f"\nDry run only. Total HF downloads across listed models: {total}")
        print("Use --write to push these rows to the snapshot Feishu table.")
        return

    client = FeishuClient()
    records = get_today_snapshot_records(client, [row["Model ID"] for row in rows])
    today_records = existing_today_records(records, today_start)
    skipped = []
    forbidden_creates = []
    create_forbidden = False
    for row in rows:
        model_id = row["Model ID"]
        if model_id in today_records:
            res = update_snapshot_record(client, today_records[model_id], row)
            action = "更新"
        elif args.create_missing:
            if create_forbidden:
                forbidden_creates.append(model_id)
                print(f"  [跳过] {model_id}: 飞书禁止新增记录")
                continue
            try:
                res = create_snapshot_record(client, row)
            except RuntimeError as exc:
                if " 403 " not in str(exc) or "'code': 91403" not in str(exc):
                    raise
                create_forbidden = True
                forbidden_creates.append(model_id)
                print(f"  [跳过] {model_id}: 飞书禁止新增记录 (403/91403)")
                continue
            action = "新增"
        else:
            skipped.append(model_id)
            print(f"  [跳过] {model_id}: 今天没有已有记录，未传 --create-missing")
            continue
        status = "OK" if res.get("code") == 0 else res.get("msg", "FAIL")
        print(f"  [{action}] {model_id}: HF={row['HF总下载量']} -> {status}")
        time.sleep(args.sleep)

    if skipped:
        print(f"\nSkipped {len(skipped)} missing rows. Use --create-missing to create them if the Feishu app has permission.")
    if forbidden_creates:
        print(
            f"\nSkipped {len(forbidden_creates)} creates because Feishu returned 403/91403 Forbidden. "
            "The app can read this table, but it does not have permission to add records."
        )

    print("\nDone!")


if __name__ == "__main__":
    main()
