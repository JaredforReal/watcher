import os
import requests
import time
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

HF_ORG = os.environ["HF_ORG"]
MS_ORG = os.environ["MS_ORG"]
GITHUB_ORG = os.environ.get("GITHUB_ORG", HF_ORG)
FEISHU_APP_ID = os.environ["FEISHU_APP_ID"]
FEISHU_APP_SECRET = os.environ["FEISHU_APP_SECRET"]
FEISHU_BASE_TOKEN = os.environ["FEISHU_BASE_TOKEN"]
FEISHU_DOWNLOADS_TABLE_ID = os.environ["FEISHU_DOWNLOADS_TABLE_ID"]
FEISHU_REPO_STARS_TABLE_ID = os.environ["FEISHU_REPO_STARS_TABLE_ID"]

DEFAULT_MODEL_IDS = [
    "GLM-4.5",
    "GLM-4.5-Air",
    "GLM-4.5-Air-Base",
    "GLM-4.5-Air-FP8",
    "GLM-4.5-Base",
    "GLM-4.5-FP8",
    "GLM-4.5V",
    "GLM-4.5V-FP8",
    "GLM-4.6",
    "GLM-4.6-FP8",
    "GLM-4.6V",
    "GLM-4.6V-FP8",
    "GLM-4.6V-Flash",
    "GLM-4.7",
    "GLM-4.7-FP8",
    "GLM-4.7-Flash",
    "GLM-5",
    "GLM-5-FP8",
    "GLM-5.1",
    "GLM-5.1-FP8",
    "GLM-ASR-nano-2512",
    "GLM-Image",
    "GLM-OCR",
    "GLM-TTS",
]

DEFAULT_MODEL_TAGS = {
    "GLM-4.5": "GLM-4.5",
    "GLM-4.5-Air": "GLM-4.5-Air",
    "GLM-4.5-Air-Base": "GLM-4.5-Air",
    "GLM-4.5-Air-FP8": "GLM-4.5-Air",
    "GLM-4.5-Base": "GLM-4.5",
    "GLM-4.5-FP8": "GLM-4.5",
    "GLM-4.5V": "GLM-4.5V",
    "GLM-4.5V-FP8": "GLM-4.5V",
    "GLM-4.6": "GLM-4.6",
    "GLM-4.6-FP8": "GLM-4.6",
    "GLM-4.6V": "GLM-4.6V",
    "GLM-4.6V-FP8": "GLM-4.6V",
    "GLM-4.6V-Flash": "GLM-4.6V-Flash",
    "GLM-4.7": "GLM-4.7",
    "GLM-4.7-FP8": "GLM-4.7",
    "GLM-4.7-Flash": "GLM-4.7-Flash",
    "GLM-5": "GLM-5",
    "GLM-5-FP8": "GLM-5",
    "GLM-5.1": "GLM-5.1",
    "GLM-5.1-FP8": "GLM-5.1",
    "GLM-ASR-nano-2512": "GLM-ASR",
    "GLM-Image": "GLM-Image",
    "GLM-OCR": "GLM-OCR",
    "GLM-TTS": "GLM-TTS",
}


def get_feishu_token():
    url = "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal"
    res = requests.post(url, json={
        "app_id": FEISHU_APP_ID,
        "app_secret": FEISHU_APP_SECRET,
    }, timeout=10)
    try:
        data = res.json()
    except ValueError as exc:
        raise RuntimeError(f"获取飞书 token 失败: HTTP {res.status_code}, 非 JSON 响应") from exc

    token = data.get("tenant_access_token")
    if res.status_code != 200 or data.get("code") != 0 or not token:
        raise RuntimeError(f"获取飞书 token 失败: HTTP {res.status_code}, {data}")
    return token


class FeishuClient:
    def __init__(self):
        self.token = get_feishu_token()

    def refresh_token(self):
        self.token = get_feishu_token()

    def request(self, method, url, **kwargs):
        for attempt in range(2):
            headers = kwargs.pop("headers", {}).copy()
            headers.update({
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json",
            })
            res = requests.request(method, url, headers=headers, timeout=20, **kwargs)
            try:
                data = res.json()
            except ValueError as exc:
                raise RuntimeError(f"飞书请求失败: HTTP {res.status_code}, 非 JSON 响应") from exc

            if data.get("code") == 99991663 and attempt == 0:
                print("飞书 token 已失效，刷新后重试...")
                self.refresh_token()
                continue
            return data

        return data


def get_github_star(repo):
    headers = github_headers()
    res = requests.get(f"https://api.github.com/repos/{repo}", headers=headers, timeout=10)
    data = res.json()
    if res.status_code != 200:
        print(f"    ⚠ GitHub repo {repo} 请求失败: {data.get('message', res.status_code)}")
        return None
    return data.get("stargazers_count", 0)


def get_hf_downloads(model_id):
    url = f"https://huggingface.co/api/models/{model_id}?expand[]=downloadsAllTime"
    res = requests.get(url, timeout=10)
    return res.json().get("downloadsAllTime", 0)


def get_ms_downloads(model_id):
    url = f"https://modelscope.cn/api/v1/models/{model_id}"
    res = requests.get(url, timeout=10)
    return res.json().get("Data", {}).get("Downloads", 0)


def csv_env(name):
    value = os.environ.get(name, "")
    return [item.strip() for item in value.split(",") if item.strip()]


def mapping_env(name):
    mapping = {}
    for item in csv_env(name):
        if "=" in item:
            key, value = item.split("=", 1)
        elif ":" in item:
            key, value = item.split(":", 1)
        else:
            continue
        key = key.strip()
        value = value.strip()
        if key and value:
            mapping[key] = value
    return mapping


def github_headers():
    token = os.environ.get("GITHUB_TOKEN")
    return {"Authorization": f"Bearer {token}"} if token else {}


def get_hf_model_ids():
    configured = csv_env("MODEL_IDS")
    if configured:
        return configured
    if DEFAULT_MODEL_IDS:
        return DEFAULT_MODEL_IDS

    url = "https://huggingface.co/api/models"
    res = requests.get(url, params={"author": HF_ORG, "limit": 1000}, timeout=30)
    data = res.json()
    if res.status_code != 200:
        raise RuntimeError(f"读取 HF 模型列表失败: HTTP {res.status_code}, {data}")

    prefix = f"{HF_ORG}/"
    model_ids = []
    for item in data:
        full_id = item.get("modelId", "")
        if full_id.startswith(prefix):
            model_ids.append(full_id[len(prefix):])
    return sorted(set(model_ids), key=str.lower)


def get_github_repos():
    configured = csv_env("GITHUB_REPOS")
    if configured:
        return configured

    headers = github_headers()
    repos = []
    for owner_type in ("orgs", "users"):
        page = 1
        owner_repos = []
        while True:
            url = f"https://api.github.com/{owner_type}/{GITHUB_ORG}/repos"
            res = requests.get(
                url,
                headers=headers,
                params={"per_page": 100, "page": page, "type": "public"},
                timeout=20,
            )
            data = res.json()
            if res.status_code == 404:
                break
            if res.status_code != 200:
                raise RuntimeError(f"读取 GitHub 仓库列表失败: HTTP {res.status_code}, {data}")
            if not data:
                break
            owner_repos.extend(item["name"] for item in data if not item.get("archived"))
            if len(data) < 100:
                break
            page += 1
        if owner_repos:
            repos = owner_repos
            break

    if not repos:
        print(f"GitHub org/user {GITHUB_ORG} 未发现公开仓库，跳过 GitHub Stars 更新")
    return sorted(set(repos), key=str.lower)


def get_today_records(client, is_downloads, today_start, today_end):
    """只读取今天的飞书记录，避免每次扫描完整历史表。"""
    if is_downloads:
        url = f"https://open.feishu.cn/open-apis/bitable/v1/apps/{FEISHU_BASE_TOKEN}/tables/{FEISHU_DOWNLOADS_TABLE_ID}/records/search"
        field_names = ["Model ID", "Tag", "日期"]
    else:
        url = f"https://open.feishu.cn/open-apis/bitable/v1/apps/{FEISHU_BASE_TOKEN}/tables/{FEISHU_REPO_STARS_TABLE_ID}/records/search"
        field_names = ["GitHub Repo", "日期"]

    all_records = []
    page_token = None

    while True:
        payload = {
            "page_size": 500,
            "field_names": field_names,
            "filter": {
                "conjunction": "and",
                "conditions": [
                    {
                        "field_name": "日期",
                        "operator": "is",
                        "value": ["Today"],
                    },
                ],
            },
        }
        if page_token:
            payload["page_token"] = page_token
        res = client.request("POST", url, json=payload)
        if res.get("code") != 0:
            raise RuntimeError(f"读取今天记录失败: {res}")

        items = res.get("data", {}).get("items", [])
        all_records.extend(items)

        if not res.get("data", {}).get("has_more"):
            break
        page_token = res["data"].get("page_token")

    table_name = "Downloads" if is_downloads else "Stars"
    print(f"{table_name} 表今天已有 {len(all_records)} 条记录")
    return all_records


def get_historical_model_tags(client, model_ids):
    """按 Model ID 服务端过滤查历史 Tag，避免扫描整张 Downloads 表。"""
    tags = DEFAULT_MODEL_TAGS.copy()
    tags.update(mapping_env("MODEL_TAGS"))
    missing_model_ids = sorted(set(model_ids) - set(tags), key=str.lower)
    if not missing_model_ids:
        return tags

    url = f"https://open.feishu.cn/open-apis/bitable/v1/apps/{FEISHU_BASE_TOKEN}/tables/{FEISHU_DOWNLOADS_TABLE_ID}/records/search"
    for model_id in missing_model_ids:
        payload = {
            "page_size": 1,
            "field_names": ["Model ID", "Tag"],
            "filter": {
                "conjunction": "and",
                "conditions": [
                    {
                        "field_name": "Model ID",
                        "operator": "is",
                        "value": [model_id],
                    },
                    {
                        "field_name": "Tag",
                        "operator": "isNotEmpty",
                        "value": [],
                    },
                ],
            },
        }
        res = client.request("POST", url, json=payload)
        if res.get("code") != 0:
            raise RuntimeError(f"读取 {model_id} 历史 Tag 失败: {res}")
        items = res.get("data", {}).get("items", [])
        if not items:
            continue
        tag = extract_text(items[0].get("fields", {}).get("Tag"))
        if tag:
            tags[model_id] = tag
        time.sleep(0.05)

    print(f"历史 Tag 回填 {len(tags)} 个模型")
    return tags


def create_record(client, fields, is_downloads=True):
    """向飞书表格新增一条记录 (POST)"""
    table_id = FEISHU_DOWNLOADS_TABLE_ID if is_downloads else FEISHU_REPO_STARS_TABLE_ID
    url = f"https://open.feishu.cn/open-apis/bitable/v1/apps/{FEISHU_BASE_TOKEN}/tables/{table_id}/records"
    return client.request("POST", url, json={"fields": fields})


def update_record(client, record_id, fields, is_downloads=True):
    if is_downloads:
        url = f"https://open.feishu.cn/open-apis/bitable/v1/apps/{FEISHU_BASE_TOKEN}/tables/{FEISHU_DOWNLOADS_TABLE_ID}/records/{record_id}"
    else:
        url = f"https://open.feishu.cn/open-apis/bitable/v1/apps/{FEISHU_BASE_TOKEN}/tables/{FEISHU_REPO_STARS_TABLE_ID}/records/{record_id}"
    return client.request("PUT", url, json={"fields": fields})


def extract_text(field_value):
    """从富文本字段中提取纯文本"""
    if isinstance(field_value, list) and field_value:
        return field_value[0].get("text", "")
    return str(field_value) if field_value else ""


def get_today_range():
    """返回今天 00:00:00 和 23:59:59 的毫秒时间戳"""
    today = datetime.now().date()
    start = int(datetime.combine(today, datetime.min.time()).timestamp() * 1000)
    end = int(datetime.combine(today, datetime.max.time()).timestamp() * 1000)
    return start, end


def is_today(date_val, today_start, today_end):
    """判断飞书日期字段值是否为今天"""
    if not date_val:
        return False
    ts = int(date_val)
    return today_start <= ts <= today_end


def main():
    client = FeishuClient()
    today_start, today_end = get_today_range()

    # --- Downloads 表 ---
    print("\n--- 更新 HF & 魔搭下载量 ---")
    dl_records = get_today_records(client, True, today_start, today_end)
    model_ids = set(get_hf_model_ids())
    print(f"HF 模型列表共 {len(model_ids)} 个")
    model_tags = {}  # model_id -> tag
    today_dl_map = {}  # model_id -> record_id
    for item in dl_records:
        fields = item.get("fields", {})
        model_id = extract_text(fields.get("Model ID"))
        if not model_id or model_id == "Total":
            continue
        model_ids.add(model_id)
        tag = extract_text(fields.get("Tag"))
        if tag and model_id not in model_tags:
            model_tags[model_id] = tag
        if is_today(fields.get("日期"), today_start, today_end):
            today_dl_map[model_id] = item["record_id"]

    historical_tags = get_historical_model_tags(client, model_ids - set(model_tags))
    model_tags.update(historical_tags)

    for model_id in sorted(model_ids):
        hf_id = f"{HF_ORG}/{model_id}"
        ms_id = f"{MS_ORG}/{model_id}"
        hf_dl = get_hf_downloads(hf_id)
        ms_dl = get_ms_downloads(ms_id)
        fields = {
            "Model ID": model_id,
            "HF总下载量": hf_dl,
            "魔搭总下载量": ms_dl,
            "日期": today_start,
        }
        if model_id in model_tags:
            fields["Tag"] = model_tags[model_id]

        if model_id in today_dl_map:
            res = update_record(client, today_dl_map[model_id], fields, is_downloads=True)
            action = "更新"
        else:
            res = create_record(client, fields, is_downloads=True)
            action = "新增"
        status = "OK" if res.get("code") == 0 else res.get("msg", "FAIL")
        print(f"  [{action}] {model_id}: HF={hf_dl}, 魔搭={ms_dl} -> {status}")
        time.sleep(0.1)

    # --- Stars 表 ---
    print("\n--- 更新 GitHub Stars ---")
    star_records = get_today_records(client, False, today_start, today_end)
    repos = set(get_github_repos())
    print(f"GitHub 仓库列表共 {len(repos)} 个")
    today_star_map = {}  # repo -> record_id
    for item in star_records:
        fields = item.get("fields", {})
        repo = extract_text(fields.get("GitHub Repo"))
        if not repo or repo == "Total":
            continue
        repos.add(repo)
        if is_today(fields.get("日期"), today_start, today_end):
            today_star_map[repo] = item["record_id"]

    for repo in sorted(repos):
        full_repo = f"{GITHUB_ORG}/{repo}"
        stars = get_github_star(full_repo)
        if stars is None:
            continue
        fields = {
            "GitHub Repo": repo,
            "GitHub Stars": stars,
            "日期": today_start,
        }

        if repo in today_star_map:
            res = update_record(client, today_star_map[repo], fields, is_downloads=False)
            action = "更新"
        else:
            res = create_record(client, fields, is_downloads=False)
            action = "新增"
        status = "OK" if res.get("code") == 0 else res.get("msg", "FAIL")
        print(f"  [{action}] {repo}: {stars} stars -> {status}")
        time.sleep(0.1)

    print("\nDone!")


if __name__ == "__main__":
    main()
