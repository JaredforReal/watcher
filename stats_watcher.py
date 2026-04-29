import os
import requests
import time
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

HF_ORG = os.environ["HF_ORG"]
MS_ORG = os.environ["MS_ORG"]
FEISHU_APP_ID = os.environ["FEISHU_APP_ID"]
FEISHU_APP_SECRET = os.environ["FEISHU_APP_SECRET"]
FEISHU_BASE_TOKEN = os.environ["FEISHU_BASE_TOKEN"]
FEISHU_DOWNLOADS_TABLE_ID = os.environ["FEISHU_DOWNLOADS_TABLE_ID"]
FEISHU_REPO_STARS_TABLE_ID = os.environ["FEISHU_REPO_STARS_TABLE_ID"]


def get_feishu_token():
    url = "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal"
    res = requests.post(url, json={
        "app_id": FEISHU_APP_ID,
        "app_secret": FEISHU_APP_SECRET,
    })
    return res.json()["tenant_access_token"]


def get_github_star(repo):
    res = requests.get(f"https://api.github.com/repos/{repo}", timeout=10)
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


def get_all_records(token, is_downloads=True):
    """读取飞书表格全部记录，返回完整的记录列表"""
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    if is_downloads:
        url = f"https://open.feishu.cn/open-apis/bitable/v1/apps/{FEISHU_BASE_TOKEN}/tables/{FEISHU_DOWNLOADS_TABLE_ID}/records/search"
    else:
        url = f"https://open.feishu.cn/open-apis/bitable/v1/apps/{FEISHU_BASE_TOKEN}/tables/{FEISHU_REPO_STARS_TABLE_ID}/records/search"

    all_records = []
    page_token = None

    while True:
        payload = {"page_size": 100}
        if page_token:
            payload["page_token"] = page_token
        res = requests.post(url, headers=headers, json=payload).json()
        if res.get("code") != 0:
            print(f"读取记录失败: {res}")
            break

        items = res.get("data", {}).get("items", [])
        all_records.extend(items)

        if not res.get("data", {}).get("has_more"):
            break
        page_token = res["data"].get("page_token")

    table_name = "Downloads" if is_downloads else "Stars"
    print(f"{table_name} 表共 {len(all_records)} 条记录")
    return all_records


def create_record(token, fields, is_downloads=True):
    """向飞书表格新增一条记录 (POST)"""
    table_id = FEISHU_DOWNLOADS_TABLE_ID if is_downloads else FEISHU_REPO_STARS_TABLE_ID
    url = f"https://open.feishu.cn/open-apis/bitable/v1/apps/{FEISHU_BASE_TOKEN}/tables/{table_id}/records"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    res = requests.post(url, headers=headers, json={"fields": fields})
    return res.json()


def update_record(token, record_id, fields, is_downloads=True):
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    if is_downloads:
        url = f"https://open.feishu.cn/open-apis/bitable/v1/apps/{FEISHU_BASE_TOKEN}/tables/{FEISHU_DOWNLOADS_TABLE_ID}/records/{record_id}"
    else:
        url = f"https://open.feishu.cn/open-apis/bitable/v1/apps/{FEISHU_BASE_TOKEN}/tables/{FEISHU_REPO_STARS_TABLE_ID}/records/{record_id}"
    res = requests.put(url, headers=headers, json={"fields": fields})
    return res.json()


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
    token = get_feishu_token()
    today_start, today_end = get_today_range()

    # --- Downloads 表 ---
    print("\n--- 更新 HF & 魔搭下载量 ---")
    dl_records = get_all_records(token, is_downloads=True)
    model_ids = set()
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
            res = update_record(token, today_dl_map[model_id], fields, is_downloads=True)
            action = "更新"
        else:
            res = create_record(token, fields, is_downloads=True)
            action = "新增"
        status = "OK" if res.get("code") == 0 else res.get("msg", "FAIL")
        print(f"  [{action}] {model_id}: HF={hf_dl}, 魔搭={ms_dl} -> {status}")
        time.sleep(0.1)

    # --- Stars 表 ---
    print("\n--- 更新 GitHub Stars ---")
    star_records = get_all_records(token, is_downloads=False)
    repos = set()
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
        full_repo = f"{HF_ORG}/{repo}"
        stars = get_github_star(full_repo)
        if stars is None:
            continue
        fields = {
            "GitHub Repo": repo,
            "GitHub Stars": stars,
            "日期": today_start,
        }

        if repo in today_star_map:
            res = update_record(token, today_star_map[repo], fields, is_downloads=False)
            action = "更新"
        else:
            res = create_record(token, fields, is_downloads=False)
            action = "新增"
        status = "OK" if res.get("code") == 0 else res.get("msg", "FAIL")
        print(f"  [{action}] {repo}: {stars} stars -> {status}")
        time.sleep(0.1)

    print("\nDone!")


if __name__ == "__main__":
    main()
