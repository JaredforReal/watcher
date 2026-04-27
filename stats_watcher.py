import os
import requests
import time
from dotenv import load_dotenv

load_dotenv()

HF_ORG = os.environ["HF_ORG"]
MS_ORG = os.environ["MS_ORG"]
FEISHU_APP_ID = os.environ["FEISHU_APP_ID"]
FEISHU_APP_SECRET = os.environ["FEISHU_APP_SECRET"]
FEISHU_BASE_TOKEN = os.environ["FEISHU_BASE_TOKEN"]
FEISHU_TABLE_ID = os.environ["FEISHU_TABLE_ID"]
# ====================================================


def get_feishu_token():
    url = "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal"
    res = requests.post(url, json={
        "app_id": FEISHU_APP_ID,
        "app_secret": FEISHU_APP_SECRET,
    })
    return res.json()["tenant_access_token"]


def get_github_star(repo):
    res = requests.get(f"https://api.github.com/repos/{repo}", timeout=10)
    return res.json().get("stargazers_count", 0)


def get_hf_downloads(model_id):
    url = f"https://huggingface.co/api/models/{model_id}?expand[]=downloadsAllTime"
    res = requests.get(url, timeout=10)
    return res.json().get("downloadsAllTime", 0)


def get_ms_downloads(model_id):
    url = f"https://modelscope.cn/api/v1/models/{model_id}"
    res = requests.get(url, timeout=10)
    return res.json().get("Data", {}).get("Downloads", 0)


def get_all_records(token):
    """读取飞书表格全部记录，返回完整的记录列表"""
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    url = f"https://open.feishu.cn/open-apis/bitable/v1/apps/{FEISHU_BASE_TOKEN}/tables/{FEISHU_TABLE_ID}/records/search"

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

    print(f"飞书表格共 {len(all_records)} 条记录")
    return all_records


def extract_text(field_value):
    """从富文本字段中提取纯文本"""
    if isinstance(field_value, list) and field_value:
        return field_value[0].get("text", "")
    return str(field_value) if field_value else ""


def update_record(token, record_id, fields):
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    url = f"https://open.feishu.cn/open-apis/bitable/v1/apps/{FEISHU_BASE_TOKEN}/tables/{FEISHU_TABLE_ID}/records/{record_id}"
    res = requests.put(url, headers=headers, json={"fields": fields})
    return res.json()


def main():
    token = get_feishu_token()
    records = get_all_records(token)

    # 解析表格记录
    model_records = []  # [{model_id, record_id, github_repo}]
    repo_to_records = {}  # repo -> [(record_id, model_id)]
    total_record_id = None
    for item in records:
        fields = item.get("fields", {})
        model_id = extract_text(fields.get("Model ID"))
        github_repo = extract_text(fields.get("GitHub Repo"))
        if model_id == "Total":
            total_record_id = item["record_id"]
            continue
        if not model_id:
            continue
        model_records.append({
            "model_id": model_id,
            "record_id": item["record_id"],
            "github_repo": github_repo,
        })
        if github_repo:
            repo_to_records.setdefault(github_repo, []).append((item["record_id"], model_id))

    # 采集并更新 HF 下载量 + 魔搭下载量
    total_hf = 0
    total_ms = 0
    print("\n--- 更新 HF & 魔搭下载量 ---")
    for r in model_records:
        model_id = r["model_id"]
        hf_id = f"{HF_ORG}/{model_id}"
        ms_id = f"{MS_ORG}/{model_id}"

        hf_downloads = get_hf_downloads(hf_id)
        ms_downloads = get_ms_downloads(ms_id)
        total_hf += hf_downloads
        total_ms += ms_downloads

        res = update_record(token, r["record_id"], {
            "HF总下载量": hf_downloads,
            "魔搭总下载量": ms_downloads,
        })
        status = "OK" if res.get("code") == 0 else res.get("msg")
        print(f"  {model_id}: HF={hf_downloads}, 魔搭={ms_downloads} -> {status}")
        time.sleep(0.2)

    # 更新 Total 行
    if total_record_id:
        res = update_record(token, total_record_id, {
            "HF总下载量": total_hf,
            "魔搭总下载量": total_ms,
        })
        status = "OK" if res.get("code") == 0 else res.get("msg")
        print(f"\n  Total: HF={total_hf}, 魔搭={total_ms} -> {status}")

    # 采集并更新 GitHub Stars（表格 GitHub Repo -> zai-org/Repo）
    print("\n--- 更新 GitHub Stars ---")
    for repo, recs in repo_to_records.items():
        full_repo = f"{HF_ORG}/{repo}"
        stars = get_github_star(full_repo)
        for record_id, model_id in recs:
            res = update_record(token, record_id, {"GitHub Stars": stars})
            status = "OK" if res.get("code") == 0 else res.get("msg")
            print(f"  {repo} -> {model_id}: {stars} stars ({status})")
        time.sleep(0.2)

    print("\nDone!")


if __name__ == "__main__":
    main()
