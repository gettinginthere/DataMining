import json
import pandas as pd
import dask.dataframe as dd
from collections import defaultdict
from itertools import combinations
from datetime import datetime

ELECTRONICS = {
    "智能手机", "笔记本电脑", "平板电脑", "智能手表",
    "耳机", "音响", "相机", "摄像机", "游戏机"
}

def load_product_id_to_category(catalog_path):
    with open(catalog_path, "r", encoding="utf-8") as f:
        catalog = json.load(f)
    df = pd.DataFrame(catalog["products"])
    return {row["id"]: row["category"] for _, row in df.iterrows()}

def extract_user_purchases(row, id_to_category):
    try:
        purchase = json.loads(row["purchase_history"])
        item_ids = [item["id"] for item in purchase.get("items", [])]
        categories = {id_to_category[i] for i in item_ids if i in id_to_category and id_to_category[i] in ELECTRONICS}
        if not categories:
            return None
        purchase_date = purchase.get("purchase_date", "")
        return {
            "user_id": row["id"],
            "date": purchase_date,
            "month": purchase_date[:7],  # YYYY-MM
            "categories": list(categories)
        }
    except:
        return None

def main():
    parquet_path = "cleaned_data_filtered/part.*.parquet"
    catalog_path = "product_catalog.json"

    print("加载商品目录...")
    id_to_category = load_product_id_to_category(catalog_path)

    print("加载数据...")
    ddf = dd.read_parquet(parquet_path, engine="pyarrow", columns=["id", "purchase_history"])

    print("解析电子产品购买记录...")
    df_parsed = ddf.map_partitions(
        lambda df: df.apply(lambda row: extract_user_purchases(row, id_to_category), axis=1)
    ).dropna().compute()

    df_clean = pd.DataFrame(df_parsed.tolist())

    print("生成月度购买频率表...")
    month_cat_counts = defaultdict(int)
    for _, row in df_clean.iterrows():
        month = row["month"]
        for cat in row["categories"]:
            month_cat_counts[(month, cat)] += 1

    monthly_df = pd.DataFrame([
        {"month": m, "category": c, "count": count}
        for (m, c), count in month_cat_counts.items()
    ])
    monthly_df = monthly_df.sort_values(by=["month", "category"])
    monthly_df.to_csv("monthly_electronics_stats.csv", index=False, encoding="utf-8-sig")
    print("月度电子产品购买频率已写入 monthly_electronics_stats.csv")

    print("分析用户的时序购买行为...")
    user_events = df_clean.sort_values(by=["user_id", "date"])
    user_group = user_events.groupby("user_id")

    transitions = defaultdict(int)
    for _, group in user_group:
        timeline = []
        for _, row in group.iterrows():
            for cat in row["categories"]:
                timeline.append((row["date"], cat))
        timeline = sorted(set(timeline), key=lambda x: x[0])
        for (d1, cat1), (d2, cat2) in zip(timeline, timeline[1:]):
            if cat1 != cat2:
                transitions[(cat1, cat2)] += 1

    print("\n常见的电子产品购买顺序（A → B）：")
    sorted_trans = sorted(transitions.items(), key=lambda x: x[1], reverse=True)
    for (cat1, cat2), count in sorted_trans[:15]:
        print(f"【{cat1}】→【{cat2}】：{count} 次")

if __name__ == "__main__":
    main()
