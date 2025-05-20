import json
import pandas as pd
import dask.dataframe as dd
from sklearn.preprocessing import MultiLabelBinarizer
from mlxtend.frequent_patterns import fpgrowth, association_rules

ELECTRONICS = {
    "智能手机", "笔记本电脑", "平板电脑", "智能手表",
    "耳机", "音响", "相机", "摄像机", "游戏机"
}


def load_product_id_to_category(catalog_path):
    with open(catalog_path, "r", encoding="utf-8") as f:
        catalog = json.load(f)
    df = pd.DataFrame(catalog["products"])
    return {row["id"]: row["category"] for _, row in df.iterrows()}

def extract_order_categories(purchase_json_str, id_to_category):
    try:
        purchase = json.loads(purchase_json_str)
        item_ids = [item["id"] for item in purchase.get("items", [])]
        categories = list({id_to_category[i] for i in item_ids if i in id_to_category})
        return categories if categories else None
    except:
        return None

def transform_to_boolean_df(transactions):
    mlb = MultiLabelBinarizer()
    binary_matrix = mlb.fit_transform(transactions)
    return pd.DataFrame(binary_matrix, columns=mlb.classes_)

def main():
    catalog_path = "product_catalog.json"
    parquet_path = "cleaned_data_filtered/part.*.parquet"

    print("加载商品类别...")
    id_to_category = load_product_id_to_category(catalog_path)

    print("读取购买记录并提取订单商品类别...")
    ddf = dd.read_parquet(parquet_path, engine="pyarrow", columns=["purchase_history"])
    ddf["categories"] = ddf["purchase_history"].map(
        lambda x: extract_order_categories(x, id_to_category),
        meta=("categories", "object")
    )

    transactions = ddf["categories"].dropna().compute()

    if transactions.empty:
        print("没有有效的商品类别交易数据。")
        return

    print(f"提取到 {len(transactions)} 条订单商品类别数据")

    print("转换为布尔编码矩阵...")
    df_boolean = transform_to_boolean_df(transactions)

    print("使用 FP-Growth 挖掘频繁项集...")
    freq_items = fpgrowth(df_boolean, min_support=0.02, use_colnames=True)

    print("生成关联规则（置信度 ≥ 0.5）...")
    rules = association_rules(freq_items, metric="confidence", min_threshold=0.5)

    if rules.empty:
        print("未发现关联规则。")
    else:
        print("\n满足条件的频繁规则：\n")
        for _, row in rules.iterrows():
            antecedent = ', '.join(row['antecedents'])
            consequent = ', '.join(row['consequents'])
            print(f"【{antecedent}】 => 【{consequent}】 | 支持度: {row['support']:.3f}, 置信度: {row['confidence']:.3f}")

        print("\n特别关注电子产品与其他类别的规则：\n")
        for _, row in rules.iterrows():
            ant = row['antecedents']
            con = row['consequents']
            if (ELECTRONICS & ant) or (ELECTRONICS & con):
                antecedent = ', '.join(ant)
                consequent = ', '.join(con)
                print(f"【{antecedent}】 => 【{consequent}】 | 支持度: {row['support']:.3f}, 置信度: {row['confidence']:.3f}")

if __name__ == "__main__":
    main()
