import json
import pandas as pd
import dask.dataframe as dd
from sklearn.preprocessing import MultiLabelBinarizer
from mlxtend.frequent_patterns import fpgrowth, association_rules
import gc

def load_product_id_to_category(catalog_path):
    with open(catalog_path, "r", encoding="utf-8") as f:
        catalog = json.load(f)
    product_df = pd.DataFrame(catalog["products"])
    return {row["id"]: row["category"] for _, row in product_df.iterrows()}

def extract_items_and_refund_flag(purchase_json_str, id_to_category):
    try:
        purchase = json.loads(purchase_json_str)
        payment_status = purchase.get("payment_status", "")
        refund_flag = payment_status in ["已退款", "部分退款"]
        item_ids = [item["id"] for item in purchase.get("items", [])]
        categories = [id_to_category[i] for i in item_ids if i in id_to_category]
        if not categories:
            return None
        categories.append("退款" if refund_flag else "未退款")
        return categories
    except:
        return None

def filter_frequent_categories(transactions, min_occurrence=100):
    flat = [item for sublist in transactions for item in sublist]
    counts = pd.Series(flat).value_counts()
    keep = set(counts[counts >= min_occurrence].index)
    return [[cat for cat in t if cat in keep] for t in transactions]

def transform_to_sparse_df(transactions):
    transactions = [t for t in transactions if len(t) >= 2]
    if not transactions:
        return pd.DataFrame()
    mlb = MultiLabelBinarizer(sparse_output=True)
    encoded = mlb.fit_transform(transactions)
    sparse_df = pd.DataFrame.sparse.from_spmatrix(encoded, columns=mlb.classes_)
    return sparse_df

def main():
    catalog_path = "product_catalog.json"
    parquet_path = "cleaned_data_filtered/part.*.parquet"

    print("加载商品目录...")
    id_to_category = load_product_id_to_category(catalog_path)

    print("加载并解析购买记录...")
    ddf = dd.read_parquet(parquet_path, engine="pyarrow", columns=["purchase_history"])

    print("提取商品类别 + 退款状态标签...")
    extractor = lambda x: extract_items_and_refund_flag(x, id_to_category)
    ddf["tags"] = ddf["purchase_history"].map(extractor, meta=("tags", "object"))

    transactions = ddf["tags"].dropna().compute()
    gc.collect()

    print(f"共提取 {len(transactions)} 条原始记录，筛选频繁类别中...")
    transactions = filter_frequent_categories(transactions, min_occurrence=100)

    if not transactions:
        print("没有找到有效的交易记录。")
        return

    print(f"剩余 {len(transactions)} 条有效记录。")

    print("转换为稀疏布尔矩阵...")
    transaction_df = transform_to_sparse_df(transactions)

    if transaction_df.empty:
        print("无法生成特征矩阵。")
        return

    print("使用 FPGrowth 挖掘频繁项集（支持稀疏矩阵）...")
    freq_items = fpgrowth(transaction_df, min_support=0.005, use_colnames=True)

    print(f"发现 {len(freq_items)} 个频繁项集。")

    print("生成关联规则（目标：退款）...")
    rules = association_rules(freq_items, metric="confidence", min_threshold=0.4)

    refund_rules = rules[rules["consequents"].apply(lambda x: "退款" in x)]

    if refund_rules.empty:
        print("未发现与退款相关的有效规则。")
    else:
        print("\n发现与退款相关的规则：\n")
        for _, row in refund_rules.iterrows():
            antecedent = ', '.join(row['antecedents'])
            consequent = ', '.join(row['consequents'])
            print(f"【{antecedent}】 => 【{consequent}】 | 支持度: {row['support']:.4f}, 置信度: {row['confidence']:.3f}")

if __name__ == "__main__":
    main()
