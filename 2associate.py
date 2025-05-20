import json
import pandas as pd
import dask.dataframe as dd
from mlxtend.frequent_patterns import apriori, association_rules
from tqdm import tqdm

def load_high_value_product_categories(catalog_path, price_threshold=5000):
    with open(catalog_path, "r", encoding="utf-8") as f:
        catalog = json.load(f)
    product_df = pd.DataFrame(catalog["products"])
    high_value = product_df[product_df["price"] > price_threshold]
    id_to_category = dict(zip(high_value["id"], high_value["category"]))
    return id_to_category

def extract_category_payment_tags(purchase_json_str, id_to_category):
    try:
        purchase = json.loads(purchase_json_str)
        item_ids = [item["id"] for item in purchase.get("items", [])]
        payment_method = purchase.get("payment_method")

        # 只提取高价值商品的类别
        categories = [id_to_category[i] for i in item_ids if i in id_to_category]
        if categories and payment_method:
            return categories + [payment_method]
    except:
        return None
    return None

def transform_to_boolean_df(transactions):
    from sklearn.preprocessing import MultiLabelBinarizer
    mlb = MultiLabelBinarizer()
    encoded = mlb.fit_transform(transactions)
    return pd.DataFrame(encoded, columns=mlb.classes_)

def main():
    catalog_path = "product_catalog.json"
    parquet_path = "cleaned_data_filtered/part.*.parquet"

    print("加载高价值商品的类别映射...")
    id_to_category = load_high_value_product_categories(catalog_path)

    print("加载并解析购买记录...")
    ddf = dd.read_parquet(parquet_path, engine="pyarrow", columns=["purchase_history"])

    extractor = lambda x: extract_category_payment_tags(x, id_to_category)
    ddf["tags"] = ddf["purchase_history"].map(extractor, meta=("tags", "object"))

    print("收集有效交易数据（含高价值商品）...")
    transactions = ddf["tags"].dropna().compute()

    if transactions.empty:
        print("没有找到高价值商品的交易记录。")
        return

    print(f"共找到 {len(transactions)} 条交易记录。")

    print("转换为布尔编码的交易矩阵...")
    transaction_df = transform_to_boolean_df(transactions)

    print("执行频繁项集挖掘（Apriori）...")
    freq_items = apriori(transaction_df, min_support=0.01, use_colnames=True)

    print("生成关联规则（支持度≥0.01，置信度≥0.6）...")
    rules = association_rules(freq_items, metric="confidence", min_threshold=0.6)

    if rules.empty:
        print("未发现满足条件的关联规则。")
    else:
        print("\n满足条件的规则：\n")
        for _, row in rules.iterrows():
            antecedent = ', '.join(row['antecedents'])
            consequent = ', '.join(row['consequents'])
            print(f"【{antecedent}】 => 【{consequent}】 | 支持度: {row['support']:.3f}, 置信度: {row['confidence']:.3f}")

if __name__ == "__main__":
    main()
