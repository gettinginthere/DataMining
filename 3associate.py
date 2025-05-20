import json
import pandas as pd
import dask.dataframe as dd
import dask

def load_high_value_product_ids(catalog_path, price_threshold=5000):
    with open(catalog_path, "r", encoding="utf-8") as f:
        catalog = json.load(f)
    product_df = pd.DataFrame(catalog["products"])
    high_value_ids = set(product_df[product_df["price"] > price_threshold]["id"])
    return high_value_ids

def extract_payment_method_if_high_value(purchase_json_str, high_value_ids):
    try:
        purchase = json.loads(purchase_json_str)
        item_ids = [item["id"] for item in purchase.get("items", [])]
        if any(i in high_value_ids for i in item_ids):
            return purchase.get("payment_method")
    except Exception:
        return None
    return None

def make_dask_extractor(high_value_ids):
    def func(purchase_json_str):
        return extract_payment_method_if_high_value(purchase_json_str, high_value_ids)
    return func

def main():
    catalog_path = "product_catalog.json"
    parquet_path = "cleaned_data_filtered/part.*.parquet"

    print("加载高价值产品 ID...")
    high_value_ids = load_high_value_product_ids(catalog_path)

    print("读取 Parquet 数据...")
    ddf = dd.read_parquet(parquet_path, engine="pyarrow", columns=["purchase_history"])

    print("提取高价值支付方式...")
    extractor_func = make_dask_extractor(high_value_ids)
    ddf["high_value_payment_method"] = ddf["purchase_history"].map(
        extractor_func,
        meta=("high_value_payment_method", "object")
    )

    print("统计首选支付方式...")
    payment_counts = (
        ddf["high_value_payment_method"]
        .dropna()
        .value_counts()
        .compute()
    )

    if payment_counts.empty:
        print("未找到高价值商品的购买记录。")
    else:
        top_payment_method = payment_counts.idxmax()
        print("\n高价值商品（价格 > 5000）的首选支付方式为：", top_payment_method)
        print("\n各支付方式使用频率：")
        print(payment_counts)

if __name__ == "__main__":
    main()
