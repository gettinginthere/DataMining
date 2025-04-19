import dask.dataframe as dd
import json
import pandas as pd
from dask.diagnostics import ProgressBar
def safe_json_loads(s):
    try:
        if pd.isnull(s) or not s.strip():
            return {}
        return json.loads(s)
    except json.JSONDecodeError:
        print(f"解析失败的数据片段：{s[:50]}") 
        return {}
    except Exception as e:
        print(f"未知错误：{e}，数据片段：{s[:50]}")
        return {}

ddf = dd.read_parquet('cleaned_data/part.*.parquet')

def filter_partition(df_part):
    purchase_data = df_part['purchase_history'].apply(safe_json_loads)
    payment_status = purchase_data.apply(lambda x: x.get('payment_status', ''))
    mask = payment_status == '已支付'
    return df_part[mask]
with ProgressBar():
    filtered_ddf = ddf.map_partitions(filter_partition, meta=ddf._meta)
    filtered_ddf = filtered_ddf.repartition(partition_size="2048MB")
    filtered_ddf.to_parquet(
        'cleaned_data_filtered',
        write_index=False,
        engine='pyarrow',
        compression='snappy'
    )