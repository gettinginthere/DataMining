from dask.diagnostics import ProgressBar
import dask.dataframe as dd
import json
import pandas as pd
import numpy as np

def safe_json_loads(s):
    try:
        return json.loads(s) if pd.notnull(s) and s.strip() else {}
    except:
        return {}

def calculate_spent(df_part):
    df_part = df_part.copy()
    df_part['purchase_data'] = df_part['purchase_history'].apply(safe_json_loads)
    
    df_part['avg_price'] = df_part['purchase_data'].apply(
        lambda x: x.get('avg_price', 0.0)
    ).astype('float32') 
    
    df_part['item_count'] = df_part['purchase_data'].apply(
        lambda x: len(x.get('items', []))
    ).astype('uint16')  
    
    # 计算购买额
    df_part['total_spent'] = df_part['avg_price'] * df_part['item_count']
    return df_part.drop(columns=['purchase_data'])

with ProgressBar():
    ddf = dd.read_parquet('cleaned_data_filtered/part.*.parquet')
    meta = ddf._meta.copy()
    meta['avg_price'] = pd.Series(dtype='float32')
    meta['item_count'] = pd.Series(dtype='uint16')
    meta['total_spent'] = pd.Series(dtype='float32')

    ddf = ddf.map_partitions(calculate_spent, meta=meta)

    ddf = ddf.persist() 
    print("现有列:", ddf.columns.tolist())  

    # 计算阈值（此时total_spent已存在）
    all_spent = ddf['total_spent'].compute()
    threshold = np.percentile(all_spent[all_spent > 0], 95)
    print(f"第5%购买额的阈值为：{threshold:.2f}")

    filtered_top5 = ddf[ddf['total_spent'] >= threshold]
    filtered_top5.to_parquet(
        'top5_users',
        write_index=False,
        engine='pyarrow',
        compression='snappy'
    )