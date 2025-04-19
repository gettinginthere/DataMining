import dask.dataframe as dd
import json
import time
import pandas as pd

# 记录开始时间
start_time = time.time()

ddf = dd.read_parquet('part-*.parquet')
def parse_login_history(login_history_str):
    """解析JSON并返回first_login"""
    try:
        return json.loads(login_history_str).get('first_login')
    except (TypeError, json.JSONDecodeError, AttributeError):
        return None


def safe_date_conversion(df):
    """安全转换日期列"""
    df = df.copy()
    df['registration_date'] = dd.to_datetime(
        df['registration_date'], 
        format='%Y-%m-%d', 
        errors='coerce'
    )
    df['last_login'] = dd.to_datetime(
        df['last_login'], 
        format='%Y-%m-%dT%H:%M:%S%z', 
        errors='coerce'
    )
    df['first_login'] = dd.to_datetime(
        df['login_history'].map(parse_login_history), 
        format='%Y-%m-%d',
        errors='coerce'
    )
    return df

meta = {
    **{col: dtype for col, dtype in ddf.dtypes.items()},
    'registration_date': 'datetime64[ns]',
    'last_login': 'datetime64[ns]',
    'first_login': 'datetime64[ns]'
}

ddf = ddf.map_partitions(safe_date_conversion, meta=meta)

# 条件1：年龄异常
age_condition = (ddf['age'] < 18) | (ddf['age'] > 100)

# 条件2：注册时间异常（使用日期date比较）
date_condition = (
    (ddf['registration_date'].dt.date > ddf['last_login'].dt.date) |
    (ddf['registration_date'].dt.date > ddf['first_login'].dt.date)
)

combined_condition = age_condition | date_condition
invalid_count = combined_condition.astype(int).sum().compute()
print(f"[统计结果] 需删除的异常条目数: {invalid_count:,}")

cleaned_ddf = ddf[~combined_condition]
output_path = 'cleaned_data'
cleaned_ddf.to_parquet(
    output_path,
    engine='pyarrow',
    overwrite=True,
    write_index=False
)

end_time = time.time()
time_cost = end_time - start_time
print(f"\n[性能报告] 总处理时间: {time_cost:.2f}秒")
print(f"        输出文件位置: {output_path}")