import dask.dataframe as dd
from dask.diagnostics import ProgressBar
from collections import defaultdict
import json
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.pyplot as plt
from matplotlib import font_manager

font_path = "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"
font_prop = font_manager.FontProperties(fname=font_path)

plt.rcParams['font.family'] = font_prop.get_name()
plt.rcParams['axes.unicode_minus'] = False  

def safe_json_loads(s):
    try:
        return json.loads(s) if pd.notnull(s) and s.strip() else {}
    except Exception as e:
        print(f"解析失败数据: {s[:50]}")  
        return {}

def process_partition(partition):
    local_counts = defaultdict(int)
    
    purchase_histories = partition['purchase_history'].apply(safe_json_loads)
    valid_mask = purchase_histories.apply(lambda x: x.get('payment_status') == '已支付')
    valid_records = purchase_histories[valid_mask]
    
    for ph in valid_records:
        category = ph.get('categories', '未分类')
        items = ph.get('items', [])
        local_counts[category] += len(items)
    
    return local_counts

def merge_counts(results):
    global_counts = defaultdict(int)
    for local in results:
        for k, v in local.items():
            global_counts[k] += v
    return global_counts

def validate_data(ddf):
    null_ratio = ddf['purchase_history'].isnull().mean().compute()
    print(f"[数据质量] 空值比例: {null_ratio:.2%}")

    sample = ddf['purchase_history'].sample(frac=0.01).compute()
    invalid_samples = sample[sample.apply(lambda x: not isinstance(safe_json_loads(x), dict))]
    print(f"[数据质量] 无效样本数: {len(invalid_samples)}")

if __name__ == "__main__":
    block_size = "2048MB"  # 根据内存调整
    ddf = dd.read_parquet('cleaned_data/part.*.parquet', 
                         blocksize=block_size,
                         engine='pyarrow')
    
    validate_data(ddf)
    with ProgressBar():
        intermediate = ddf.map_partitions(process_partition)
        partial_results = intermediate.compute()
    
    final_counts = merge_counts(partial_results) 
    def plot_pie(category_counts):
        labels, sizes = zip(*sorted(category_counts.items(), key=lambda x: -x[1]))
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.pie(sizes, labels=labels, 
               autopct=lambda p: f'{p:.1f}%' if p > 1 else '', 
               startangle=90,
               wedgeprops={'linewidth': 1, 'edgecolor': 'white'})
        ax.axis('equal')
        plt.title('商品品类销量分布', pad=20)
        plt.savefig('category_distribution.png', dpi=300, bbox_inches='tight')
    
    plot_pie(final_counts)