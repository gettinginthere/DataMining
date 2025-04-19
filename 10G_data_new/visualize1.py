import dask.dataframe as dd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
# 分布式读取数据
ddf = dd.read_parquet('cleaned_data/part.*.parquet')

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.histplot(ddf['age'], bins=30, kde=True, color='skyblue')
plt.title('Age')
plt.xlabel('年龄')

plt.subplot(1, 2, 2)
sns.histplot(ddf['income'], bins=30, kde=True, color='salmon')
plt.title('Income')
plt.xlabel('收入')
plt.tight_layout()
plt.savefig('matplotlib_hist.png')  # 保存静态图片
print("Matplotlib直方图已保存为 matplotlib_hist.png")
plt.show()