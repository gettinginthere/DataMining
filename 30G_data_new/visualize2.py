import altair as alt
from dask.diagnostics import ProgressBar
import dask.dataframe as dd

ddf = dd.read_parquet('cleaned_data/part.*.parquet')
with ProgressBar():
    timeline = ddf.groupby([
        dd.to_datetime(ddf.registration_date).dt.year.rename('year'),
        ddf.is_active
    ]).size().reset_index().compute()
    
timeline.columns = ['year', 'is_active', 'count']

chart = alt.Chart(timeline).mark_line(
    point=True, 
    interpolate='monotone',  # 平滑曲线
    strokeWidth=2.5
).encode(
    x=alt.X('year:O', title='注册年份', axis=alt.Axis(labelAngle=-45)),
    y=alt.Y('count:Q', title='用户数量'),
    color=alt.Color('is_active:N', 
                  legend=alt.Legend(title="活跃状态"),
                  scale=alt.Scale(scheme='category10')),
    tooltip=['year', 'is_active', 'count'],
    strokeDash='is_active:N' 
).properties(
    width=800,
    height=400,
    title="用户活跃状态年度趋势"
).interactive()

text = chart.mark_text(
    align='left',
    baseline='middle',
    dx=7,
    dy=-5
).encode(
    text='count:Q'
)

(chart + text).save('timeline_chart.html')