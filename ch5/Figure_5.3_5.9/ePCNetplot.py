import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

# 设置字体类型以避免输出中的字体问题
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# 加载数据
pyrate = np.load('./data/pyrate.npy')
nnrate_noQos = np.load('./data/nnrate_noQos.npy')
nnrate_Qos = np.load('./data/nnrate_Qos.npy')
mprate = np.load('./data/mprate.npy')
rdrate = np.load('./data/rdrate.npy')

# 创建图形和主坐标轴
plt.figure()
plt.style.use('seaborn-deep')
fig, ax = plt.subplots(figsize=(8, 6))  # 调整图形大小

# 整理数据
data = np.vstack([pyrate, nnrate_noQos, nnrate_Qos, mprate, rdrate]).T
sorted_data = np.sort(data, axis=0)
cumulative_percentiles = np.arange(1, len(sorted_data) + 1) / len(sorted_data)

# 定义线型、标签和标记
linestyle = ['-', '--', ':', '-.', 'dotted']
label = ['WMMSE', 'ePCNet No QoS', 'ePCNet QoS', 'MAX Power', 'Random Power']
markers = ['o', 's', 'D', '^', 'v']

# 绘制主图
for i in range(len(linestyle)):
    ax.plot(sorted_data[:, i], cumulative_percentiles, linestyle=linestyle[i],
            label=label[i], marker=markers[i], markevery=0.1, linewidth=2.0)

# 创建局部放大区域
axins = inset_axes(ax, width="50%", height="80%", loc='upper right',
                   bbox_to_anchor=(0.3, 0.2, 0.5, 0.5), bbox_transform=ax.transAxes)

# 选择放大的区域
x1, x2, y1, y2 = 2.5, 3.3, 0.4, 0.8  # 调整这些值以选择放大的范围
axins.set_xlim(x1, x2)
axins.set_ylim(y1, y2)

# 在局部放大区域中绘制相同的数据，调整markevery参数以减少标记密度
for i in range(len(linestyle)):
    axins.plot(sorted_data[:, i], cumulative_percentiles, linestyle=linestyle[i],
               label=label[i], marker=markers[i], markevery=0.3, linewidth=1.0)  # markevery调整为0.2

# 添加连接线
mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")

# 调整图例的位置，避免与子图重叠
ax.legend(loc='lower right', fontsize=10)   

# 添加轴标签
ax.set_xlim([0, 8])
ax.set_xlabel('sum-rate (bit/sec)', fontsize=14)
ax.set_ylabel('cumulative percentiles', fontsize=14)

# 保存并显示图像
plt.savefig('CDF.eps', format='eps', dpi=2000)
plt.show()
