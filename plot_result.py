import pandas as pd
import matplotlib.pyplot as plt

# 假设已经读入数据
result_data = pd.read_csv('./result.csv')
pre_label = result_data['predict']
real_label = result_data['real']

plt.figure(figsize=(12, 6), dpi=100) # 增加 dpi 让图更清晰

# 1. 画真实值 (用红色的线或者空心圈，减少遮挡)
# 如果是连续数据(回归)，建议用 plot 线条；如果是分类，建议用 scatter
plt.scatter(range(len(real_label)), real_label, label='Real', color='red', alpha=0.6, s=15, marker='o')

# 2. 画预测值 (用蓝色的点，稍微小一点)
plt.scatter(range(len(pre_label)), pre_label, label='Predict', color='blue', alpha=0.6, s=15, marker='x')

plt.title("Prediction vs Real Label")
plt.xlabel("Sample Index")
plt.ylabel("Value")
plt.legend(loc='best') # 自动寻找最佳位置
plt.grid(True, linestyle='--', alpha=0.5) # 加个虚线网格

plt.show()

plt.figure(figsize=(15, 5))

# 只取前 100 个数据来看 (或者 100:200)
limit = 100
subset_real = real_label[:limit]
subset_pred = pre_label[:limit]
x_axis = range(len(subset_real))

# 用 "线+点" 的方式，真实值用实线，预测值用虚线或点，对比最强烈
plt.plot(x_axis, subset_real, 'r.-', label='Real', linewidth=1.5)
plt.plot(x_axis, subset_pred, 'b.--', label='Predict', linewidth=1.5, alpha=0.7)

plt.title(f"Detail View (First {limit} samples)")
plt.xlabel("Index")
plt.ylabel("Value")
plt.legend()
plt.grid(True)
plt.show()