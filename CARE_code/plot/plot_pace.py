import numpy as np
import matplotlib.pyplot as plt

epoches = list(range(1, 11))
print(epoches)
ax = plt.gca()
y1_ndcg = [46.33, 52.56, 55.58, 56.26, 56.74, 59.61, 59.85, 60.22, 60.22, 60.32]
y2_ndcg = [47.04, 52.19, 54.82, 56.00, 56.39, 59.04, 59.41, 59.38, 59.42, 59.40]

y1_mrr = [41.01, 45.32, 48.59, 49.61, 50.26, 51.58, 51.73, 51.61, 51.51, 51.57]
y2_mrr = [40.99, 45.65, 48.75, 49.83, 50.38, 52.10, 52.14, 51.90, 51.91, 51.98]
plt.plot(epoches, y1_ndcg, color='red', linewidth=1.0, linestyle='--', label=u"a")
plt.plot(epoches, y2_ndcg, color='blue', linewidth=1.0, linestyle='--', label=u"b")

# plt.plot(epoches, y1_mrr, color='green', linewidth=1.0)
# plt.plot(epoches, y2_mrr, color='blue', linewidth=1.0)
# plt.legend(handles=y1_data, labels='a')
plt.xlabel("Epoches", fontsize=16)
plt.ylabel("NDCG Value / %", fontsize=16)
plt.title("test sample")


ax.spines['right'].set_color('none')  # right边框属性设置为none 不显示
ax.spines['top'].set_color('none')    # top边框属性设置为none 不显示
plt.show()