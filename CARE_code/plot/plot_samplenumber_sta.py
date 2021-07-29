import matplotlib.pyplot as plt
import numpy as np
import xlwt
import pickle as pkl
data_path = '/home/lixiangpeng/data/dataset/visual_dialog/data/CL_preprocess/'

mode='train'
with open(data_path+'%s_lengthstatistic_iqa.pkl'%mode, 'rb') as f:
    result = pkl.load(f)

x_list = list(result.keys())
x_list.sort()

cfd = []
cfd_sum = 0
y_list = []

for x in x_list:
    y_list.append(result[x])
    cfd_sum += result[x]
    cfd.append(cfd_sum)
tol_len = cfd[-1]

cfd = (np.array(cfd) / tol_len).tolist()

#plt.plot(x_list, y_list)
a = np.array(cfd) - 0.2
abs_a = abs(a)
b = np.argsort(abs_a)

# plt.bar(x_list, y_list)
print(x_list[b[0]]) # indices of cdf==0.2


line_x_s = [0]
line_x_e = [x_list[b[0]]]
line_y = [0.2]
#plt.hlines(line_y, xmin=line_x_s, xmax=line_x_e, color='#000000')
plt.bar(x_list[:b[0]], y_list[:b[0]], color='#EE82EE')
plt.bar(x_list[b[0]:], y_list[b[0]:], color='#6495ED')
plt.xlabel('Sample Complexity')
plt.ylabel('Sample Number')

plt.title('Complexity Statistic')
plt.show()
