import numpy as np
# 原始样本
A = np.mat(
	'3 2000; 2 3000; 4 5000; 5 8000; 1 2000',
	dtype=float)
print('A =', A, sep='\n')
# 归一化缩放：均值为0，极差为1
mu = A.mean(axis=0)
s = A.max(axis=0) - A.min(axis=0)
X = (A - mu) / s
print('X =', X, sep='\n')
# 协方差矩阵
SIGMA = X.T * X
print('SIGMA =', SIGMA, sep='\n')

# A =
# [[  3.00000000e+00   2.00000000e+03]
#  [  2.00000000e+00   3.00000000e+03]
#  [  4.00000000e+00   5.00000000e+03]
#  [  5.00000000e+00   8.00000000e+03]
#  [  1.00000000e+00   2.00000000e+03]]
# Z =
# [[-0.2452941 ]
#  [-0.29192442]
#  [ 0.29192442]
#  [ 0.82914294]
#  [-0.58384884]]
# A_approx =
# [[  2.33563616e+00   2.91695452e+03]
#  [  2.20934082e+00   2.71106794e+03]
#  [  3.79065918e+00   5.28893206e+03]
#  [  5.24568220e+00   7.66090960e+03]
#  [  1.41868164e+00   1.42213588e+03]]


# 奇异值分解获得特征矩阵
U, S, V = np.linalg.svd(SIGMA)
print('U =', U, sep='\n')
# 主成分特征矩阵
U_reduce = U[:, 0]
print('U_reduce =', U_reduce, sep='\n')
# 降维样本
Z = X * U_reduce
print('Z =', Z, sep='\n')
# 恢复到归一化缩放后的样本
X_approx = Z * U_reduce.T
print('X_approx =', X_approx, sep='\n')
# 恢复到原始样本
A_approx = np.multiply(X_approx, s) + mu
print('A_approx =', A_approx, sep='\n')

y=np.linalg.eig(SIGMA)
print(y)
