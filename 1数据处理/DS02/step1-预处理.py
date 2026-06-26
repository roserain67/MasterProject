'''
DS02数据集说明：
1.数据由商用模块化航空推进系统仿真（C-MAPSS）动态模型合成生成；
2.包含9台发动机的完整飞行的退化轨迹；
3.包含2种失效模式
（1）HPT_eff退化（训练集 units=2,5 10--对应V_var的索引：6 )
（2）HPT_eff退化与LPT_eff、LPT_flow退化相结合(训练集 units=16,18,20--对应V_var的索引：6 8 9 和 测试集 units=11,14,15--对应V_var的索引：6 8 9)）
4.数据包含完整运行到故障轨迹的多变量传感器读数，共有14个传感器的数据；
5.记录停止在发动机失效的周期/时间，共有650万个时间戳。
7.训练集6个（units=2,5,10,16,18,20）
  测试集3个（units=11,14,15）
'''
import h5py
import time
import numpy as np
import torch
import warnings

warnings.filterwarnings('ignore')

# 检测是否有可用的 GPU
if torch.cuda.is_available():
    device = torch.device('cuda')  # 将设备设置为 GPU
    print('CUDA 可用，将在 GPU 上运行')
else:
    device = torch.device('cpu')  # 将设备设置为 CPU
    print('CUDA 不可用，将在 CPU 上运行')

# 设置文件位置
filename = './N-CMAPSS_DS05.h5'  # 读取DS02子集

# 【第一块】读取原始数据
# Time tracking, Operation time (min):  0.003
t = time.process_time()

# Load data
with h5py.File(filename, 'r') as hdf:
    # Development set（训练集）
    W_dev = np.array(hdf.get('W_dev'))  # W,工况 (5263447, 4)
    X_s_dev = np.array(hdf.get('X_s_dev'))  # X_s，测量信号 (5263447, 14)
    X_v_dev = np.array(hdf.get('X_v_dev'))  # X_v，虚拟信号， (5263447, 14)
    T_dev = np.array(hdf.get('T_dev'))  # T,健康指标，(流量&效率的函数) （注意：寿命预测暂不需要） (5263447, 10)
    Y_dev = np.array(hdf.get('Y_dev'))  # RUL (in cycles) ((5263447, 1)
    A_dev = np.array(hdf.get('A_dev'))  # Auxiliary (5263447, 4)

    # Test set（测试集）
    W_test = np.array(hdf.get('W_test'))  # W (1253743, 4)
    X_s_test = np.array(hdf.get('X_s_test'))  # X_s (1253743, 14)
    X_v_test = np.array(hdf.get('X_v_test'))  # X_v (1253743, 14)
    T_test = np.array(hdf.get('T_test'))  # T (1253743, 10)
    Y_test = np.array(hdf.get('Y_test'))  # RUL (1253743, 1)
    A_test = np.array(hdf.get('A_test'))  # Auxiliary (1253743, 4)

    # Varnams，表头
    W_var = np.array(hdf.get('W_var'))  # 4列，W: alt、Mach、TRA、T2
    X_s_var = np.array(hdf.get('X_s_var'))  # 14列，X_s: T24、T30、T48、T50、P15、P2、P21、P24、Ps30、P40、P50、Nf、Nc、Wf
    X_v_var = np.array(hdf.get('X_v_var'))  # 14列， X_v: T40、P30、P45、W21、W22、W25、W31、W32、W48、W50、SmFan、SmLPC、SmHPC、phi
    T_var = np.array(hdf.get(
        'T_var'))  # 10列，健康参数：fan_eff_mod、fan_flow_mod、 LPC_eff_mod、LPC_flow_mod、 HPC_eff_mod、HPC_flow_mod、 HPT_eff_mod、 HPT_flow_mod、 LPT_eff_mod、LPT_flow_mod
    A_var = np.array(hdf.get('A_var'))  # 4列，A: units、cycle、Fc、hs

    # from np.array to list dtype U4/U5
    W_var = list(np.array(W_var, dtype='U20'))
    X_s_var = list(np.array(X_s_var, dtype='U20'))
    X_v_var = list(np.array(X_v_var, dtype='U20'))
    T_var = list(np.array(T_var, dtype='U20'))
    A_var = list(np.array(A_var, dtype='U20'))

W = np.concatenate((W_dev, W_test), axis=0)  # (6517190, 4)  # 工况
X_s = np.concatenate((X_s_dev, X_s_test), axis=0)  # (6517190, 14)  # 测量传感器
X_v = np.concatenate((X_v_dev, X_v_test), axis=0)  # (6517190, 14)  # 虚拟传感器
T = np.concatenate((T_dev, T_test), axis=0)  # (6517190, 10)  # 健康指标
Y = np.concatenate((Y_dev, Y_test), axis=0)  # (6517190, 1)  # Rul
A = np.concatenate((A_dev, A_test), axis=0)  # (6517190, 4)  # 辅助列表

np.savetxt('W.csv', W, delimiter=',', fmt='%f', header='', comments='')
np.savetxt('X_s.csv', X_s, delimiter=',', fmt='%f', header='', comments='')
np.savetxt('X_v.csv', X_v, delimiter=',', fmt='%f', header='', comments='')
np.savetxt('T.csv', T, delimiter=',', fmt='%f', header='', comments='')
np.savetxt('Y.csv', Y, delimiter=',', fmt='%f', header='', comments='')
np.savetxt('A.csv', A, delimiter=',', fmt='%f', header='', comments='')

# 输出各参数的规格
print('')
print("Operation time (min): ", (time.process_time() - t) / 60)
print('')
print("W shape: " + str(W.shape))  # 读取矩阵的长度
print("X_s shape: " + str(X_s.shape))
print("X_v shape: " + str(X_v.shape))
print("T shape: " + str(T.shape))
print("A shape: " + str(A.shape))
