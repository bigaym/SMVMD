
import numpy as np


def SMVMD(signal, alphaMin, alphaMax, beta, init, tau, eps1, eps2, K):
    # Successive Multivariate Variational Mode Decomposition
    # Input and Parameters:
    # ---------------------
    # signal    - input multivariate signal that needs to be decomposed
    # alphaMin  - the min parameter that defines the bandwidth of extracted modes
    # alphaMax  - the max parameter that defines the bandwidth of extracted modes
    #             (low value of alpha yields higher bandwidth)
    # beta      - the change rate of alpha > 1
    # init      - 0 = the first omega start at 0
    #           - 1 = the first omega initialized randomly
    # tau       - time-step of the dual ascent ( pick 0 for noise-slack )
    # eps1      - tolerance value for convergence of ADMM
    # eps2      - tolerance value for convergence of ternimation
    # K        - the maximum of mode number
    #
    #
    # Output:
    # ----------------------
    # u       - the collection of decomposed modes
    # u_hat   - spectra of the modes
    # omega   - estimated mode center-frequencies

    ## Check for getting number of channels from input signal
    row, col = signal.shape
    if row > col:
        C = col  # number of channels
        T = row  # length of the Signal
        signal = signal.T
    else:
        C = row  # number of channels
        T = col  # length of the Signal

    ## ---------- Preparations
    # Mirroring
    flag = False
    f = np.zeros((C, 2 * T))
    if T % 2 == 0:
        flag = True
        f[:, :T // 2] = -signal[:, T // 2 - 1 ::-1]
        f[:, T // 2:T // 2 + T] = signal
        f[:, T // 2 + T:2 * T] = -signal[:, T - 1:T // 2 - 1:-1]
    else:
        f[:, :(T - 1) // 2] = -signal[:, T // 2 - 1 ::-1]
        f[:, (T + 1) // 2:(T + 1) // 2 + T] = signal
        f[:, (T + 1) // 2 + T:2 * T] = -signal[:, T:T // 2:- 1]

    # Time Domain 0 to T (of mirrored signal)
    T = f.shape[1]
    t = np.arange(1, T + 1) / T

    # frequency points
    fp = t - 0.5 - 1 / T
    fp = np.asarray(fp[T // 2:])

    # Construct and center f_hat
    f_hat = np.fft.fftshift(np.fft.fft(f, axis=1), axes=1)
    f_hat_plus = f_hat[:, T // 2:]

    ## ------------ 外部初始化
    u_hat_plus = np.zeros((K, T // 2, C), dtype=np.complex128)  # 存储模式的频域结果
    omega = np.zeros(K)  # 中心频率存储
    omega_iter = [[] for _ in range(K)]  # 中心频率迭代历程存储
    k = 0  # 模式个数
    sum_uk = 0  # 前k-1个模式和
    sum_fuk = 0  # 前k-1个滤波因子和
    eps = np.finfo(np.float64).eps
    tol2 = eps2 + eps  # 算法终止阈值

    ## ----------- 执行算法求解
    while tol2 > eps2 and k < K:
        k += 1  # 模式数加一

        ## -----------模式k提取初始化参数设置
        if init == 0:
            omega[k - 1] = 0.25  # 归一化频率中点初始化
        else:
            omega[k - 1] = np.random.rand() / 2.56  # 随机初始化

        omega_iter[k - 1].append(omega[k - 1])  # 记录k个模式中心频率初始值
        alpha = alphaMin  # 初始带宽控制参数alpha
        lambda_hat = np.zeros((C, T // 2))  # 初始化拉格朗日乘子
        tol1 = eps1 + eps  # 模式k更新终止阈值
        n = 1  # 初始化迭代计数器
        uk_hat_c = np.zeros((C, T // 2), dtype=np.complex128)  # 初始化当前模式频域结果
        uk_hat_n = np.zeros((C, T // 2), dtype=np.complex128)  # 初始化下一个模式频域结果
        filt_uk = alpha * (fp - omega[k - 1]) ** 2  # 初始化第k个模式滤波器

        ## ----------- 迭代更新提取第k个模式
        while tol1 > eps1 and n < 100:
            # 更新第k个多变量模式
            for c in range(C):
                uk_hat_n[c, :] = (f_hat_plus[c, :] + (filt_uk ** 2) * uk_hat_c[c, :] + lambda_hat[c, :] / 2) / (
                        (1 + (filt_uk ** 2)) * (1 + 2 * filt_uk + sum_fuk)
                )

            # 更新中心频率
            numerator = np.sum((np.abs(uk_hat_n.T) ** 2)*np.asarray(fp).reshape(-1, 1))
            denominator = np.sum(np.abs(uk_hat_n.flatten()) ** 2)
            omega[k - 1] = numerator / denominator

            filt_uk = alpha * (fp - omega[k - 1]) ** 2

            # Dual ascent
            lambda_hat = lambda_hat + tau * (
                    (f_hat_plus - uk_hat_n + lambda_hat / 2) / (1 + filt_uk ** 2) - lambda_hat / 2
            )

            # converged yet?
            tol1 = np.sum(np.abs(uk_hat_n - uk_hat_c) ** 2) / (np.sum(np.abs(uk_hat_c) ** 2) + eps)
            uk_hat_c = uk_hat_n

            alpha = min(alpha * beta, alphaMax)

            n += 1  # 循环计数器加一
            omega_iter[k - 1].append(omega[k - 1])  # 记录k个模式中心频率迭代历程

        sum_uk = sum_uk + uk_hat_c
        sum_fuk = sum_fuk + 1 / (filt_uk ** 2)
        u_hat_plus[k - 1, :, :] = uk_hat_c.T

        # converged yet?
        tol2 = np.sum(np.abs(f_hat_plus - sum_uk) ** 2) / np.sum(np.abs(f_hat_plus) ** 2)

    ## ------ Post-processing and cleanup
    # discard the last item, which maybe a noise item
    K = k
    omega = omega[:K]
    omega_iter = omega_iter[:K]
    u_hat_plus = u_hat_plus[:K, :, :]

    # Signal reconstruction
    u_hat = np.zeros((K, T, C), dtype=np.complex128)
    u_hat[:, T // 2:, :] = u_hat_plus
    u_hat[:, T // 2:0:-1, :] = np.conj(u_hat_plus)
    u_hat[:, 0, :] = np.conj(u_hat[:, -1, :])

    u = np.zeros((K, T, C))
    for k in range(K):
        for c in range(C):
            u[k, :, c] = np.real(np.fft.ifft(np.fft.ifftshift(u_hat[k, :, c])))

    # remove mirror part
    if flag:
        u = u[:, T // 4:3 * T // 4, :]
    else:
        x = u[:, (T + 2) // 4:(3 * T + 2) // 4, :]
        u = u[:, (T + 2) // 4:(3 * T + 2) // 4, :]

    # recompute spectrum
    u_hat = np.zeros((K, np.max(signal.shape), C), dtype=np.complex128)
    for k in range(K):
        for c in range(C):
            u_hat[k, :, c] = np.fft.fftshift(np.fft.fft(u[k, :, c]))

    return u, u_hat, omega_iter


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from scipy.io import loadmat

    fn = '../data.mat'
    semg = loadmat(fn)['data']
    # print(semg.keys())

    data = semg[10, :, :]

    del semg

    print(data.shape)
    u, _, _ = SMVMD(
        signal=data,
        K=5,
        tau=0,
        alphaMin=10,
        alphaMax=10000,
        beta=1.5,
        eps1=1e-6,
        eps2=1e-6,
        init=0
    )
    print(u.shape)
    # a = np.random.rand(10).reshape(2, 5)
    # print(a)
    # b = a[:, 3::-1]
    # print(b)
    fig, ax = plt.subplots(10, 1, figsize=(10, 12))
    for i in range(10):
        ax[i].plot(data[:, i])
        ax[i].plot(np.sum(u[:, :, i], axis=0))

    plt.show()