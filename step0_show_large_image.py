import numpy as np
import matplotlib.pyplot as plt
import struct
import os


# --- 1. 读取函数 (保持不变) ---
def read_echo_file(filename):
    if not os.path.exists(filename):
        raise FileNotFoundError(f"文件不存在: {filename}")
    file_size = os.path.getsize(filename)

    with open(filename, 'rb') as fid:
        # 读取头文件关键参数
        header_bytes = fid.read(512)
        ptr = 0
        head = {}
        head['Flag'] = struct.unpack_from('I', header_bytes, ptr)[0];
        ptr += 4
        head['CenterFrequency'] = struct.unpack_from('d', header_bytes, ptr)[0];
        ptr += 8
        head['BandWidth'] = struct.unpack_from('d', header_bytes, ptr)[0];
        ptr += 8
        head['SampleRate'] = struct.unpack_from('d', header_bytes, ptr)[0];
        ptr += 8
        head['SampleBits'] = struct.unpack_from('i', header_bytes, ptr)[0];
        ptr += 4
        head['PulseWidth'] = struct.unpack_from('d', header_bytes, ptr)[0];
        ptr += 8
        head['PRF'] = struct.unpack_from('d', header_bytes, ptr)[0];
        ptr += 8
        head['RangeNear'] = struct.unpack_from('d', header_bytes, ptr)[0];
        ptr += 8
        head['RecLength'] = struct.unpack_from('i', header_bytes, ptr)[0];
        ptr += 4

        if head['SampleBits'] == 8:
            bytes_per_sample = 1
            dtype = np.int8
        elif head['SampleBits'] == 16:
            bytes_per_sample = 2
            dtype = np.int16
        else:
            bytes_per_sample = 4
            dtype = np.float32

        packet_size = 128 + 2 * head['RecLength'] * bytes_per_sample
        pulse_count = (file_size - 512) // packet_size

        raw_data = np.zeros((head['RecLength'], pulse_count), dtype=np.complex64)
        fid.seek(512)
        full_data = fid.read()

        for i in range(pulse_count):
            offset = i * packet_size
            data_offset = offset + 128
            data_len = 2 * head['RecLength'] * bytes_per_sample
            raw_bytes = full_data[data_offset: data_offset + data_len]
            col_data = np.frombuffer(raw_bytes, dtype=dtype)
            raw_data[:, i] = col_data[0::2] + 1j * col_data[1::2]

    return raw_data, head


# --- 2. 核心成像算法 ---
def rd_imaging_algorithm(filename):
    print(f"Step 1: 读取回波数据: {filename}")
    raw_echo, head = read_echo_file(filename)

    Nr, Na = raw_echo.shape
    C = 299792458.0
    Fc = head['CenterFrequency']
    Fs = head['SampleRate']
    Tp = head['PulseWidth']
    BW = head['BandWidth']
    PRF = head['PRF']
    R_near = head['RangeNear']
    V_plat = 400.0  # 平台速度
    Lambda = C / Fc

    # 坐标轴定义
    # 方位频率轴 (Doppler frequency axis)
    fa = np.fft.fftshift(np.fft.fftfreq(Na, d=1 / PRF))
    # 距离频率轴 (Range frequency axis) - 用于SRC
    fr = np.fft.fftfreq(Nr, d=1 / Fs)

    # 距离向时间轴 (Range Time)
    dr = C / (2 * Fs)
    r_axis = R_near + np.arange(Nr) * dr

    # ==========================================================
    # Step 2: 距离压缩 (Range Compression) - 两个流程公用
    # ==========================================================
    print("Step 2: 距离压缩...")
    Kr = BW / Tp
    N_ref = int(np.round(Tp * Fs))
    t_ref = np.arange(-N_ref // 2, N_ref // 2) / Fs

    # 生成匹配滤波器 (Conj of Chirp)
    # 信号通常是 exp(-j * pi * K * t^2) => 匹配滤波器是 exp(j...)
    replica = np.exp(1j * np.pi * Kr * t_ref ** 2)

    # 频域相乘
    H_range = np.fft.fft(replica, n=Nr)
    S_range = np.fft.fft(raw_echo, n=Nr, axis=0)
    s_rc = np.fft.ifft(S_range * H_range.reshape(-1, 1), axis=0)

    # ==========================================================
    # Step 3: 变换到 RD 域 (Range-Doppler Domain) - 公用
    # ==========================================================
    print("Step 3: 变换到 RD 域 (方位向FFT)...")
    # 注意：这里使用 fftshift 将零频移到中央，方便后续计算
    s_rd = np.fft.fftshift(np.fft.fft(s_rc, axis=1), axes=1)

    # 计算徙动因子 D(fa)
    # D(fa) = sqrt(1 - (lambda * fa / 2V)^2)
    with np.errstate(invalid='ignore'):
        D_fa = np.sqrt(1 - (Lambda * fa / (2 * V_plat)) ** 2)

    # 处理 D_fa 可能出现的 NaN (在波束外)
    D_fa = np.nan_to_num(D_fa, nan=1.0)

    # 准备方位压缩相位的公共部分
    # Phase = 4 * pi * R0 * D(fa) / Lambda
    R0_mat = r_axis.reshape(-1, 1)
    D_fa_mat = D_fa.reshape(1, -1)

    # 构造方位匹配滤波器 Haz (两个流程都会用到，但RCMC后对应的R0不同)
    # 注意：这里的相位是用于补偿回波相位的，回波相位是 -4pi*R/lambda，所以滤波器是 +
    # 但具体正负号取决于数据的 time/freq 定义。通常尝试调整符号直到聚焦。
    # 这里假设标准正向补偿。
    Phase_Az = 4 * np.pi * R0_mat * D_fa_mat / Lambda
    Haz = np.exp(1j * Phase_Az)

    # ##################################################################
    # 分支 A: 仅做 "距离脉压 + 方位脉压" (无 SRC, 无 RCMC)
    # ##################################################################
    print(">>> 分支 A: 处理 [仅距离+方位压缩] 结果...")
    # 直接在 RD 域乘以方位匹配滤波器
    # 此时数据里的距离弯曲还没校正，能量分布在弯曲的轨迹上
    s_ac_simple = s_rd * Haz

    # IFFT 回到图像域
    img_simple = np.fft.ifft(np.fft.ifftshift(s_ac_simple, axes=1), axis=1)

    # ##################################################################
    # 分支 B: 全流程 (SRC + RCMC + 方位脉压)
    # ##################################################################
    print(">>> 分支 B: 处理 [全流程: SRC + RCMC + 方位压缩] 结果...")

    # --- Step 4.1: SRC (Secondary Range Compression) ---
    # 针对大入射角，必须做 SRC。SRC 需在 2D 频域进行。
    print("   [B1] 执行 SRC (二阶距离压缩)...")

    # s_rd 当前是 [Range Time, Az Freq]。变回 [Range Freq, Az Freq]
    S_2D_Freq = np.fft.fft(s_rd, axis=0)

    fr_mat = fr.reshape(-1, 1)  # (Nr, 1)

    # SRC 滤波器构造
    # 补偿相位: - pi * fr^2 / Kr * (1/D(fa) - 1)
    # 这里的 Kr 是调频斜率。如果图像变模糊，尝试将下面的 Phase_SRC 里的负号去掉或反转。
    # (1/D - 1) 是正数，fr^2 是正数。通常需补偿一项二次相位。
    Phase_SRC = -np.pi * (fr_mat ** 2) / Kr * (1.0 / D_fa_mat - 1.0)
    H_src = np.exp(1j * Phase_SRC)

    # 应用 SRC 并变回 RD 域 (Range Time, Az Freq)
    S_2D_SRC = S_2D_Freq * H_src
    s_rd_src = np.fft.ifft(S_2D_SRC, axis=0)

    # --- Step 4.2: RCMC (Range Cell Migration Correction) ---
    print("   [B2] 执行 RCMC (距离徙动校正)...")
    s_rcmc = np.zeros_like(s_rd_src)
    orig_indices = np.arange(Nr)

    # 遍历每一个方位频率点进行校正
    for k in range(Na):
        d_val = D_fa[k]
        if d_val == 0: d_val = 1.0  # 保护

        # 徙动量计算: R(fa) = R0 / D(fa)
        # 我们要把 R(fa) 处的数据搬回到 R0 处
        # 变换关系: r_new = r_old * D + R_near * (D - 1) ?
        # 更直观的: old_index = new_index * (1/D) + offset

        # R_migration = R0 * (1/D - 1)
        # Shift in samples = R_migration / dr
        # target_idx = current_idx + shift
        # 也就是 data[current] 其实是来自更远的 range

        # 正确的重采样索引:
        # 我们希望 s_rcmc[r, k] 对应距离 r。
        # 而原始数据 s_rd_src 在该频率 k 下，距离 r 的能量跑到了 r/D(fa)
        # 不完全是 r/D，而是 R_total = R_ref(fa) + (n * dr).
        # 准确公式: R(fa) = R0(n) / D(fa) ≈ R0(n) + R0(n)*(1/D - 1)

        # 计算插值所需的旧索引位置
        # input_coords = output_coords * Scale + Shift
        factor = 1.0 / d_val
        shift_base = (R_near / dr) * (factor - 1.0)

        indices_to_sample = orig_indices * factor + shift_base

        # 线性插值 (为了速度，精度要求极高可用 sinc 插值)
        # 分别插值实部和虚部
        col_data = s_rd_src[:, k]
        s_rcmc[:, k] = np.interp(indices_to_sample, orig_indices, col_data.real, left=0, right=0) + \
                       1j * np.interp(indices_to_sample, orig_indices, col_data.imag, left=0, right=0)

    # --- Step 4.3: 方位压缩 (Azimuth Compression) ---
    print("   [B3] 执行方位压缩...")
    # 此时 RCMC 已把能量拉直，可以直接使用统一的 R0 向量进行匹配
    s_ac_full = s_rcmc * Haz

    # IFFT 回到图像域
    img_full = np.fft.ifft(np.fft.ifftshift(s_ac_full, axes=1), axis=1)

    return img_simple, img_full, head


# --- 3. 增强对比可视化函数 ---
def plot_comparison(img_simple, img_full, head):
    print("Step 5: 处理可视化 (切除无效区 + 99.9%截断)...")

    # --- 切除逻辑 (去除脉冲宽度的无效数据) ---
    pulse_samples = int(head['PulseWidth'] * head['SampleRate'])
    margin = 0
    crop_start = pulse_samples + margin

    def process_image(complex_img):
        # 1. 切除前端无效数据
        if crop_start < complex_img.shape[0]:
            img_crop = complex_img[crop_start:, :]
        else:
            img_crop = complex_img

        # 2. 取幅度并转置 (Range为横轴，Azimuth为纵轴)
        amp = np.abs(img_crop).T

        # 3. 动态范围截断 (Clamping) 以增强对比度
        # 使用 99.9% 分位点作为最大值，防止个别强点掩盖细节
        p99 = np.percentile(amp, 99.9)
        if p99 == 0: p99 = 1  # 防止除零
        amp_clipped = np.clip(amp, 0, p99)

        # 归一化
        amp_norm = amp_clipped / p99
        return amp_norm

    vis_simple = process_image(img_simple)
    vis_full = process_image(img_full)

    # --- 绘图 ---
    plt.figure(figsize=(16, 8))

    # 左图：基础结果
    plt.subplot(1, 2, 1)
    plt.imshow(vis_simple, cmap='gray', aspect='auto', origin='upper')
    plt.title(f'Basic Result (No RCMC, No SRC)\nRange+Az Compression Only\n(Defocus & Curvature visible)', fontsize=12)
    plt.xlabel('Range (samples)')
    plt.ylabel('Azimuth (lines)')

    # 右图：全流程结果
    plt.subplot(1, 2, 2)
    plt.imshow(vis_full, cmap='gray', aspect='auto', origin='upper')
    plt.title(f'Full RD Result (With SRC & RCMC)\nBeamInc=60 (Large Slant Range Correction)', fontsize=12)
    plt.xlabel('Range (samples)')
    plt.ylabel('Azimuth (lines)')

    plt.tight_layout()
    plt.show()


# --- 主程序入口 ---
if __name__ == "__main__":
    file_name = 'data/Echo_Scene5.dat'  # 请确保路径正确
    try:
        # 1. 执行算法
        img_res1, img_res2, head_info = rd_imaging_algorithm(file_name)

        # 2. 画图
        plot_comparison(img_res1, img_res2, head_info)

    except Exception as e:
        print(f"发生错误: {e}")
        import traceback

        traceback.print_exc()