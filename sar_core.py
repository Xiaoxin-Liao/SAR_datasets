import numpy as np
import struct
import os


# --- 1. 读取函数 (保持不变) ---
def read_echo_file(filename):
    if not os.path.exists(filename):
        raise FileNotFoundError(f"文件不存在: {filename}")
    file_size = os.path.getsize(filename)

    with open(filename, 'rb') as fid:
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

        dtype = np.int16 if head['SampleBits'] == 16 else np.int8 if head['SampleBits'] == 8 else np.float32
        bytes_per_sample = 2 if head['SampleBits'] == 16 else 1 if head['SampleBits'] == 8 else 4

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


# --- 2. 成像算法核心 (含 SRC) ---
def process_sar_data(filename):
    """
    更新后的处理流程：
    img_input: 仅距离+方位压缩 (无 SRC, 无 RCMC)
    img_label: 全流程 (SRC + RCMC + 方位压缩)
    """
    raw_echo, head = read_echo_file(filename)
    Nr, Na = raw_echo.shape
    C, Fc = 299792458.0, head['CenterFrequency']
    Fs, Tp = head['SampleRate'], head['PulseWidth']
    BW, PRF = head['BandWidth'], head['PRF']
    R_near, V_plat = head['RangeNear'], 400.0                    #若要适配不同平台速度，需手动改变
    Lambda = C / Fc
    dr = C / (2 * Fs)

    # 坐标轴
    fr = np.fft.fftfreq(Nr, d=1 / Fs)  # 距离频率
    fa = np.fft.fftshift(np.fft.fftfreq(Na, d=1 / PRF))  # 方位频率
    r_axis = R_near + np.arange(Nr) * dr

    # ==========================
    # Step 1: 距离压缩
    # ==========================
    Kr = BW / Tp
    N_ref = int(np.round(Tp * Fs))
    t_ref = np.arange(-N_ref // 2, N_ref // 2) / Fs
    replica = np.exp(1j * np.pi * Kr * t_ref ** 2)  # 注意：这里通常用 conj, 但你提供的代码是用正相位匹配
    H_range = np.fft.fft(replica, n=Nr)
    S_range = np.fft.fft(raw_echo, n=Nr, axis=0)
    s_rc = np.fft.ifft(S_range * H_range.reshape(-1, 1), axis=0)

    # ==========================
    # Step 2: 变换到 RD 域
    # ==========================
    s_rd = np.fft.fftshift(np.fft.fft(s_rc, axis=1), axes=1)

    # 计算 D(fa)
    with np.errstate(invalid='ignore'):
        D_fa = np.sqrt(1 - (Lambda * fa / (2 * V_plat)) ** 2)
    D_fa = np.nan_to_num(D_fa, nan=1.0)

    # 准备方位相位
    R0_mat = r_axis.reshape(-1, 1)
    D_fa_mat = D_fa.reshape(1, -1)
    Phase_Az = 4 * np.pi * R0_mat * D_fa_mat / Lambda
    Haz = np.exp(1j * Phase_Az)

    # ==========================
    # 生成 Input (无 SRC, 无 RCMC)
    # ==========================
    s_ac_simple = s_rd * Haz
    img_input = np.fft.ifft(np.fft.ifftshift(s_ac_simple, axes=1), axis=1)

    # ==========================
    # 生成 Label (SRC + RCMC)
    # ==========================

    # [SRC] 二次距离压缩
    S_2D_Freq = np.fft.fft(s_rd, axis=0)  # RD -> 2D Freq
    fr_mat = fr.reshape(-1, 1)

    # SRC 相位补偿公式
    Phase_SRC = -np.pi * (fr_mat ** 2) / Kr * (1.0 / D_fa_mat - 1.0)
    H_src = np.exp(1j * Phase_SRC)

    S_2D_SRC = S_2D_Freq * H_src
    s_rd_src = np.fft.ifft(S_2D_SRC, axis=0)  # 2D Freq -> RD (SRC done)

    # [RCMC] 距离徙动校正
    s_rcmc = np.zeros_like(s_rd_src)
    orig_indices = np.arange(Nr)

    for k in range(Na):
        d_val = D_fa[k]
        if d_val == 0: d_val = 1.0

        factor = 1.0 / d_val
        shift_base = (R_near / dr) * (factor - 1.0)
        indices_to_sample = orig_indices * factor + shift_base

        # 线性插值
        col_data = s_rd_src[:, k]
        s_rcmc[:, k] = np.interp(indices_to_sample, orig_indices, col_data.real, left=0, right=0) + \
                       1j * np.interp(indices_to_sample, orig_indices, col_data.imag, left=0, right=0)

    # [方位压缩]
    s_ac_full = s_rcmc * Haz
    img_label = np.fft.ifft(np.fft.ifftshift(s_ac_full, axes=1), axis=1)

    # ==========================
    # 后处理: 切除无效区
    # ==========================
    pulse_samples = int(Tp * Fs)
    margin = 0  # 稍微加大一点 margin 确保安全
    crop_start = pulse_samples + margin

    if crop_start < Nr:
        img_input = img_input[crop_start:, :]
        img_label = img_label[crop_start:, :]

    return img_input, img_label, head


# --- 追加到 sar_core.py 的末尾 ---

def process_sar_data_with_noise(filename, snr_db=None):
    """
    带信噪比控制的SAR成像处理。
    snr_db: 设定的信噪比(dB)。如果为None，则不添加噪声。
    返回: img_input(无SRC/RCMC的输入), img_label(全流程RD标签), head
    """
    raw_echo, head = read_echo_file(filename)

    # ==========================
    # 注入高斯白噪声 (AWGN)
    # ==========================
    if snr_db is not None:
        # 计算信号功率
        signal_power = np.mean(np.abs(raw_echo) ** 2)
        # 根据 SNR 计算噪声功率 (SNR = 10 * log10(Ps / Pn))
        noise_power = signal_power / (10 ** (snr_db / 10))
        # 生成复高斯白噪声
        noise = np.sqrt(noise_power / 2) * (np.random.randn(*raw_echo.shape) + 1j * np.random.randn(*raw_echo.shape))
        raw_echo = raw_echo + noise

    Nr, Na = raw_echo.shape
    C, Fc = 299792458.0, head['CenterFrequency']
    Fs, Tp = head['SampleRate'], head['PulseWidth']
    BW, PRF = head['BandWidth'], head['PRF']
    R_near, V_plat = head['RangeNear'], 400.0
    Lambda = C / Fc
    dr = C / (2 * Fs)

    fr = np.fft.fftfreq(Nr, d=1 / Fs)
    fa = np.fft.fftshift(np.fft.fftfreq(Na, d=1 / PRF))
    r_axis = R_near + np.arange(Nr) * dr

    # Step 1: 距离压缩
    Kr = BW / Tp
    N_ref = int(np.round(Tp * Fs))
    t_ref = np.arange(-N_ref // 2, N_ref // 2) / Fs
    replica = np.exp(1j * np.pi * Kr * t_ref ** 2)
    H_range = np.fft.fft(replica, n=Nr)
    S_range = np.fft.fft(raw_echo, n=Nr, axis=0)
    s_rc = np.fft.ifft(S_range * H_range.reshape(-1, 1), axis=0)

    # Step 2: 变换到 RD 域
    s_rd = np.fft.fftshift(np.fft.fft(s_rc, axis=1), axes=1)
    with np.errstate(invalid='ignore'):
        D_fa = np.sqrt(1 - (Lambda * fa / (2 * V_plat)) ** 2)
    D_fa = np.nan_to_num(D_fa, nan=1.0)
    R0_mat = r_axis.reshape(-1, 1)
    D_fa_mat = D_fa.reshape(1, -1)
    Phase_Az = 4 * np.pi * R0_mat * D_fa_mat / Lambda
    Haz = np.exp(1j * Phase_Az)

    # 生成 Input (无 SRC, 无 RCMC)
    s_ac_simple = s_rd * Haz
    img_input = np.fft.ifft(np.fft.ifftshift(s_ac_simple, axes=1), axis=1)

    # 生成 Label (SRC + RCMC)
    S_2D_Freq = np.fft.fft(s_rd, axis=0)
    fr_mat = fr.reshape(-1, 1)
    Phase_SRC = -np.pi * (fr_mat ** 2) / Kr * (1.0 / D_fa_mat - 1.0)
    H_src = np.exp(1j * Phase_SRC)
    S_2D_SRC = S_2D_Freq * H_src
    s_rd_src = np.fft.ifft(S_2D_SRC, axis=0)

    s_rcmc = np.zeros_like(s_rd_src)
    orig_indices = np.arange(Nr)
    for k in range(Na):
        d_val = D_fa[k]
        if d_val == 0: d_val = 1.0
        factor = 1.0 / d_val
        shift_base = (R_near / dr) * (factor - 1.0)
        indices_to_sample = orig_indices * factor + shift_base
        col_data = s_rd_src[:, k]
        s_rcmc[:, k] = np.interp(indices_to_sample, orig_indices, col_data.real, left=0, right=0) + \
                       1j * np.interp(indices_to_sample, orig_indices, col_data.imag, left=0, right=0)

    # 方位压缩
    s_ac_full = s_rcmc * Haz
    img_label = np.fft.ifft(np.fft.ifftshift(s_ac_full, axes=1), axis=1)

    # 切除无效区
    pulse_samples = int(Tp * Fs)
    crop_start = pulse_samples
    if crop_start < Nr:
        img_input = img_input[crop_start:, :]
        img_label = img_label[crop_start:, :]

    return img_input, img_label, head