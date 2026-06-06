import numpy as np
import os
import glob
from sar_core import process_sar_data
from tqdm import tqdm

# 配置
DATA_DIR = 'data'
SAVE_DIR = 'dataset'
PATCH_SIZE = 256
STRIDE = 64  # 重叠切片增加数据量

os.makedirs(SAVE_DIR, exist_ok=True)


def create_dataset():
    scene_files = [f'Echo_Scene{i}.dat' for i in range(1, 7)]  # Scene 1-6

    all_inputs = []
    all_labels = []

    # 用于计算归一化参数
    global_max_val = 0.0

    print("Step 1: 处理场景并切片...")
    for fname in scene_files:
        full_path = os.path.join(DATA_DIR, fname)
        print(f"Processing {fname}...")

        try:
            # 获取复数图像
            img_in, img_lbl, _ = process_sar_data(full_path)

            # 更新最大值用于后续归一化 (使用标签的幅度)
            current_max = np.percentile(np.abs(img_lbl), 99.9)
            global_max_val = max(global_max_val, current_max)

            # 滑动窗口切片
            H, W = img_in.shape
            for r in range(0, H - PATCH_SIZE + 1, STRIDE):
                for c in range(0, W - PATCH_SIZE + 1, STRIDE):
                    # 提取 Patch
                    patch_in = img_in[r:r + PATCH_SIZE, c:c + PATCH_SIZE]
                    patch_lbl = img_lbl[r:r + PATCH_SIZE, c:c + PATCH_SIZE]

                    # 转换为 (2, H, W) 格式: Channel 0=Real, Channel 1=Imag
                    # 这里先不归一化，训练时再除以 global_max_val
                    inp_stack = np.stack([patch_in.real, patch_in.imag], axis=0)
                    lbl_stack = np.stack([patch_lbl.real, patch_lbl.imag], axis=0)

                    all_inputs.append(inp_stack)
                    all_labels.append(lbl_stack)

        except Exception as e:
            print(f"Skipping {fname}: {e}")

    # 转换为 numpy 数组
    all_inputs = np.array(all_inputs, dtype=np.float32)
    all_labels = np.array(all_labels, dtype=np.float32)

    print(f"生成了 {len(all_inputs)} 个切片")
    print(f"全局归一化系数 (Max Abs): {global_max_val}")

    # 保存
    np.save(os.path.join(SAVE_DIR, 'train_input.npy'), all_inputs)
    np.save(os.path.join(SAVE_DIR, 'train_label.npy'), all_labels)
    np.save(os.path.join(SAVE_DIR, 'norm_factor.npy'), np.array(global_max_val))
    print("数据集保存完毕。")


if __name__ == "__main__":
    create_dataset()