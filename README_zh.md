# LoudLoss

🎉 本文章已被 [21st NCMMSC](https://www.ncmmsc.org.cn/) 录用，期待与您现场交流！🤝

论文 [**Loud-loss: A Perceptually Motivated Loss Function for Speech Enhancement Based on Equal-Loudness Contours**](https://arxiv.org/abs/2511.05945) 的代码与预训练模型。[English](README.md)

包含 VoiceBank+DEMAND 上的三个 GTCRN Mapping 版本：MSE、LoudLoss、LoudLoss + SI-SNR。

## 安装

建议使用 Python 3.9；已验证 PyTorch 2.7.1+cu118。在本目录执行：

```bash
pip install -r requirements.txt
```

## 数据

获取 [VoiceBank+DEMAND](https://datashare.ed.ac.uk/handle/10283/2791)，准备 **16 kHz 单声道 WAV**，保持原始幅度：

```text
dataset/
├── test_noisy/
│   ├── p232_001.wav
│   └── ...
├── test_clean/
│   ├── p232_001.wav
│   └── ...
├── trainset_noisy/
│   ├── p227_001.wav
│   └── ...
└── trainset_clean/
    ├── p227_001.wav
    └── ...
```

已有 16 kHz 官方数据时，可按附带清单整理（仅测试用 `--split test`）：

```bash
python prepare_data.py --source-root /path/to/voicebank16k --output-dir /path/to/dataset --split all
export DATA_ROOT=/path/to/dataset
```

脚本不重采样。

## 测试

权重已包含在 `checkpoints/`，以下命令自动加载对应模型：

```bash
python evaluate.py --config configs/mse.yaml --data-root "$DATA_ROOT" --device cuda --output-dir runs/mse
python evaluate.py --config configs/loudloss.yaml --data-root "$DATA_ROOT" --device cuda --output-dir runs/loudloss
python evaluate.py --config configs/loudloss_sisnr.yaml --data-root "$DATA_ROOT" --device cuda --output-dir runs/loudloss_sisnr
```

CPU 测试改用 `--device cpu`；添加 `--limit 8` 可快速检查。每次使用新的输出目录，结果包含逐句指标和汇总。

三个模型均已完成全部 824 条测试：

| 版本 | 论文 WB-PESQ | 实测 WB-PESQ |
|---|---:|---:|
| MSE | 2.17 | 2.169840 |
| LoudLoss | 2.93 | 2.927940 |
| LoudLoss + SI-SNR | 2.92 | 2.919758 |

## 训练

```bash
python train.py --config configs/loudloss.yaml --data-root "$DATA_ROOT" --device cuda --output-dir runs/train-loudloss
```

其他版本替换为 `configs/mse.yaml` 或 `configs/loudloss_sisnr.yaml`。训练保存 `best.ckpt` 和 `last.ckpt`；`--resume` 可恢复本版本的训练检查点。

默认沿用历史数据划分：10,802 条训练数据，测试集同时用于验证与检查点选择。独立验证集可通过 `--val-manifest` 指定。

## 使用 LoudLoss

输入为 16 kHz 波形，形状为 `[batch, 1, samples]`：

```python
from loudloss import LoudLoss

loss, diagnostics = LoudLoss()(pred_waveform, clean_waveform)
loss.backward()
```

## 训练log

```bash
tensorboard --logdir logs
```

## 许可

自有代码与权重仅允许非商业研究和教学使用，见 [LICENSE](LICENSE)。GTCRN 的 MIT 许可一并保留在该文件中。

## 致谢

感谢 [GTCRN](https://github.com/Xiaobin-Rong/gtcrn) 提供的代码。

## 引用

如果我们的工作对您有帮助，请引用：

```bibtex
@article{li2025loud,
  title={Loud-loss: A perceptually motivated loss function for speech enhancement based on equal-loudness contours},
  author={Li, Zixuan and Zhang, Xueliang and Zhao, Changjiang and Gao, Shuai and Miao, Lei and Yan, Zhipeng and Sun, Ying and Zhu, Chong},
  journal={arXiv preprint arXiv:2511.05945},
  year={2025}
}
```
