# LoudLoss

🎉 Our paper has been accepted by [21st NCMMSC](https://www.ncmmsc.org.cn/). We look forward to meeting you there! 🤝

Code and pretrained models for [**Loud-loss: A Perceptually Motivated Loss Function for Speech Enhancement Based on Equal-Loudness Contours**](https://arxiv.org/abs/2511.05945). [中文](README_zh.md)

Includes three GTCRN magnitude-mapping experiments on VoiceBank+DEMAND: MSE, LoudLoss and LoudLoss + SI-SNR.

## Installation

Python 3.9 recommended; verified with PyTorch 2.7.1+cu118. Run from this directory:

```bash
pip install -r requirements.txt
```

## Data

Obtain [VoiceBank+DEMAND](https://datashare.ed.ac.uk/handle/10283/2791) and prepare **mono 16 kHz WAV** files without amplitude normalization:

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

Arrange an existing 16 kHz official dataset using the included manifests (`--split test` for evaluation only):

```bash
python prepare_data.py --source-root /path/to/voicebank16k --output-dir /path/to/dataset --split all
export DATA_ROOT=/path/to/dataset
```

The script does not resample.

## Evaluation

Pretrained weights are included in `checkpoints/` and loaded automatically:

```bash
python evaluate.py --config configs/mse.yaml --data-root "$DATA_ROOT" --device cuda --output-dir runs/mse
python evaluate.py --config configs/loudloss.yaml --data-root "$DATA_ROOT" --device cuda --output-dir runs/loudloss
python evaluate.py --config configs/loudloss_sisnr.yaml --data-root "$DATA_ROOT" --device cuda --output-dir runs/loudloss_sisnr
```

Use `--device cpu` for CPU evaluation or add `--limit 8` for a quick check. Use a new output directory for each run. Outputs include per-utterance metrics and means.

All three models were evaluated on all 824 test utterances:

| Model | Paper WB-PESQ | Reproduced WB-PESQ |
|---|---:|---:|
| MSE | 2.17 | 2.169840 |
| LoudLoss | 2.93 | 2.927940 |
| LoudLoss + SI-SNR | 2.92 | 2.919758 |

## Training

```bash
python train.py --config configs/loudloss.yaml --data-root "$DATA_ROOT" --device cuda --output-dir runs/train-loudloss
```

Use `configs/mse.yaml` or `configs/loudloss_sisnr.yaml` for the other objectives. Training saves `best.ckpt` and `last.ckpt`; `--resume` restores a checkpoint produced by this trainer.

Defaults retain the historical split: 10,802 training utterances, with the test set also used for validation and checkpoint selection. Use `--val-manifest` to specify an independent validation set.

## Using LoudLoss

Inputs are 16 kHz waveforms shaped `[batch, 1, samples]`:

```python
from loudloss import LoudLoss

loss, diagnostics = LoudLoss()(pred_waveform, clean_waveform)
loss.backward()
```

## Train log

```bash
tensorboard --logdir logs
```

## License

Original code and weights are for noncommercial research and education only; see [LICENSE](LICENSE). The upstream GTCRN MIT license is included in the same file.

## Acknowledgements

We thank [GTCRN](https://github.com/Xiaobin-Rong/gtcrn) for providing their code.

## Citation

If you find our work helpful, please cite:

```bibtex
@article{li2025loud,
  title={Loud-loss: A perceptually motivated loss function for speech enhancement based on equal-loudness contours},
  author={Li, Zixuan and Zhang, Xueliang and Zhao, Changjiang and Gao, Shuai and Miao, Lei and Yan, Zhipeng and Sun, Ying and Zhu, Chong},
  journal={arXiv preprint arXiv:2511.05945},
  year={2025}
}
```
