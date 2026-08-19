# CSV Split ChangeLog

## HGCN
- `models/missing_modality_baselines/third_party/HGCN/HGCN_code/train.py`
- 删除了 `seed_fit_split.pkl` 读取入口，不再依赖 pkl split。
- 新增 `--split_root` 和 `--split_dir`。
- `if_fit_split=True` 时，改为按 fold 读取 `splits_{fold}.csv`。
- 读取逻辑支持带/不带 `Unnamed: 0` 索引列的 CSV。

## Flex-MoE
- `models/missing_modality_baselines/third_party/flex-moe/data.py`
- 删除了 `PTID_splits.json` / `PTID_splits_mimic.json` 的读取入口。
- 新增 `_resolve_split_csv_path()` 和 `_load_split_csv()`。
- `load_and_preprocess_data()` 与 `load_and_preprocess_data_mimic()` 现在都从 CSV 读 `train / val / test` 三列。
- 读取逻辑同样兼容带/不带 `Unnamed: 0` 索引列的 CSV。

## Flex-MoE 入口参数
- `models/missing_modality_baselines/third_party/flex-moe/main.py`
- 新增 `--split_csv`，用于显式指定 split CSV 文件路径。

## 结果
- 两个 dataloader 现在都只走 CSV split 解析，不再走 pkl/json split 解析。
