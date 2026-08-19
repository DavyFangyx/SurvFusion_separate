# Baseline Integration Report 2

## HGCN: `avail -> use_type / in_mask`

- `forward()` 实际签名是 `forward(self, all_thing, train_use_type=None, use_type=None, in_mask=[], mix=False)`。
- 原码里没有 `avail` 入参；`use_type` 是调用方决定的，不是模型内部从 `avail` 推出来的。
- `forward()` 只看 `data_type = use_type`，然后按 `img / rna / cli` 三个分支选择性执行图卷积。
- `train_use_type` 只是“训练全集顺序”，默认是 `['img', 'rna', 'cli']`，不是一个自动和 `avail` 同步的对象。
- `use_type == train_use_type` 时，`pool_x` 直接拼接当前可见分支，没有显式零占位。
- `use_type != train_use_type` 时，代码会先建 `tmp_x = zeros(len(train_use_type), C)`，再按 `train_use_type` 顺序填入可见模态，缺失槽保留 0。
- `in_mask` 也不是 `~avail`。
- `in_mask` 为空时会变成全 `False`；训练时实际传入的是 `generate_mask(num=len(train_use_type))`，它是随机 mask，不是 availability 反转。

### 结论

- HGCN 不是“`avail` 直接映射到 `in_mask`”的原始设计。
- 若要原样迁移，`avail` 只能在 wrapper 层决定 `use_type` 和是否走 subset 分支，不能直接当成 `in_mask = ~avail`。

### 3 模态时的槽位

- 固定三槽只在 `use_type != train_use_type` 的 subset 路径里成立。
- 槽位顺序由 `train_use_type` 决定，默认就是 `[img, rna, cli]`。
- 所以“可用几个模态就只有几个槽”只发生在上游选择了 subset 时，不是模型内部始终如此。

## Flex-MoE: `batch_mcs` 编码

- `batch_mcs` 来自 `data_dict['modality_comb']`，`collate_fn` 只是把它 stack 成 batch。
- 它不是 binary availability vector，而是“模态子集 id”。
- 编码规则由 `get_modality_combinations(args.modality)` 决定：
  - 先按子集大小从大到小枚举；
  - 再把每个子集排序成字符串做 key；
  - `0` 永远是 full modality。

### 3 模态映射表

如果你用 3 个符号 `W/G/C` 表示 `wsi/gene/clinic`，则索引是：

| id | key |
|---|---|
| 0 | `CGW` |
| 1 | `GW` |
| 2 | `CW` |
| 3 | `CG` |
| 4 | `W` |
| 5 | `G` |
| 6 | `C` |

### 结论

- `batch_mcs` 不是按 `(wsi, gene, clinic)` 的二进制位编码。
- wrapper 必须固定一个 modality 字符顺序，并严格复用同一套 subset->id 规则，否则 `missing_embeds[batch_mcs]` 会错位。
