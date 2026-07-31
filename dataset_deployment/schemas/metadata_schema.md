# metadata schema

推荐字段：

- `case_id`
- `slide_id`
- `age`
- `site`
- `survival_months`
- `censorship`
- `is_female`
- `oncotree_code`
- `rna_file_name`

约定：

- `case_id` 为患者级 TCGA submitter id
- `slide_id` 保留原始 `.svs` 文件名
- `rna_file_name` 保存原始 RNA 源文件名，缺失时留空
