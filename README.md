# pachong_model

可直接执行的抖音 / 小红书评论抓取与数据集构建项目。

## 安装

```bash
pip install -r requirements.txt
python -m playwright install chromium
```

## 抓数据

单链接：

```bash
python scripts/run_crawler.py \
  --platform douyin \
  --url "https://www.douyin.com/video/xxxxxxxxxxxx" \
  --max-comments 200
```

批量链接：

```bash
python scripts/run_crawler.py \
  --url-file urls.txt \
  --platform auto
```

如需携带登录态，把浏览器导出的 `storage_state` 或 cookies 文件填到 `configs/config.yaml`，或者命令行传：

```bash
python scripts/run_crawler.py \
  --platform xiaohongshu \
  --url-file xhs_urls.txt \
  --storage-state data/browser_state/xhs_state.json
```

抓取后会在 `data/raw/` 生成：

- `*_records.json`
- `*_records.jsonl`
- `*_comments.csv`
- `*_summary.json`

## 构建训练集

```bash
python scripts/build_dataset.py
```

输出在 `data/processed/`，默认包含 `train/val/test` 的 `jsonl` 和 `csv`。

## 训练模型

默认自动选择后端。

- 有 Paddle / PaddleNLP：走 Paddle
- 没有 Paddle：自动回退 `scikit-learn`

```bash
python scripts/train_classifier.py \
  --train-file data/processed/dataset_train.jsonl \
  --val-file data/processed/dataset_val.jsonl \
  --test-file data/processed/dataset_test.jsonl
```

## 预测和排序

```bash
python scripts/predict_quality.py \
  --model-dir checkpoints/best_model \
  --input-file data/processed/dataset_test.jsonl \
  --output-file data/processed/predictions
```

## 目录

- `scripts/run_crawler.py`: 抓取入口
- `scripts/build_dataset.py`: 预处理与伪标签
- `scripts/train_classifier.py`: 训练入口
- `scripts/predict_quality.py`: 预测与排序
- `scripts/bootstrap_server.sh`: 服务器初始化
- `scripts/run_server_pipeline.sh`: 服务器整条流水线
- `configs/config.server.yaml`: 服务器推荐配置
