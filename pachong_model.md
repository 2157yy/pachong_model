# 爬虫 + LLM 数据获取方案文档

## 一、当前环境配置






### 1.1 硬件配置

| 项目 | 配置 |
|------|------|
| **CPU** | x86_64 架构 |
| **内存** | 1.5 TB (可用 1.4 TB) |
| **磁盘** | 3.5 TB (可用 2.6 TB) |
| **GPU** | **Iluvatar BI-V150S** × 1 |
| **显存** | 32 GB |
| **GPU 频率** | SM: 500MHz, Memory: 1600MHz |
| **设备 UUID** | GPU-da6d167f-927f-518f-8f27-c9d9d56e3e68 |

### 1.2 软件环境

| 项目 | 版本/状态 |
|------|-----------|
| **操作系统** | Ubuntu 20.04.6 LTS (Focal Fossa) |
| **内核版本** | 5.4.0-216-generic |
| **Python** | 3.10.10 |
| **PaddlePaddle** | 3.3.0 (GPU: iluvatar) |
| **paddle-iluvatar-gpu** | 3.3.0 |
| **paddleformers** | 0.4.0 |
| **Transformers** | 4.55.4 |
| **Playwright** | 1.53.0 |
| **BeautifulSoup4** | 4.14.3 |
| **Requests** | 2.32.5 |
| **IX-ML** | 4.3.8 |
| **Driver Version** | 4.3.8 |

### 1.3 环境说明

当前运行在**飞桨社区 Jupyter 容器环境**中 (`jupyter-16694338-10131723`)。

**GPU 状态**: ✅ Iluvatar BI-V150S 已就绪，32GB 显存可用

**查看显卡状态命令**:
```bash
ixsmi                    # 查看 GPU 实时状态（类似 nvidia-smi）
ixsmi -L                 # 列出 GPU 设备
```

**PaddlePaddle 使用 Iluvatar GPU**:
```python
import paddle
paddle.set_device('iluvatar_gpu:0')  # 设置使用 Iluvatar GPU
print(paddle.device.cuda.device_count())  # 查看可用 GPU 数量
```

---

## 二、GPU 训练能力评估

### 2.1 Iluvatar BI-V150S 32GB 训练能力

| 精度 | 可用显存 | 可训练最大模型 |
|------|----------|----------------|
| **FP32 (全精度)** | ~28GB | **~3B 参数** |
| **FP16 (混合精度)** | ~28GB | **~7B 参数** |
| **INT8 (量化)** | ~28GB | **~15B+ 参数** |

### 2.2 具体模型训练参考

| 模型 | 参数量 | FP32 显存需求 | FP16 显存需求 | 能否训练 |
|------|--------|---------------|---------------|----------|
| **ERNIE-base** | 110M | ~2GB | ~1GB | ✅ 轻松训练 |
| **ERNIE-large** | 340M | ~6GB | ~3GB | ✅ 轻松训练 |
| **BERT-large** | 340M | ~7GB | ~3.5GB | ✅ 轻松训练 |
| **ChatGLM-6B** | 6B | ~24GB | ~12GB | ✅ 可训练 |
| **LLaMA-7B** | 7B | ~28GB | ~14GB | ✅ FP16 可训练 |
| **LLaMA-13B** | 13B | ~52GB | ~26GB | ⚠️ FP16 勉强 |
| **ChatGLM-6B (LoRA)** | 6B | ~16GB | ~16GB | ✅ 可训练 |

### 2.3 本项目推荐配置

**对话质量分类任务**（三分类）推荐方案：

| 方案 | 模型 | 参数量 | 显存占用 | 推荐度 |
|------|------|--------|----------|--------|
| **方案 A (推荐)** | ERNIE-base | 110M | ~2GB | ⭐⭐⭐⭐⭐ |
| **方案 B** | ERNIE-large | 340M | ~6GB | ⭐⭐⭐⭐ |
| **方案 C (轻量)** | TextCNN+ERNIE | 50M | ~1GB | ⭐⭐⭐⭐ |
| **方案 D (大模型)** | ChatGLM-6B (LoRA) | 6B | ~16GB | ⭐⭐⭐ |

**推荐配置详情**:
```yaml
# ERNIE-base 训练配置
model: ernie-3.0-base
precision: FP16  # 混合精度训练
batch_size: 64   # 32GB 显存可设较大
max_length: 512
gradient_accumulation: 1

# 显存占用估算 (~5-6GB):
# - 模型参数：~220MB (FP16)
# - 梯度：~220MB
# - 优化器状态：~660MB (Adam)
# - 激活值 + 批次：~2-4GB
```

---

## 三、项目目标

训练一个智能爬虫模型，用于从**抖音**、**小红书**平台爬取优质对话数据，辅助对话模型训练。

### 3.1 数据需求

| 平台 | 目标数据 | 字段说明 |
|------|----------|----------|
| **抖音** | 视频评论、回复对 | 文案、评论、回复、点赞数、时间 |
| **小红书** | 笔记评论、回复对 | 标题、正文、评论、回复、点赞/收藏数 |

### 3.2 优质数据标准

| 维度 | 标准 |
|------|------|
| **互动量** | 点赞数 > 100, 收藏数 > 50, 评论数 > 20 |
| **内容长度** | 评论/回复长度 > 20 字 |
| **对话质量** | 有来有回的多轮对话，非广告/水评 |
| **语言质量** | 语句通顺，无明显错别字 |

---

## 四、技术方案

### 4.1 整体架构

```
┌─────────────────────────────────────────────────────────────────┐
│                        数据采集流程                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐  │
│  │ 种子 URL  │───>│ 规则爬虫 │───>│ 质量标注 │───>│ 训练数据 │  │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘  │
│                          │                                      │
│                          v                                      │
│                    ┌──────────┐    ┌──────────┐    ┌──────────┐ │
│                    │ 分类模型 │<───│ 模型训练 │<───│ 数据清洗 │ │
│                    └──────────┘    └──────────┘    └──────────┘ │
│                          │                                      │
│                          v                                      │
│                    ┌──────────────────────────────────────────┐ │
│                    │     智能爬虫：URL 质量预测 → 优先爬取      │ │
│                    └──────────────────────────────────────────┘ │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 阶段划分

#### 阶段一：规则爬虫 + 数据收集

**目标**: 搭建基础爬虫，收集原始数据并自动生成伪标签

**技术选型**:
- **爬虫框架**: Playwright (处理动态加载、反爬)
- **数据存储**: JSON/CSV → 后续可迁移至数据库
- **伪标签规则**: 基于互动数据自动标注

**伪标签生成规则**:

```python
def generate_pseudo_label(post):
    """
    根据规则生成质量标签
    返回：'high' (优质), 'medium' (普通), 'low' (低质)
    """
    score = 0
    
    # 互动量评分
    if post['likes'] > 1000: score += 3
    elif post['likes'] > 100: score += 2
    elif post['likes'] > 10: score += 1
    
    if post['collects'] > 500: score += 2
    elif post['collects'] > 50: score += 1
    
    if post['comments'] > 100: score += 2
    elif post['comments'] > 20: score += 1
    
    # 内容长度评分
    if len(post['content']) > 200: score += 2
    elif len(post['content']) > 50: score += 1
    
    # 互动率评分
    engagement_rate = (post['likes'] + post['collects']) / max(post['views'], 1)
    if engagement_rate > 0.05: score += 2
    elif engagement_rate > 0.01: score += 1
    
    # 标签判定
    if score >= 8: return 'high'
    elif score >= 5: return 'medium'
    else: return 'low'
```

#### 阶段二：质量分类模型训练

**目标**: 训练分类模型，实现内容质量自动预测

**框架选择**: PaddlePaddle + paddleformers (适配 Iluvatar GPU)

**模型选型**:

| 模型 | 参数量 | 显存需求 | 推荐度 |
|------|--------|----------|--------|
| ERNIE-base | 110M | ~4GB | ⭐⭐⭐⭐⭐ |
| Chinese-BERT-wwm | 110M | ~4GB | ⭐⭐⭐⭐ |
| RoBERTa-base | 125M | ~4GB | ⭐⭐⭐⭐ |
| TextCNN | 10M | ~1GB | ⭐⭐⭐ |

**推荐**: `ernie-3.0-base` 或 `bert-base-chinese` (PaddleNLP 内置)

**训练配置**:

```yaml
model:
  name: ernie-3.0-base
  max_length: 512
  
training:
  batch_size: 32
  learning_rate: 2e-5
  epochs: 5
  warmup_ratio: 0.1
  
data:
  train_ratio: 0.8
  val_ratio: 0.1
  test_ratio: 0.1

device:
  name: iluvatar_gpu:0
```

**PaddleNLP 训练示例**:
```python
import paddle
from paddlenlp.transformers import ErnieTokenizer, ErnieForSequenceClassification

paddle.set_device('iluvatar_gpu:0')

model = ErnieForSequenceClassification(num_classes=3)  # high/medium/low
tokenizer = ErnieTokenizer.from_pretrained('ernie-3.0-base')
```

#### 阶段三：智能爬取部署

**目标**: 部署模型，实现 URL 质量预测和优先级调度

**工作流程**:
1. 输入候选 URL 列表
2. 模型预测每个 URL 的内容质量
3. 按预测分数排序，优先爬取高分 URL
4. 爬取后更新模型（在线学习）

### 4.3 目录结构

```
pachong_model/
├── pachong_model.md          # 方案文档
├── requirements.txt          # 依赖列表
├── data/
│   ├── raw/                  # 原始爬取数据
│   ├── processed/            # 处理后数据
│   └── labels/               # 标注数据
├── src/
│   ├── crawler/              # 爬虫模块
│   │   ├── douyin.py
│   │   ├── xiaohongshu.py
│   │   └── base.py
│   ├── model/                # 模型模块
│   │   ├── classifier.py
│   │   └── trainer.py
│   └── utils/                # 工具模块
│       ├── labeler.py        # 伪标签生成
│       └── preprocess.py     # 数据预处理
├── configs/                  # 配置文件
├── checkpoints/              # 模型检查点
└── scripts/                  # 运行脚本
```

---

## 五、实施计划

### 5.1 任务列表

| 序号 | 任务 | 预计工作量 | 优先级 |
|------|------|------------|--------|
| 1 | 环境配置 (PyTorch、CUDA) | 1 天 | P0 |
| 2 | 搭建基础爬虫框架 | 3-5 天 | P0 |
| 3 | 实现抖音/小红书爬虫 | 5-7 天 | P0 |
| 4 | 伪标签生成模块 | 2 天 | P1 |
| 5 | 数据预处理 pipeline | 2 天 | P1 |
| 6 | 分类模型训练 | 3-5 天 | P1 |
| 7 | 模型评估与调优 | 2-3 天 | P2 |
| 8 | 智能爬取部署 | 3-5 天 | P2 |

### 5.2 里程碑

- **M1**: 完成基础爬虫，获取 1000+ 条原始数据
- **M2**: 完成伪标签生成，训练数据集 ready
- **M3**: 模型训练完成，准确率 > 80%
- **M4**: 智能爬虫上线，自动化运行

---

## 六、风险与挑战

| 风险 | 描述 | 应对措施 |
|------|------|----------|
| **反爬机制** | 抖音/小红书反爬严格 | 使用 Playwright 模拟浏览器、设置请求间隔、使用代理池 |
| **登录验证** | 部分数据需登录 | 准备测试账号、考虑 Cookie 池方案 |
| **GPU 资源** | 当前环境无 GPU | 申请 GPU 配额或使用 CPU 训练小模型 |
| **数据合规** | 平台数据使用限制 | 仅用于研究、遵守 robots.txt、控制爬取频率 |
| **数据质量** | 伪标签可能不准确 | 后续人工抽样校验、迭代优化规则 |

---

## 七、下一步行动

1. **环境确认**: ✅ GPU 已就绪 (Iluvatar BI-V150S 32GB)
2. **选择起始平台**: 优先攻克抖音或小红书其中之一
3. **准备测试账号**: 准备用于爬虫的测试账号
4. **搭建爬虫框架**: 使用 Playwright 搭建基础爬虫

---

## 八、参考资源

- PaddlePaddle 文档：https://www.paddlepaddle.org.cn/documentation
- PaddleNLP 文档：https://paddlenlp.readthedocs.io/
- ERNIE 模型：https://paddlenlp.readthedocs.io/zh/latest/model_zoo/ernie.html
- Playwright 文档：https://playwright.dev/python
- Iluvatar 芯石驱动文档：https://www.iluvatar.ai/
