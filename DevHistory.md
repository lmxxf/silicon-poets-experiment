# 开发日志

## 2026-03-14 实验框架搭建

### 项目目标
基于 EmrysRyu《硅基诗学》理论，用 LLM 潜空间实证验证诗歌的语义虫洞（维度折叠）。5 组诗/白话对照，4 个实验。公众号第 121 篇素材。

### ⚠️ DGX Spark 70B 生存指南（血泪教训）

**能跑 70B INT8 的唯一已验证版本组合：**

| 包 | 版本 | 说明 |
|---|------|------|
| transformers | **4.46.0** | 4.57+ 量化系统变了，内存暴涨，70B forward 直接死机 |
| compressed-tensors | **0.8.0** | 0.9+ API 改了（`compress_model`），和新版 transformers 对不上 |
| accelerate | 1.13.0 | |
| 基础镜像 | nvcr.io/nvidia/pytorch:25.11-py3 | 镜像本身没问题，别在里面乱升级包 |

**时间线：为什么之前能跑现在不能跑？**

- **2025-12-01**：transformers v5.0.0 发布，量化系统重写，模型加载方式变了
- **2026-02-12**：Paper 15 用 `magical_bhabha` 容器跑 70B SAE 实验，当时 transformers 还是 4.46 左右，正常
- **某天**：容器里 `pip install` 了什么，transformers 升到 4.57.6，compressed-tensors 升到 0.14.0.1
- **2026-03-14**：70B forward 死机。新建干净容器 `poet_exp`，指定旧版本，恢复正常

**铁律：DGX Spark 上跑 70B，不要升级 transformers 和 compressed-tensors。能跑就不要动。**

### 目录结构

| 宿主机路径 | 容器路径 | 说明 |
|-----------|---------|------|
| `/home/lmxxf/work/poet-traversing/` | `/workspace/poet-traversing/` | 源文件（git 仓库），`poet_exp` 容器可直接访问 |
| `/home/lmxxf/work/ai-theorys-study/arxiv/wechat121/` | `/workspace/ai-theorys-study/arxiv/wechat121/` | **硬链接**到源文件 |

- `poet_exp` 挂载了整个 `~/work`，两个路径都能直接访问
- 硬链接是文件级别的，新建文件需要手动 `ln` 过去
- **注意**：`Write` 工具写文件会创建新 inode，已有硬链接会断！用 `Edit` 工具不会断

### Git 仓库
- `git@github.com:lmxxf/silicon-poets-experiment.git`
- 分支：main

### 容器信息

**当前容器：**
- 名称：`poet_exp`
- 镜像：`nvcr.io/nvidia/pytorch:25.11-py3`
- 挂载：`-v /home/lmxxf/work:/workspace`（整个 work 目录）
- Python 包：`transformers==4.46.0`, `compressed-tensors==0.8.0`, `accelerate==1.13.0`
- 创建命令：`docker run -d --gpus all --name poet_exp -v /home/lmxxf/work:/workspace nvcr.io/nvidia/pytorch:25.11-py3 sleep infinity`
- 装依赖：`pip install transformers==4.46.0 compressed-tensors==0.8.0 accelerate matplotlib seaborn scikit-learn`

**废弃容器：**
- 名称：`magical_bhabha`
- 死因：包被升级，70B INT8 forward OOM 死机

### DGX Spark 已知坑
- **OOM 不报错，直接死机**：128GB 共享内存满了不会抛 CUDA OOM，整机变僵尸，只能拔电
- 参考：https://forums.developer.nvidia.com/t/dgx-spark-becomes-unresponsive-zombie-instead-of-throwing-cuda-oom/353752
- 70B INT8 + `output_hidden_states=True` 全层：**旧版 transformers 4.46.0 能跑，4.57.6 死机**
- 70B + `output_attentions=True` → 必死（任何版本）
- 14B fp16（~28GB）安全，但 d_model=5120 太小，信号分辨率不够

### 脚本文件

| 文件 | 用途 | 模型 |
|------|------|------|
| `exp123_qwen72b.py` | 实验一（残差流轨迹+跳跃距离）+ 实验三（余弦相似度+跨层追踪） | Llama-3.3-70B-INT8 |
| `exp4_sae_llama70b.py` | 实验四（SAE 特征分解，双域共激活检测） | Llama-3.3-70B-INT8 |
| `test_70b_forward.py` | 最小测试：70B 能不能完成一次 forward | Llama-3.3-70B-INT8 |

文件名是历史遗留，实际都用 Llama-70B。

### 实验结果

#### 第一轮：Qwen3-14B（废弃）

14B d_model=5120 太小，跳跃距离比 0.92x~1.22x 没有显著差异，非对角线指标有系统偏差（白话 token 多导致绝对数量碾压）。

#### 第二轮：Llama-70B Layer 50 hook（废弃）

transformers 4.46.0 下 forward hook 输出全是 embedding 层数据，所有跳跃距离比 = 1.00x，最大跳跃全在 `<|begin_of_text|>→第一个字`。hook 注册方式和 transformers 4.46 不兼容。

#### 第三轮：Llama-70B output_hidden_states=True（当前）✅

transformers 4.46.0 + compressed-tensors 0.8.0，`output_hidden_states=True` 全层采样，正常工作。

**实验一：跳跃距离 — 无显著差异**

| 诗组 | 跳跃距离比 | 最大跳跃处 |
|------|-----------|-----------|
| 春风知别苦 | 0.96x | 春→风 |
| 大漠孤烟直 | 1.04x | 乱码（tokenizer） |
| 空山不见人 | 0.99x | 空→山 |
| 君不见高堂 | 0.99x | 君→不 |
| 落霞与孤鹜 | 1.02x | 落→霞 |

结论：**欧氏距离测的是 tokenizer 边界效应，不是语义跳跃。这个指标不对。**

**实验三：余弦相似度 — 有信号！🔥**

| 诗组 | 诗·非对角线高相似度对 | 白话·非对角线高相似度对 |
|------|---------------------|---------------------|
| 春风知别苦 | **0** | 15 |
| 大漠孤烟直 | **0** | 9 |
| 空山不见人 | **0** | 5 |
| 君不见高堂 | **0** | 4 |
| 落霞与孤鹜 | **0** | 32 |

**5 组诗全部 = 0，白话全部 > 0。** 热力图对比极其清晰：
- 诗组矩阵全蓝，token 之间余弦相似度 < 0.8，高度正交
- 白话组矩阵大面积红橙，远距离 token 共享语义空间，高度冗余

**核心发现：虫洞不是"把远距离 token 拉近"，而是"用最少的 token 撑开最大的语义体积"。诗的每个字占据独立的语义维度——这就是《硅基诗学》说的"意象正交基"。**

**EID：有效内在维度 — 诗用更少的字撑开更大的语义体积 🔥**

对最后一层 hidden states 做 SVD，计算谱熵 → EID = exp(H)。归一化 EID = EID / token 数。

| 诗组 | 诗·归一化EID | 白话·归一化EID | 比值 |
|------|-------------|---------------|------|
| 春风知别苦 | 0.643 | 0.205 | **3.13x** |
| 落霞与孤鹜 | 0.496 | 0.177 | **2.81x** |
| 君不见高堂 | 0.509 | 0.214 | **2.38x** |
| 大漠孤烟直 | 0.505 | 0.214 | **2.36x** |
| 空山不见人 | 0.561 | 0.269 | **2.08x** |

**5 组归一化 EID 比值全部 > 2x。与余弦相似度互相印证：token 正交 → 撑开独立维度 → 语义体积 2-3 倍。**

#### 第四轮：实验四 SAE（Goodfire SAE Layer 50）✅

用 SAE 分解桥接 token 的特征激活，检测双域共激活。

| 诗组 | 桥接token | 最强共享对象 | Pearson | 共享特征数 | 解读 |
|------|----------|------------|---------|-----------|------|
| 春风知别苦 | **知** | 别/苦 | 0.58/0.50 | 32/31 | 同时激活自然域和情感域 |
| 大漠孤烟直 | **直** | 烟/长 | 0.56/0.55 | 39/31 | 桥接视觉形态域和空间尺度域 |
| 空山不见人 | **闻** | ，但/人₂/见 | 0.65/0.54/0.48 | 61/59/50 | 同时激活视觉域和听觉域 |
| 君不见高堂 | **暮** | 成/朝/丝 | 0.63/0.65/0.61 | 50/33/41 | 桥接时间起点和终点 |
| 落霞与孤鹜 | **齐** | 飞/鹜₂/一 | 0.69/0.66/0.56 | 70/65/45 | 同时激活运动域和色彩域 |

**关键发现：「空」与「闻」共享特征 = 0——视觉清零和听觉激活在 SAE 特征层完全正交。**

### 实验完成状态

- [x] 实验一：跳跃距离（阴性结果，欧氏距离测的是 tokenizer 边界效应）
- [x] 实验三：余弦相似度（5 组全 0 vs 白话全 > 0）
- [x] EID：有效内在维度（5 组归一化比值全 > 2x）
- [x] 实验四：SAE 双域共激活（5 组桥接 token 均检测到跨域共激活）
- [x] 公众号 121 篇已写完：`/home/lmxxf/work/ai-theorys-study/wechat/121.md`
- [ ] 实验二（attention）：砍掉，70B 上 output_attentions 会死机
