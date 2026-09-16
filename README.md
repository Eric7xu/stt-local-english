# stt-local-english

本地英文语音转文本 CLI（macOS Apple Silicon，基于 `mlx-whisper` + Whisper `large-v3-turbo`）。

把任意视频/音频文件转成带时间戳的 **SRT / VTT / Markdown / JSON / TXT**。
专为「批量转字幕 → LLM 精修」的流程设计：词级时间轴、断点续转、JSON 结构化输出。

---

## 特性一览

| 特性 | 说明 |
|---|---|
| 🍎 GPU 原生 | 跑满 M 系列芯片统一内存（MLX），实测 M5 约 **11–13× 实时** |
| 📄 5 种输出 | `srt` / `vtt` / `md` / `txt` / `json`，可同时多选 |
| ⏱ 词级时间轴 | 字幕条起止 = 第一个词开口 → 末词收尾（`--word-timestamps`） |
| 🛡 去伪影回退 | 词对齐不可靠的片段自动回退为句级等比切分，杜绝字幕词重复 |
| ♻️ 断点续转 | 已生成全部目标格式的文件自动跳过，中断重跑即可续转 |
| 📦 模型缓存 | 权重磁盘级缓存 + 进程内只加载一次 |
| 🌲 目录批量 | 支持递归扫描子目录，输出镜像目录结构 |

---

## 安装

### ⚠️ 平台要求（先读）

| 项 | 要求 | 说明 |
|---|---|---|
| 操作系统 | **transcribe：mac 用 mlx 引擎；Linux/Win 自动用 faster-whisper 引擎** | Linux/Win 上 CPU 可跑（NVIDIA 更快）；`
      --vad` 可治音乐段幻觉 |
| `polish` / `judge` | ✅ 全平台 | 纯 Python 标准库 |
| ffmpeg | 仅 transcribe 需要 | `brew install ffmpeg` / `apt install ffmpeg` |
| 包管理 | uv | `brew install uv` 或见 [uv 文档](https://docs.astral.sh/uv/) |
| 网络 | 首次需联网 | 下载依赖 + 模型权重（后续离线可用） |

### 5 步从零开始（全新机器）

```bash
# 1. 装前置（已装可跳过）
brew install uv ffmpeg

# 2. 克隆（私有仓库需先被授予访问权限）
git clone git@github.com:Eric7xu/stt-local-english.git
cd stt-local-english

# 3. 创建环境并安装依赖（uv 会自动装 Python 3.12）
uv sync

# 4. 验证
uv run stt-local --help

# 5. 转第一个文件（首次会自动下载模型，见下节）
uv run stt-local your_video.mp4 -o ./out -f srt -f md
```

**全局安装**（任意目录直接用 `stt-local`）：

```bash
uv tool install .
# 升级 / 卸载
uv tool upgrade stt-local-english
uv tool uninstall stt-local-english
```

---

## 模型怎么来？（不入库，按需下载）

模型权重**不放进 git 仓库**（默认版约 1.5 GB，避免仓库臃肿），改为：

1. 首次执行转写时，`mlx-whisper` 自动从 HuggingFace 拉取 `mlx-community/whisper-large-v3-turbo`
2. 存入本机缓存 `~/.cache/huggingface/hub/models--mlx-community--whisper-large-v3-turbo/`
3. **只下载一次**，之后完全离线可用
4. 删除缓存即重新下载；换机器 = 各自下载一次

```bash
# 想看缓存大小 / 手动清理
ls ~/.cache/huggingface/hub/ | grep whisper
rm -rf ~/.cache/huggingface/hub/models--mlx-community--whisper-large-v3-turbo
```

**嫌大 / 网慢？** 换 int4 量化版（约 500 MB，效果略降）：

```bash
stt-local clip.mp4 -m mlx-community/whisper-large-v3-turbo-int4
```

**中国大陆网络下不去 HuggingFace？** 用镜像（一行环境变量，无需代理）：

```bash
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_ENABLE_HF_TRANSFER=1   # 可选：加速下载
stt-local clip.mp4
```

---

## 快速上手

```bash
# ① 单个视频 → 在视频同目录生成 .srt + .md
stt-local 001_video.mp4

# ② 只要字幕
stt-local 001_video.mp4 --no-md

# ③ 整个目录批量（含子目录），输出镜像到 transcripts/
stt-local ./videos/ -o ./transcripts/

# ④ 正式转写推荐（词级时间轴 + 三件套，供后续 LLM 精修）
stt-local ./videos/ -o ./transcripts/videos/ \
          -f srt -f md -f json --word-timestamps

# ⑤ 先看看会转哪些文件
stt-local ./videos/ --dry-run
```

---

## 命令行参考

```
usage: stt-local [-h] [-o OUTPUT_DIR] [-m MODEL] [-f {json,md,srt,txt,vtt}]
                 [-l LANGUAGE] [--word-timestamps] [--no-md] [--force]
                 [--no-recursive] [--dry-run] [--verbose] [--version]
                 inputs [inputs ...]
```

| 参数 | 默认 | 说明 |
|---|---|---|
| `inputs` | — | 媒体文件或目录（可多个；目录默认递归） |
| `-o, --output-dir` | 无 | 输出目录，镜像输入结构；**不填则写在每个媒体文件同目录** |
| `-m, --model` | `mlx-community/whisper-large-v3-turbo` | 见下方模型表（faster 引擎自动映射为对应尺寸名） |
| `-f, --format` | `srt md` | 输出格式，可重复传（如 `-f srt -f json`） |
| `-l, --language` | `en` | 语言提示；`auto` 自动检测 |
| `--engine` | `auto` | `mlx`（Apple Silicon GPU）/ `faster`（跨平台，Linux/Win 默认）/ `auto` 自动选 |
| `--fw-device` | `auto` | faster 引擎设备：`cpu` / `cuda` / `auto` |
| `--vad` | 关 | faster 引擎：VAD 过滤非语音段（音乐/片尾幻觉的 Linux 解法） |
| `--word-timestamps` | 关 | 逐词时间戳：SRT 词级切轴、JSON 含 `words` |
| `--no-condition-previous` | 开 | 关闭“以上文为条件”解码；片尾音乐/静音段触发循环重复幻觉（`A-A-A-...`）时用它能根治 |
| `--no-md` | 关 | 快捷方式：只输出 SRT |
| `--force` | 关 | 已有结果也重新转写 |
| `--no-recursive` | 关 | 不递归子目录 |
| `--dry-run` | 关 | 只列出将处理的文件并退出 |
| `--verbose` | 关 | 详细日志（含进度条） |
| `--version` | — | 打印版本 |

### 模型选择

默认 `large-v3-turbo` 是准确率/速度的最佳平衡。其它可用 HF 仓库（MLX 转换版）：

| 模型 | 参数 | 相对速度 | 相对准确率 | 说明 |
|---|---|---|---|---|
| `mlx-community/whisper-turbo` | ~809M | ★★★★★ | ★★★★ | 快，够用 |
| `mlx-community/whisper-large-v3-turbo` | 809M | ★★★★ | ★★★★★ | **默认**，强烈推荐 |
| `mlx-community/whisper-large-v3-turbo-int4` | 809M 量化 | ★★★★★ | ★★★★☆ | 省内存版，效果略降 |
| `mlx-community/whisper-large-v3` | 1.54B | ★★ | ★★★★★ | 顶配准确率，慢 ~4× |
| `mlx-community/whisper-small` | 244M | ★★★★★★ | ★★★ | 快速试跑/验证管道 |

小模型快速验证管线：`stt-local clip.mp4 -m mlx-community/whisper-small -f txt`

---

## 输出格式详解

### SRT（字幕）
- 默认每句一条字幕；`--word-timestamps` 时每条字幕的起止时间取该条**首词→末词**的真实对齐时间
- 长句自动在词边界拆行（≤80 字符），拆出的每行各有自己的精确时间轴
- 若某片段词级对齐被判定不可靠（词重复/错位伪影），该片段自动回退为**句级等比切分**，保证不输出重复内容

### MD（正文阅读版）
段落按说话停顿（>1s 间隙）聚合，适合直接阅读/喂 LLM。

### JSON（结构化，LLM 精修首选）
```jsonc
{
  "source": "/path/001_video.mp4",   // 来源文件
  "language": "en",                  // 检测/指定语言
  "duration": 180.0,                 // 音频时长（秒）
  "elapsed": 12.3,                   // 本次转写耗时（秒）
  "segments": [
    {
      "start": 0.0,                  // 句开始（秒）
      "end": 5.9,                    // 句结束（秒）
      "text": "Welcome back to the series.",
      "words": [                     // --word-timestamps 开启时才有
        {"word": "Welcome", "start": 0.0, "end": 0.5},
        {"word": " back",   "start": 0.5, "end": 0.9}
        // ...逐词时间
      ]
    }
  ]
}
```

---

## 断点续转

判定规则：**某媒体文件的全部目标格式输出文件都已存在且非空** → 跳过。
- 批量中断 → 重跑同一条命令，完成的瞬间 `SKIP`，未完成的继续
- `--force` 强制全部重转
- 原子写入：输出先写 `.tmp` 再改名，不会留下半个文件

---

## 实测参考（开发机 Apple M5 / 24GB）

- 57 秒片段：**4.4 s** 完成（≈13× 实时）
- 全量 176 集（约 19.4 小时音频，含词级时间戳 + 3 种格式输出）：**约 1h41m**（≈11.5× 实时），0 失败
- 输出体积约 160 KB/集（srt+md+json 三件套约 160 KB）——一整套 176 集约 28 MB

---

## LLM key 放哪（不用 zshrc、不用钥匙串）

存在**专用轻量配置文件**（类似一个只干一件事的迷你 zshrc，但只被 stt-local 读取，
不进任何 shell、不进任何仓库）：

```
~/.config/stt-local/keys.env        # 权限 600，仅本人可读
```

```bash
# 填写（去掉行首 # 即生效）
# STT_LLM_API_KEY=sk-xxx
# STT_LLM_BASE_URL=https://api.deepseek.com/v1
# STT_LLM_MODEL=deepseek-chat

chmod 600 ~/.config/stt-local/keys.env   # 确认权限
```

- 读取优先级：CLI 参数 > 环境变量 > keys.env
- 修改即生效，无需 source / 重启
- 想临时在 shell 里用同一套变量：`set -a; source ~/.config/stt-local/keys.env; set +a`
- **迁移到新机器**：随项目拷走这一个文件即可（同 gh 的 ~/.config/gh 模式）

## LLM 精修（stt-local polish）

对已转写的 `.json` 做**对齐安全**的错字修复：LLM 只返回逐句修正文本（id 一一对应），
时间轴永远由本工具保管，字幕不会因精修而错位。支持任意 OpenAI 兼容端点。

```bash
# 配置（写一次 ~/.config/stt-local/keys.env，见上一节；也可用环境变量临时覆盖）
# STT_LLM_API_KEY / STT_LLM_BASE_URL / STT_LLM_MODEL

# 先 mock 自测管线（不联网、不花钱）
stt-local polish transcripts/videos/ --mock -o /tmp/pol-test

# 正式精修（推荐配词表，保护专有名词不被“改错”）
stt-local polish transcripts/videos/ -o transcripts/videos-polished \
    --glossary terms.txt --jobs 4
```

要点：
- 输出三件套到 `-o` 目录：`.json`（含 `polished: true` 与用量统计）、`.srt`、`.md`
- 断点续转：每批完成即写 checkpoint，中断重跑不重复计费
- 约束校验：句数/顺序/长度比例不合格的批次自动重试，最终回退原文（日志可见 fallback）
- 幻觉治理：LLM 判定为非语音垃圾的片段输出空串，srt/md 中跳过
- 术语词表 `terms.txt` 每行一个词（如 `CLAUDE.md`、`agent skills`、``）
- 返回格式已用 few-shot 示例 + 显式 id 规则约束（不得重新编号/遗漏 id），减少无效载荷重试

## 二遍裁判（stt-local judge）

用**另一个模型家族**对精修结果逐条独立评审（避免自评偏见），产出 OK/SUSPECT 报告：

```bash
stt-local judge transcripts/polished/ \
    --orig-root transcripts/ \
    --glossary terms.txt \
    --llm-model openai/gpt-4o-mini \
    -o judge-report.md
```

要点：
- 原始转录目录默认自动探测（精修路径中名为 `polished` 的祖先目录被剥离）；也可用 `--orig-root` 显式指定
- 裁判 prompt 注入同一份 glossary —— **不带词表的裁判会把正确的产品名修复误报成 SUSPECT**（实测教训）
- 严格校验：返回 id 集合必须与输入完全相等（防重新编号），int 键全链路归一化
- 断点续转 + `--mock` 无网络自测
- 实测校准：术语修复/垃圾删除 → OK；真语音被替换/破坏 → SUSPECT（fixture 三方向验证通过）

---

## 开发 / CI

仓库自带 GitHub Actions 冒烟测试（`.github/workflows/smoke.yml`）：在 macOS arm64
runner 上 `uv sync` → 用 whisper-tiny 转写 `tests/sample.mp3` → 校验 srt/md/json 输出。

- 任何 push 到 `main` 或新 PR 都会自动跑
- **`main` 分支受保护：`transcribe-smoke` 必须通过才能合并 PR**
- 本地快速预演 CI：
  ```bash
  uv sync --frozen
  uv run stt-local tests/sample.mp3 -o tests/out -m mlx-community/whisper-tiny \
    -f srt -f md -f json --word-timestamps
  uv run python tests/smoke_check.py tests/out
  ```

---

## 常见问题（FAQ）

**Q：报 `mlx-whisper is not available`？**
不是 Apple Silicon 或依赖没装好。确认芯片 `sysctl -n machdep.cpu.brand_string` 含 Apple，并 `uv sync` 重装。

**Q：我是 Windows / Linux / Intel Mac，能用吗？**
现在**都能转写了**：非 macOS 平台 `uv sync` 会自动装 faster-whisper 引擎（CPU 即可，有 NVIDIA 更快），命令完全一致；Mac 上则默认用 mlx 引擎（可用 `--engine faster` 切换）。`polish` / `judge` 本就全平台。

**Q：报 ffmpeg 相关错误？**
mlx-whisper 内部调用系统 ffmpeg 解码。macOS 装：`brew install ffmpeg`。

**Q：中文/其它语言视频被转成乱码英文？**
本工具面向英文。中文请用 FunASR/SenseVoice 系模型；或 `-l zh`（Whisper 也能转中文，但准确率不如专用模型）。

**Q：某个文件一直 FAIL？**
看具体报错。内存不足（OOM）就换 `-int4` 量化模型；单个文件反复失败会跳过继续跑剩余，跑完看日志 `FAIL` 行汇总即可。

**Q：字幕里有莫名其妙的重复词？**
词级对齐伪影。本工具已内置检测回退；若仍出现，说明该段词时间戳完全不可信，属模型偶发行为，重转 `--force` 单个文件通常可解。

**Q：可以边用边看进度吗？**
默认每完成一个文件打印一行 `[i/n] OK ...`；加 `--verbose` 显示逐段进度条。批量建议 `2>&1 | tee transcribe.log`。

**Q：输出默认写在哪？**
不加 `-o` 时**写在媒体文件旁边**（同名不同扩展名），避免污染原目录请用 `-o`。

---

## 项目结构

```
stt-local-english/
├── pyproject.toml            # 依赖与 CLI 入口 (stt-local)
├── README.md
└── src/stt_local_english/
    ├── __init__.py           # 版本与常量（模型默认值、支持格式）
    ├── cli.py                # argparse 命令行入口
    ├── engine.py             # mlx-whisper 封装：转录 + 段清洗
    ├── batch.py              # 文件发现、批量循环、断点续转判定
    └── output.py             # srt/vtt/md/txt/json 渲染 + 词级/回退算法
```

---

## 后续 LLM 精修建议流程

1. 读每集 `.json` 的 `segments[].text` 让 LLM 改正错词/补标点/统一术语
2. 输出与 `segments` 一一对应的修正文本数组（顺序不可变）
3. 回填到 SRT：时间轴（`start/end` 或 `words`）原样保留，只替换文本
4. 校验：新 SRT 与旧 SRT 的 cue 数量一致、时间单调递增

> `.json` 里的 `words` 数组用于更精细的字幕回填；如只做段落级润色，忽略 `words`、
> 逐句对齐 `segments` 即可。
