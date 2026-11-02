# 实时翻译基准测试

这个目录把“感觉更快/更准”改成可重复的数字。`smoke.jsonl` 中的 10 条合成语音只用于快速回归；正式结论应使用至少 200 条真实人声，并覆盖安静、远场、环境噪声、专名、数字和代码。

项目同时提供 `fleurs200.jsonl` 的可复现生成流程：从 Google FLEURS 官方测试集选取 100 个中英文共享句子 ID，每个 ID 各取一段中文和英文真实人声，共 200 条。相同 ID 的另一语言文本作为翻译参考。FLEURS 使用 CC BY 4.0，来源与许可见 `benchmarks/FLEURS-ATTRIBUTION.md`。

## 快速运行

```bash
cd /path/to/MyTranslator
python3 tools/benchmark_realtime.py --suite benchmarks/suites/smoke.jsonl --runs 3
```

首次准备 200 条真实人声（官方压缩包合计约 0.9 GB，生成后会删除压缩包，只保留选中的 WAV）：

```bash
python3 tools/prepare_fleurs200.py --download --remove-archives
python3 tools/benchmark_realtime.py --suite benchmarks/suites/fleurs200.jsonl --runs 1 --tag fleurs200
```

工具不依赖第三方 Python 包。若 8341/8342 端口上的 MyTranslator 引擎已运行，会直接复用；否则会从项目目录启动相同参数的 `llama-server` 和 `whisper-server`，结束时只关闭自己启动的进程。已构建 `target/debug/myt-zhconv` 或 release 版本时，测试工具会复用应用相同的繁转简规则。

报告写入 `benchmarks/results/`，同时包含 JSON 原始数据和 Markdown 摘要。端到端延迟按“VAD 尾静音预算 + ASR + 翻译”计算，默认尾静音为快速模式的 420 ms。

## 当前真实语音基线（2026-10-22）

`fleurs200` 一次完整运行的结果：中文 CER 8.92%，英文 WER 5.67%，翻译 chrF 46.40%，ASR P50/P95 833/1199ms，翻译 P50/P95 375/703ms，保守按 420ms 尾静音计算的端到端 P50/P95 为 1623/2271ms。中文和英文录音中位时长分别为 11.79s 和 9.96s，明显长于应用快速模式 8s 的最大单段，因此这组端到端延迟是长句压力测试，不等同于短对话体验。

原始报告：`benchmarks/results/benchmark-20261022-232550-fleurs200.json` 与同名 Markdown。该套件没有专门术语断言，因此术语保持率为 `n/a`；术语回归继续由 `smoke.jsonl` 覆盖。

## 数据格式

每行一个 JSON 对象：

```json
{
  "id": "zh-real-001",
  "audio": "benchmarks/audio/real/zh-real-001.wav",
  "source_lang": "zh",
  "direction": "zh2en",
  "categories": ["real", "noise", "names"],
  "transcript": "人工核对的逐字稿",
  "translation_reference": "A human-reviewed reference translation.",
  "target_terms": ["Required Name", ["allowed spelling A", "allowed spelling B"]]
}
```

- 音频必须是 16 kHz、单声道 WAV；如不是，请先转码。
- `transcript` 必须人工复核，不能把任一 ASR 的输出当答案。
- `target_terms` 中字符串表示必须出现；数组表示任一别名出现即可。
- 若套件没有术语断言，术语保持率显示为 `n/a`，不会伪装成 100%。
- 同一个真实录音可制作安静版和加噪版，但报告中应分别标记类别。

## 通过目标

- 中文 CER ≤ 5%
- 英文 WER ≤ 6%
- 端到端 P50 ≤ 1.3 s
- 端到端 P95 ≤ 1.8 s
- 关键术语保持率 ≥ 95%
- 30 分钟纯环境声零有效字幕（需用单独的长时稳定性测试验证）

可用 `--fail-on-target` 让未达标时返回非零退出码，适合之后接入 CI。

## 系统音频全链路回归

`benchmark_realtime.py` 直接调用模型服务，适合快速判断 ASR/翻译模型是否退化；
`benchmark_live_pipeline.py` 则启动一个隐藏、隔离配置的 MyTranslator 进程，使用
`afplay` 将测试 WAV 按真实速度播放，经 ScreenCaptureKit 系统音频入口走完整链路：

```text
系统音频 → VAD/候选断句 → 滚动 ASR → 稳定前缀 → 说话人回合
        → 语义缓冲 → 翻译队列 → 本机 WebSocket 事件 → JSON/HTML 报告
```

运行 CoVoST 2 小型中英套件：

```bash
python3 tools/benchmark_live_pipeline.py \
  --suite benchmarks/suites/covost2-zh-en-small.jsonl \
  --tag covost2-small
```

运行 AISHELL-4 换人压力集：

```bash
python3 tools/benchmark_live_pipeline.py \
  --suite benchmarks/suites/aishell4-speaker-stress.jsonl \
  --tag aishell4-speaker
```

每次结果位于 `benchmarks/results/live/<时间>-<标签>/`：

- `report.json`：时间线、全部 WebSocket 事件、逐组对齐、CER/chrF、延迟和换人结果；
- `report.html`：可直接浏览的汇总与逐句对照；
- `app.pipeline.log`：本次运行对应的应用日志切片；
- `timeline.json`：每段参考音频在回放流中的准确起止时间；
- `app.stdout.log` / `app.stderr.log`：测试进程诊断信息。

全链路工具默认使用独立的 `127.0.0.1:18080` 事件端口，不修改用户保存的输入源，
也不会显示主窗口或悬浮字幕。测试结束时只终止它自己创建的进程组。若报告作为
下一次的对比基准，可传入 `--baseline <旧 report.json>`。

## 准备 CoVoST 2 中文→英文小型集

项目当前已落地 60 条测试样本：360.5 秒语音、51 位匿名说话人，短/中/长句为
27/26/7，音频总计约 11MB。它来自 `fixie-ai/covost2` 的公开 test 预览，套件逐条保存
镜像提交版本、中文原文、英文参考译文、匿名 `client_id` 和哈希。重新生成：

```bash
python3 tools/prepare_covost2_small.py --download-preview --count 60
```

该预览最多提供 100 条。完整 CoVoST 2 音频目前由 Mozilla Data Collective 管理，必须
由用户登录、接受数据集条款并取得访问权限。数据集页面：
`https://mozilladatacollective.com/datasets/cmp5kv8j101hvo007hjamlqet`。

下载 `covost-2-chinese-china-english-*.tar.gz` 后运行：

```bash
python3 tools/prepare_covost2_small.py \
  /path/to/covost-2-chinese-china-english-13b8c8d5.tar.gz \
  --count 100
```

准备器只从 test 划分保留选中的音频，按短/中/长句轮转抽样，并限制单个
说话人的占比。生成 WAV 为 16kHz、单声道、PCM16，同时把中文原文、英文参考译文、
`client_id`、源文件哈希和转换后哈希写入套件。项目已经缓存 Meta 官方中文→英文
翻译标签。`--download-preview` 只使用镜像公开提供的预览资源，不尝试访问或绕过
MDC 受控下载；需要更大样本时应使用上面的已授权官方包导入方式。

MDC 当前页面把这份整理后的数据标为 CC-BY-NC-4.0，并附加禁止重新识别说话人、
克隆声音等限制；该套件仅用于本地非商业回归，不应随应用发布包分发。

## 准备 AISHELL-4 换人压力集

推荐从 AISHELL 官方 Hugging Face 组织镜像下载全部小型 TextGrid，并只下载算法选中的
一场会议。会议音频以 16MB 分块并行、可续传下载，切片成功后自动删除：

```bash
python3 tools/prepare_aishell4_speaker_stress.py \
  --download-selected --download-workers 8 --count 40
```

当前落地套件来自测试会议 `L_R003S01C02`：40 个回合、6 名说话人、32 个必须换人的
边界，合计 93.1 秒语音。生成的 WAV 总计约 2.9MB。若本地已有 OpenSLR 官方
`test.tar.gz`，也可把压缩包路径作为位置参数直接导入；`--download` 则保留完整整包
下载方式。若网络需要代理，请按当前终端环境设置标准 `HTTPS_PROXY`/`HTTP_PROXY`；
脚本不会把代理地址写进项目配置。

准备器扫描官方 TextGrid，说话人回合按时间排序，选择换人最密集的连续窗口；排除高度
重叠片段以便边界可评分，只从一个会议和指定麦克风通道切出 40 个短片。套件保存
`speaker_id`、`must_break_before`、自然停顿时长、原会议时间戳及哈希。报告中：

- `cross_speaker_merge_violations` 必须为 0；
- `detected_speaker_changes / expected_speaker_changes` 衡量换人断句召回；
- 每个翻译批次通过 `fragment_id → speaker_turn_id` 验证，避免仅凭文本猜测是否跨人。

AISHELL-4 为 CC BY-SA 4.0。正式发布测试片段或报告中嵌入音频时必须保留署名并遵循
相同方式共享要求；本项目默认只在本机保存音频，报告不嵌入音频数据。
