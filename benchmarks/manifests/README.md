# 实时测试数据清单

此目录只保存能够公开保留的来源信息和被选中会议的原始标注副本；受访问控制或体积
较大的原始音频放在 `benchmarks/cache/`，生成的少量测试 WAV 放在
`benchmarks/audio/`。这些目录中的音频不得打入 MyTranslator 安装包。

- CoVoST 2 zh-CN→en：MDC 数据集 `cmp5kv8j101hvo007hjamlqet`，当前整理版本
  CC-BY-NC-4.0，并受页面列出的附加限制约束。
- AISHELL-4：OpenSLR SLR111，CC BY-SA 4.0。

每个 JSONL 样本都应记录源数据集、split、原始路径/时间范围及 SHA-256；任何缺少
人工参考文本或来源信息的网络音频都不能进入正式回归基准。
