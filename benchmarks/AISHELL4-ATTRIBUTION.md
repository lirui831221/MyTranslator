# AISHELL-4 换人压力测试数据说明

- 数据集：AISHELL-4
- 官方下载：https://openslr.org/111/
- 许可：Creative Commons Attribution-ShareAlike 4.0（CC BY-SA 4.0）
- 用途：MyTranslator 的 VAD、换人断句、说话人回合约束和远场 ASR 回归测试

`prepare_aishell4_speaker_stress.py` 从官方 test 划分选择一个真实会议中的连续说话人
回合，只保留生成测试所需的短音频、原始 TextGrid 副本、时间范围和哈希。报告不会
嵌入或重新分发音频。

引用：

```text
Yihui Fu, Luyao Cheng, Shubo Lv, Yukai Jv, Yuxiang Kong, Zhuo Chen,
Yanxin Hu, Lei Xie, Jian Wu, Hui Bu, Xin Xu, Jun Du, and Jingdong Chen.
AISHELL-4: An Open Source Dataset for Speech Enhancement, Separation,
Recognition and Speaker Diarization in Conference Scenario. Interspeech 2021.
```
