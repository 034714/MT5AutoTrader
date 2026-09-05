# -*- coding: utf-8 -*-
"""
srlab — 支撑/阻力位检测（V3 融合算法）

内嵌自 https://github.com/rosemarycox5334-debug/Detect_support_and_resistance_levels
（保留 base/profile/pivots 逐字不动，detectors/metrics/probability 按本项目需要
精简为纯 numpy 实现，去掉了 scipy 与 pandas 依赖）。

算法概要（V3，见 detectors.py 内的实测注释）：
  - 成交量分布（体积守恒 + 时间衰减）
  - 收盘价堆积
  - 因果 ZigZag 极值结构（枢轴聚集）
  - 历史触及统计（事件去重 + Wilson 下界）
  - 排序分 edge 只用实测有效的两个因子（触及次数 + 陈旧度）；
    p_stall（价格停滞倾向）单独输出，用于止损/止盈摆放
数据文件 models/prob_models.json 为可选的概率模型（触及/守住概率），
缺失时概率字段为 null，关键位检测不受影响。
"""
