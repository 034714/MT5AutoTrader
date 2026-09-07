# MT5AutoTrader

MT5AutoTrader 是一套独立的 MT5 策略训练、回测和自动交易工具。

> **由 AI 编写**：全部代码由 AI（ZCode）在用户的自然语言指导下编写与调试，人类负责提出需求、验收和决策。我们鼓励你也用 AI 来改进和修复它——给任何 AI 助手（Claude / ChatGPT / ZCode / Cursor 等）附上 [AGENTS.md](AGENTS.md)（AI 交接规则文档）和本 README，就能快速上手改代码；欢迎提交 PR，并请注明 AI 参与情况。
> 
 交流 QQ 群：133822181    感兴趣的求Star(ฅ⁍̴̀◊⁍̴́)و ̑̑

## 功能总览

- 从 MT5 直接取 K 线数据并训练（周期可选 M5~D1，内部自动保存数据文件）
- 本地策略训练（CPU），支持从检查点续训
- 离线策略回测和报告
- 手动导入/绑定策略，不会自动接管训练中的中间结果
- 新收盘 K 线触发信号对账：同方向信号去重，反向信号先平后开
- 每 5~10 秒实时价格监控：初始止损 + 阶梯保本止损（只收紧不放松）
- 支撑/阻力位（S/R）优化开仓止盈止损：基于 V3 融合算法（成交量分布 + 极值结构 + ATR 归一化）自动识别关键位
- 关键位前"止盈一半"：到价自动平掉部分仓位锁利（可由概率模型把关），网页上可拖动/输入精确设置
- 网页交互式 K 线图：关键位、持仓线、历史成交标记、拖动设置止损/止盈
- 网页持仓面板：实时仓位同步，支持手动平仓/改止损/改止盈
- 正确的历史成交记录（按持仓单合并，含手续费与库存费）
- 默认 dry-run 演练模式，切换真实下单需要网页二次确认

## 开箱即用（Windows 便携版）

到 [Releases](https://github.com/034714/MT5AutoTrader/releases) 下载 `MT5AutoTrader-x.x.x-windows-x64.zip`，解压到任意目录（**路径不要带中文**），然后：

1. 双击 `install.bat`（仅首次需要，联网安装 Python 依赖；若包内已带预装依赖则自动跳过）
2. 打开 MT5 终端并登录你的账户
3. 双击 `start.bat`，浏览器自动打开 <http://127.0.0.1:8900>
4. 默认是 **dry-run 演练模式**（不发真实订单），先在网页上训练/绑定策略、启动 Runner、观察信号
5. 确认无误后再在"交易控制"页切换真实下单（需要二次确认，且要点亮 MT5 的"算法交易"按钮）

停止：双击 `stop.bat`。详细教程（训练、续训、风控规则、故障排查、换账户说明）见 [说明.txt](说明.txt)。

### 从源码运行（开发者）

```text
git clone https://github.com/034714/MT5AutoTrader.git
cd MT5AutoTrader
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
start.bat   （会优先使用 .venv）
```

- 仅 Windows 可用（MetaTrader5 包依赖 Windows 的 MT5 终端）
- MT5 终端需手动打开并登录，本软件不会自动拉起它
- Python 查找顺序：项目 `.venv` → `runtime\python.exe`（便携包内置）→ `fallback_python.txt` → 系统 PATH

## 快速上手（网页流程）

1. 训练页选"从 MT5 获取并训练"（或选已有数据文件）
2. 策略库导入/绑定 MT5 品种（训练完成的策略自动出现在策略库）
3. 交易控制页启动 Runner（默认 dry-run），在总览看信号和持仓
4. 风控参数页设置止损、阶梯保本、支撑/阻力优化
5. 总览页的 K 线图可直接拖动设置止损/止盈/到价分批平仓

## 目录

- `src/app.py`：FastAPI 看板
- `src/trading/`：MT5 客户端、信号引擎、风控、Runner、支撑/阻力位引擎（`srlab/`）
- `src/model_core/`：策略训练/公式执行核心
- `src/data_pipeline/`：Parquet 和 MT5 数据管线
- `strategies/`：本软件策略库（运行时生成）
- `trader_config.json`：交易、绑定和风控配置（运行时生成，模板见 `trader_config.example.json`）
- `logs/`：看板、Runner、训练和回测日志
- `说明.txt`：中文详细使用教程
- `AGENTS.md`：给 AI 助手的交接规则（已知 Windows 坑都记在里面）

## AI驱动

1. 把本仓库 clone 给你的 AI 助手，并让它先读 [AGENTS.md](AGENTS.md)（记录了架构约定和已踩过的坑）
2. 明确描述需求或 bug 现象（贴上 `logs/` 里的相关日志更好）
3. 修改后按需跑测试（全部 mock，不碰真实 MT5；默认只打印失败项）：
   `python src\tests\test_trading.py risk`（风控）/ `sr`（支撑阻力）/ `runner`（交易流程）/ `signal`（信号）/ `quick`（秒级冒烟）；不带参数 = 全量（发版前跑）
   
## 许可证与致谢

本项目整体基于 [GNU AGPL-3.0](LICENSE) 发布。包含的第三方代码：

- [AlphaMaster](https://github.com/rosemarycox5334-debug/AlphaMaster)（AGPL-3.0）——策略训练引擎（`src/model_core/`、信号/特征计算链路）源自该项目，策略 JSON 格式与其兼容
- [Detect_support_and_resistance_levels](https://github.com/rosemarycox5334-debug/Detect_support_and_resistance_levels)（GPL-3.0）——支撑/阻力位 V3 融合算法，内嵌于 `src/trading/srlab/`（含概率模型 `models/*.json`）

  
## 打赏与支持
如果你觉得这个程序对你有帮助的话，可以打赏激励作者继续优化程序，感谢你的支持和鼓励！
<img width="1279" height="1743" alt="微信图片_20260905190039_1_4" src="https://github.com/user-attachments/assets/aa16aadc-1322-4e62-b884-bbb3ed6f152c" />
<img width="1350" height="2025" alt="微信图片_20260905190040_2_4" src="https://github.com/user-attachments/assets/8e90a3bc-dc5f-424a-9524-a2121a57349a" />

## 免责声明
本工具仅用于技术研究，不构成任何投资建议。市场有风险，交易需谨慎。
