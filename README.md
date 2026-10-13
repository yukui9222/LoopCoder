# LoopCoder

基于 Python 和 Textual 的本地智能编程助手，支持代码检索与编辑、命令执行、测试验证和多轮任务处理。

## 主要功能

- **Agent Loop**：模型决策、工具执行与结果回写，支持 OpenAI 兼容接口和 Anthropic 协议。
- **工具与交互**：文件检索、代码编辑、Shell 和 MCP 工具，提供流式输出与工具执行轨迹。
- **上下文与会话**：JSONL 事件日志、会话恢复与分叉、L1–L4 分层上下文压缩及归档取回。
- **权限与审批**：allow/ask/deny 策略、持久授权、写入前 diff 预览及过期预览检测。

## 快速开始

需要 Python 3.11 或更新版本。

```sh
git clone https://github.com/yukui9222/LoopCoder.git
cd LoopCoder
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

在待开发项目的根目录创建 `loopcoder.toml`，将以下占位值替换为实际的模型 ID 和服务地址：

```toml
default_model = "custom/model-id"

[providers.custom]
type = "openai-compatible"
base_url = "https://your-provider.example/v1"
api_key_env = "LOOPCODER_API_KEY"

[models."custom/model-id"]
label = "My model"
```

设置密钥后，在该项目目录启动终端界面：

```sh
export LOOPCODER_API_KEY="your-api-key"
loopcoder --tui
```

也可以直接执行一轮任务：

```sh
loopcoder --message "检查项目结构，并修复一个问题后运行相关测试"
```

配置密钥与会话数据保存在本地，请勿提交到 Git。

## 许可

基于 [FirstCoder](https://github.com/KomorGiaoGiao/FirstCoder) 源码整理，遵循 MIT License，保留原作者版权声明，详见 [LICENSE](LICENSE)。
