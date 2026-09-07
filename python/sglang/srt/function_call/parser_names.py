"""Tool-call parser names, kept in a dependency-free module.

`server_args` needs these for the CLI `--tool-call-parser` choices; importing
`function_call_parser` for them costs ~3 s (OpenAI protocol models -> xgrammar
-> transformers -> torch.distributed.fsdp). `function_call_parser` asserts at
import that this list matches `FunctionCallParser.ToolCallParserEnum`.
"""

TOOL_CALL_PARSER_NAMES = [
    "apertus2509",
    "cohere_command4",
    "deepseekv3",
    "deepseekv31",
    "deepseekv32",
    "deepseekv4",
    "dots",
    "glm",
    "glm45",
    "glm47",
    "gpt-oss",
    "kimi_k2",
    "kimi_k3",
    "lfm2",
    "ling3",
    "llama3",
    "mimo",
    "minicpm5",
    "mistral",
    "muse",
    "poolside_v1",
    "pythonic",
    "qwen",
    "qwen25",
    "qwen3_coder",
    "spark25",
    "step3",
    "step3p5",
    "minimax-m2",
    "minimax-m3",
    "trinity",
    "interns1",
    "hermes",
    "hunyuan",
    "gigachat3",
    "gemma4",
    "inkling",
]
