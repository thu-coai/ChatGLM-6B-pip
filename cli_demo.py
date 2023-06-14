import os
import platform
import signal
from transformers import AutoTokenizer, AutoModel
import readline

cli_demo_model, cli_demo_tokenizer = None, None

os_name = platform.system()
clear_command = 'cls' if os_name == 'Windows' else 'clear'
stop_stream = False


def build_prompt(history):
    prompt = "欢迎使用 ChatGLM-6B 模型，输入内容即可进行对话，clear 清空对话历史，stop 终止程序"
    for query, response in history:
        prompt += f"\n\n用户：{query}"
        prompt += f"\n\nChatGLM-6B：{response}"
    return prompt


def signal_handler(signal, frame):
    global stop_stream
    stop_stream = True


def launch_cli_demo(
        model_name_or_path="THUDM/chatglm-6b",
        trust_remote_code=True,
        model=None,
        tokenizer=None,
    ):
    global cli_demo_model, cli_demo_tokenizer
    if tokenizer is None:
        cli_demo_tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code)
    else:
        cli_demo_tokenizer = tokenizer
    if model is None:
        cli_demo_model = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code).half().cuda()
        cli_demo_model = cli_demo_model.eval()
    else:
        cli_demo_model = model

    history = []
    global stop_stream
    print("欢迎使用 ChatGLM-6B 模型，输入内容即可进行对话，clear 清空对话历史，stop 终止程序")
    while True:
        query = input("\n用户：")
        if query.strip() == "stop":
            break
        if query.strip() == "clear":
            history = []
            os.system(clear_command)
            print("欢迎使用 ChatGLM-6B 模型，输入内容即可进行对话，clear 清空对话历史，stop 终止程序")
            continue
        count = 0
        for response, history in cli_demo_model.stream_chat(cli_demo_tokenizer, query, history=history):
            if stop_stream:
                stop_stream = False
                break
            else:
                count += 1
                if count % 8 == 0:
                    os.system(clear_command)
                    print(build_prompt(history), flush=True)
                    signal.signal(signal.SIGINT, signal_handler)
        os.system(clear_command)
        print(build_prompt(history), flush=True)


if __name__ == "__main__":
    model_name_or_path = "THUDM/chatglm-6b"
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True).float()
    model = model.eval()
    launch_cli_demo(model=model, tokenizer=tokenizer)
