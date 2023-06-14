from transformers import AutoModel, AutoTokenizer
import gradio as gr
import asyncio
import mdtex2html


web_demo_model, web_demo_tokenizer = None, None

"""Override Chatbot.postprocess"""


def postprocess(self, y):
    if y is None:
        return []
    for i, (message, response) in enumerate(y):
        y[i] = (
            None if message is None else mdtex2html.convert((message)),
            None if response is None else mdtex2html.convert(response),
        )
    return y


gr.Chatbot.postprocess = postprocess


def parse_text(text):
    """copy from https://github.com/GaiZhenbiao/ChuanhuChatGPT/"""
    lines = text.split("\n")
    lines = [line for line in lines if line != ""]
    count = 0
    for i, line in enumerate(lines):
        if "```" in line:
            count += 1
            items = line.split('`')
            if count % 2 == 1:
                lines[i] = f'<pre><code class="language-{items[-1]}">'
            else:
                lines[i] = f'<br></code></pre>'
        else:
            if i > 0:
                if count % 2 == 1:
                    line = line.replace("`", "\`")
                    line = line.replace("<", "&lt;")
                    line = line.replace(">", "&gt;")
                    line = line.replace(" ", "&nbsp;")
                    line = line.replace("*", "&ast;")
                    line = line.replace("_", "&lowbar;")
                    line = line.replace("-", "&#45;")
                    line = line.replace(".", "&#46;")
                    line = line.replace("!", "&#33;")
                    line = line.replace("(", "&#40;")
                    line = line.replace(")", "&#41;")
                    line = line.replace("$", "&#36;")
                lines[i] = "<br>"+line
    text = "".join(lines)
    return text


def predict(input, chatbot, max_length, top_p, temperature, history):
    chatbot.append((parse_text(input), ""))
    for response, history in web_demo_model.stream_chat(web_demo_tokenizer, input, history, max_length=max_length, top_p=top_p,
                                               temperature=temperature):
        chatbot[-1] = (parse_text(input), parse_text(response))       

        yield chatbot, history


def reset_user_input():
    return gr.update(value='')


def reset_state():
    return [], []


def launch_web_demo(
        model_name_or_path="THUDM/chatglm-6b",
        trust_remote_code=True,
        model=None,
        tokenizer=None,
    ):
    global web_demo_model, web_demo_tokenizer
    if tokenizer is None:
        web_demo_tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code)
    else:
        web_demo_tokenizer = tokenizer
    if model is None:
        web_demo_model = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code).half().cuda()
        web_demo_model = web_demo_model.eval()
    else:
        web_demo_model = model

    try:
        loop = asyncio.get_event_loop()
    except RuntimeError as e:
        if str(e).startswith('There is no current event loop in thread'):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        else:
            raise

    with gr.Blocks() as demo:
        gr.HTML("""<h1 align="center">ChatGLM</h1>""")

        chatbot = gr.Chatbot()
        with gr.Row():
            with gr.Column(scale=4):
                with gr.Column(scale=12):
                    user_input = gr.Textbox(show_label=False, placeholder="Input...", lines=10).style(
                        container=False)
                with gr.Column(min_width=32, scale=1):
                    submitBtn = gr.Button("Submit", variant="primary")
            with gr.Column(scale=1):
                emptyBtn = gr.Button("Clear History")
                max_length = gr.Slider(0, 4096, value=2048, step=1.0, label="Maximum length", interactive=True)
                top_p = gr.Slider(0, 1, value=0.7, step=0.01, label="Top P", interactive=True)
                temperature = gr.Slider(0, 1, value=0.95, step=0.01, label="Temperature", interactive=True)

        history = gr.State([])

        submitBtn.click(predict, [user_input, chatbot, max_length, top_p, temperature, history], [chatbot, history],
                        show_progress=True)
        submitBtn.click(reset_user_input, [], [user_input])

        emptyBtn.click(reset_state, outputs=[chatbot, history], show_progress=True)

    demo.queue().launch(share=False, inbrowser=True)

if __name__ == '__main__':
    model_name_or_path = "THUDM/chatglm-6b"
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True).float()
    model = model.eval()
    launch_web_demo(model=model, tokenizer=tokenizer)
