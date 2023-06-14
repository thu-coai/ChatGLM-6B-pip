from fastapi import FastAPI, Request
from transformers import AutoTokenizer, AutoModel
import uvicorn, json, datetime
import torch

DEVICE = "cuda"
DEVICE_ID = "0"
CUDA_DEVICE = f"{DEVICE}:{DEVICE_ID}" if DEVICE_ID else DEVICE

api_model, api_tokenizer = None, None


def torch_gc():
    if torch.cuda.is_available():
        with torch.cuda.device(CUDA_DEVICE):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


app = FastAPI()


@app.post("/")
async def create_item(request: Request):
    global api_model, api_tokenizer
    json_post_raw = await request.json()
    json_post = json.dumps(json_post_raw)
    json_post_list = json.loads(json_post)
    prompt = json_post_list.get('prompt')
    history = json_post_list.get('history')
    max_length = json_post_list.get('max_length')
    top_p = json_post_list.get('top_p')
    temperature = json_post_list.get('temperature')
    response, history = api_model.chat(api_tokenizer,
                                   prompt,
                                   history=history,
                                   max_length=max_length if max_length else 2048,
                                   top_p=top_p if top_p else 0.7,
                                   temperature=temperature if temperature else 0.95)
    now = datetime.datetime.now()
    time = now.strftime("%Y-%m-%d %H:%M:%S")
    answer = {
        "response": response,
        "history": history,
        "status": 200,
        "time": time
    }
    log = "[" + time + "] " + '", prompt:"' + prompt + '", response:"' + repr(response) + '"'
    print(log)
    torch_gc()
    return answer


def launch_server(
        model_name_or_path="THUDM/chatglm-6b",
        trust_remote_code=True,
        model=None,
        tokenizer=None,
        host="0.0.0.0",
        port=8000,
        workers=1,
    ):
    global api_model, api_tokenizer
    if tokenizer is None:
        api_tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code)
    else:
        api_tokenizer = tokenizer
    if model is None:
        api_model = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code).half().cuda()
        api_model = api_model.eval()
    else:
        api_model = model
    uvicorn.run(app, host=host, port=port, workers=workers)


if __name__ == '__main__':
    model_name_or_path = "THUDM/chatglm-6b"
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True).float()
    model = model.eval()
    launch_server(model=model, tokenizer=tokenizer)
