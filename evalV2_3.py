import json
import torch
import tiktoken
from modelV2_1 import GPT, GPTConfig

# -----------------------
# CONFIG (same style as sample.py)
# -----------------------
init_from = 'resume'   # or 'resume'
out_dir = 'out'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# -----------------------
# load eval data
# -----------------------
def load_eval_data(path="eval_dataV2_2.json"):
    with open(path, "r") as f:
        return json.load(f)

# -----------------------
# eval function
# -----------------------
def evaluate(model, encode, decode):

    data = load_eval_data()

    total_prob = 0.0

    print("\n=== EVALUATION START ===\n")

    for i, item in enumerate(data):

        prompt = item["prompt"]
        response = item["response"]

        # encode prompt + response
        x = torch.tensor([encode(prompt)], dtype=torch.long, device=device)
        fixed_response = encode(response)

        # run model
        y, prob, log_prob = model.generate(
            x,
            max_new_tokens=len(fixed_response),
            temperature=1.0,
            top_k=200,
            fixed_response=fixed_response
        )

        # decode output
        generated_text = decode(y[0].tolist())

        print(f"Example {i+1}")
        print("Prompt:   ", prompt)
        print("Target:   ", response)
        print("Generated:", generated_text)
        print("Prob:     ", prob)
        print("Log Prob: ", log_prob)
        print("---------------")
        total_prob += prob
    avg_prob = total_prob / len(data)
    print("Average probability:", avg_prob)


# -----------------------
# main
# -----------------------
if __name__ == "__main__":

    # tokenizer
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s)
    decode = lambda l: enc.decode(l)

    # load model
    if init_from == 'gpt2':
        model = GPT.from_pretrained('gpt2')
    else:
        checkpoint = torch.load(f"{out_dir}/ckpt.pt", map_location=device)
        config = GPTConfig(**checkpoint["model_args"])
        model = GPT(config)
        model.load_state_dict(checkpoint["model"])

    model.to(device)
    model.eval()

    evaluate(model, encode, decode)