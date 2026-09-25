import sys, os, json
sys.path.insert(0, 'src')
import gpu; gpu.set_visible_gpu(min_free_mb=18000, wait=True)
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import registers as R, data as D
name="openai/gpt-oss-20b"
tok=AutoTokenizer.from_pretrained(name)
model=AutoModelForCausalLM.from_pretrained(name, torch_dtype="auto", device_map={"":0})
print("LOADED", name, "dtype ok", flush=True)
items=[it for it in D.load_many(["mmlu","arc"], n=4, seed=0) if it["kind"]=="mcq"]
for reg in ["neutral","profane"]:
    for it in items[:3]:
        msg=[{"role":"user","content":R.render(reg, it["content"],0)}]
        text=tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        enc=tok(text, return_tensors="pt", add_special_tokens=False).to(model.device)
        out=model.generate(**enc, max_new_tokens=512, do_sample=False)
        dec=tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        c,pred=D.score_mcq(dec, it["gold"], it["choices"])
        print(f"[{reg}] gold={it['gold']} pred={pred} correct={c} | out_tail={dec[-90:]!r}", flush=True)
print("SMOKE_OK", flush=True)
