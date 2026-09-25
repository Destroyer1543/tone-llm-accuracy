import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class ModelRunner:
    def __init__(self, model_name, dtype=torch.bfloat16, max_new_tokens=256):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map={"": 0},
        )
        self.model.eval()

    def build_prompts(self, user_texts):
        prompts = []
        for t in user_texts:
            msgs = [{"role": "user", "content": t}]
            prompts.append(
                self.tokenizer.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True
                )
            )
        return prompts

    @torch.no_grad()
    def generate(self, user_texts, batch_size=16, max_new_tokens=None, do_sample=False,
                 temperature=1.0):
        mnt = max_new_tokens or self.max_new_tokens
        prompts = self.build_prompts(user_texts)
        outputs = []
        import time as _t
        n_batches = (len(prompts) + batch_size - 1) // batch_size
        _t0 = _t.time()
        for bi, start in enumerate(range(0, len(prompts), batch_size)):
            if bi % 20 == 0 and bi > 0:
                done = start
                rate = done / max(_t.time() - _t0, 1e-6)
                eta = (len(prompts) - done) / max(rate, 1e-6)
                print(f"[gen] batch {bi}/{n_batches} ({done}/{len(prompts)} prompts) "
                      f"{rate:.1f} prompt/s ETA {eta:.0f}s", flush=True)
            batch = prompts[start:start + batch_size]
            enc = self.tokenizer(
                batch, return_tensors="pt", padding=True, add_special_tokens=False
            ).to(self.model.device)
            gen = self.model.generate(
                **enc,
                max_new_tokens=mnt,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                top_p=None,
                pad_token_id=self.tokenizer.pad_token_id,
            )

            new_tokens = gen[:, enc["input_ids"].shape[1]:]
            decoded = self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
            outputs.extend(decoded)
        return outputs
