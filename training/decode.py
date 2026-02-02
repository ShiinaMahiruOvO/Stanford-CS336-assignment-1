import torch
from cs336_basics.tokenizer import BPETokenizer
from cs336_basics.layers import softmax

@torch.no_grad()
def decode(model: torch.nn.Module,
           tokenizer: BPETokenizer,
           prompt: str,
           max_new_tokens=100,
           temperature=1.0,
           top_p=0.9,
           device=None):
    model.eval()
    tokens = torch.tensor(
        prompt, dtype=torch.long, device=device
    ).unsqueeze(0)
    
    for _ in range(max_new_tokens):
        logits = model(tokens)
        last_logits = logits[:, -1, :]
        
        if temperature <= 0:
            next_token = last_logits.argmax(dim=-1, keepdim=True)
        else:
            scaled_logits = last_logits / temperature
            probs = softmax(scaled_logits)
            
            if top_p is not None and 0.0 < top_p < 1.0:
                sorted_tokens, sorted_idx = torch.sort(probs, descending=True)
                cumsum = torch.cumsum(sorted_tokens, dim=-1)
                keep = cumsum <= top_p
                keep[..., 0] = True
                nucleus_p = torch.where(keep, sorted_tokens, 0.0)
                nucleus_p /= nucleus_p.sum(dim=-1, keepdim=True)
                sample = torch.multinomial(nucleus_p, num_samples=1)
                next_token = sorted_idx.gather(dim=-1, index=sample)
            else:
                next_token = torch.multinomial(probs, num_samples=1)
        
        tokens = torch.cat((tokens, next_token), dim=-1)
        
    out_ids = tokens[0].tolist()
    out_txt = tokenizer.decode(out_ids)
    return out_txt
