from langchain.text_splitter import TextSplitter
import torch, torch.nn.functional as F

class ModelBasedSemanticSplitter(TextSplitter):
    def __init__(self, model, tokenizer, device, n_std=1.0):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.n_std = n_std

    def get_embeddings(self, sentences):
        inputs = self.tokenizer(sentences, return_tensors="pt", padding=True, truncation=True, max_length=128).to(self.device)
        with torch.no_grad():
            out = self.model(**inputs, output_hidden_states=True)
            return out.hidden_states[-1][:, 0, :]

    def split_text(self, text):
        sents = text.split("。")
        embeds = self.get_embeddings(sents)
        sims = [F.cosine_similarity(embeds[i].unsqueeze(0), embeds[i+1].unsqueeze(0)).item() for i in range(len(embeds)-1)]
        thr = np.mean(sims) - self.n_std * np.std(sims)
        chunks, current = [], [sents[0]]
        for i in range(1, len(sents)):
            if sims[i-1] < thr:
                chunks.append("。".join(current))
                current = [sents[i]]
            else:
                current.append(sents[i])
        chunks.append("。".join(current))
        return chunks