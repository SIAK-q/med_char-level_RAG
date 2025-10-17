# pinyin_tokenizer.py
# 拼音级 Tokenizer（结构化：声母 + 韵母 + 声调）

import torch
import json
from pypinyin import pinyin, Style


class PinyinTokenizer:
    def __init__(self, max_length=128):
        self.max_length = max_length
        self.pad_token = '[PAD]'
        self.unk_token = '[UNK]'
        self.cls_token = '[CLS]'
        self.sep_token = '[SEP]'
        self.special_tokens = [
            self.pad_token, self.unk_token, self.cls_token, self.sep_token
        ]
        self._build_vocab()
        self.pad_token_id = self.token2id[self.pad_token]

    def _build_vocab(self):
        initials = [
            'b', 'p', 'm', 'f', 'd', 't', 'n', 'l',
            'g', 'k', 'h', 'j', 'q', 'x',
            'zh', 'ch', 'sh', 'r', 'z', 'c', 's', 'y', 'w'
        ]
        finals = [
            'a', 'o', 'e', 'i', 'u', 'v', 'ai', 'ei', 'ui', 'ao', 'ou', 'iu',
            'ie', 've', 'an', 'en', 'in', 'un', 'vn', 'ang', 'eng', 'ing', 'ong'
        ]
        tones = ['1', '2', '3', '4', '5']

        vocab = self.special_tokens + initials + finals + tones
        self.token2id = {t: i for i, t in enumerate(vocab)}
        self.id2token = {i: t for t, i in self.token2id.items()}

    def split_pinyin(self, p):
        tone = p[-1] if p and p[-1].isdigit() else '5'
        p_core = p[:-1] if tone != '5' else p

        for i in range(len(p_core), 0, -1):
            if p_core[:i] in self.token2id:
                initial = p_core[:i]
                final = p_core[i:]
                return [initial, final, tone]

        return [self.unk_token, self.unk_token, tone]

    def encode(self, text):
        # 转拼音
        syllables = pinyin(text, style=Style.TONE3, strict=False, errors='default')
        ids = []

        for syl in syllables:
            # 跳过无效条目（pypinyin可能返回 []、['']、[' '])
            if not syl or not syl[0].strip():
                continue

            tokens = self.split_pinyin(syl[0])
            ids.extend([
                self.token2id.get(t, self.token2id[self.unk_token]) for t in tokens
            ])

        # ⚙️ 保底逻辑：防止整句都为空
        if not ids:
            ids = [self.token2id[self.unk_token]]

        return ids

    def __call__(self, text, max_length=None, padding='max_length',
                 truncation=True, return_tensors='pt'):
        max_len = max_length or self.max_length
        ids = self.encode(text)
        ids = [self.token2id[self.cls_token]] + ids + [self.token2id[self.sep_token]]

        if truncation:
            ids = ids[:max_len]
        attention_mask = [1] * len(ids)

        if padding == 'max_length' and len(ids) < max_len:
            pad_len = max_len - len(ids)
            ids += [self.token2id[self.pad_token]] * pad_len
            attention_mask += [0] * pad_len

        result = {
            'input_ids': torch.tensor(ids).unsqueeze(0),
            'attention_mask': torch.tensor(attention_mask).unsqueeze(0)
        }
        return result

    @property
    def vocab_size(self):
        return len(self.token2id)

    def save_vocab(self, filepath):
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.token2id, f, ensure_ascii=False, indent=2)

