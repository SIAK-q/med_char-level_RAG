# -*- coding: utf-8 -*-
"""
stroke_tokenizer.py
笔画级 Tokenizer（自动定位 zh2letter.txt）
"""

import os
import torch
import json

class StrokeTokenizer:
    def __init__(self, zh2text_file=None, max_length=256):
        """
        Args:
            zh2text_file: zh2letter.txt 文件路径。如果为空，则自动在当前目录下寻找。
            max_length: 最大长度
        """
        # 自动定位 zh2letter.txt
        if zh2text_file is None:
            # 使用当前文件所在目录去定位 zh2letter.txt
            base_dir = os.path.dirname(__file__)
            zh2text_file = os.path.join(base_dir, "zh2letter.txt")
        if not os.path.exists(zh2text_file):
            print(f"[警告] 未找到 {zh2text_file}")

        self.max_length = max_length
        self.pad_token = '[PAD]'
        self.unk_token = '[UNK]'
        self.cls_token = '[CLS]'
        self.sep_token = '[SEP]'
        self.special_tokens = [
            self.pad_token, self.unk_token, self.cls_token, self.sep_token
        ]

        # 加载笔画映射
        self.char2stroke = self._load_zh2text(zh2text_file)
        # 构建词表
        self._build_vocab()
        self.pad_token_id = self.stroke2id[self.pad_token]

    def _load_zh2text(self, filepath):
        """读取汉字到笔画的映射"""
        mapping = {}
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        mapping[parts[0]] = parts[1]
        except FileNotFoundError:
            print(f"[Warning] file {filepath} not found.")
        return mapping

    def _build_vocab(self):
        """根据映射构建笔画词表"""
        strokes = set()
        for v in self.char2stroke.values():
            for s in v:
                strokes.add(s)
        strokes = sorted(list(strokes))
        self.stroke2id = {s: i + 4 for i, s in enumerate(strokes)}  # 预留4个特殊符号
        self.stroke2id[self.pad_token] = 0
        self.stroke2id[self.unk_token] = 1
        self.stroke2id[self.cls_token] = 2
        self.stroke2id[self.sep_token] = 3
        self.id2stroke = {v: k for k, v in self.stroke2id.items()}
        self.vocab_size = len(self.stroke2id)

    def text_to_strokes(self, text):
        """汉字 -> 笔画序列"""
        strokes = []
        for ch in text:
            if ch in self.char2stroke:
                strokes.extend(list(self.char2stroke[ch]))
            else:
                strokes.append(self.unk_token)
        return strokes

    def __call__(self, texts, return_tensors=None, padding=True, truncation=True, max_length=None):
        """兼容 transformers 调用风格"""
        if isinstance(texts, str):
            texts = [texts]
        if max_length is None:
            max_length = self.max_length

        all_ids = []
        for text in texts:
            strokes = self.text_to_strokes(text)
            ids = [self.stroke2id.get(self.cls_token)]
            ids.extend([self.stroke2id.get(s, self.stroke2id[self.unk_token]) for s in strokes])
            ids.append(self.stroke2id.get(self.sep_token))
            if truncation and len(ids) > max_length:
                ids = ids[:max_length]
            all_ids.append(ids)

        # padding
        if padding:
            pad_id = self.stroke2id[self.pad_token]
            max_len = max(len(ids) for ids in all_ids)
            for ids in all_ids:
                ids += [pad_id] * (max_len - len(ids))

        input_ids = torch.tensor(all_ids, dtype=torch.long)
        attention_mask = (input_ids != self.stroke2id[self.pad_token]).long()

        if return_tensors == "pt":
            return {"input_ids": input_ids, "attention_mask": attention_mask}
        else:
            return {"input_ids": all_ids, "attention_mask": attention_mask}