
# 笔画级 Tokenizer（直接使用你的 zh2text 文件）

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import json

class StrokeTokenizer:
    """笔画级 tokenizer（使用 zh2text 映射）"""
    
    def __init__(self, zh2text_file, max_length=256):
        """
        Args:
            zh2text_file: 你的 zh2text 文件路径
            max_length: 笔画序列更长，建议 256
        """
        self.max_length = max_length
        
        # 特殊 token
        self.pad_token = '[PAD]'
        self.unk_token = '[UNK]'
        self.cls_token = '[CLS]'
        self.sep_token = '[SEP]'
        self.special_tokens = [self.pad_token, self.unk_token, 
                               self.cls_token, self.sep_token]
        
        # 加载中文->笔画映射
        self.char2stroke = self._load_zh2text(zh2text_file)
        
        # 构建笔画词表
        self._build_stroke_vocab()
        
        print(f"✓ 笔画映射: {len(self.char2stroke)} 个字符")
        print(f"✓ 笔画词表: {len(self.stroke2idx)} 个token")
    
    def _load_zh2text(self, filepath):
        """加载 zh2text 文件"""
        char2stroke = {}
        
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                # 解析格式: "一    HHH"
                parts = line.split()
                if len(parts) >= 2:
                    char = parts[0]
                    stroke = parts[1]
                    char2stroke[char] = stroke
        
        return char2stroke
    
    def _build_stroke_vocab(self):
        """构建笔画符号词表"""
        # 收集所有笔画符号
        all_strokes = set()
        for stroke_seq in self.char2stroke.values():
            all_strokes.update(list(stroke_seq))
        
        # 排序（保证一致性）
        all_strokes = sorted(list(all_strokes))
        
        # 构建映射: 特殊token + 笔画符号
        vocab = self.special_tokens + all_strokes
        self.stroke2idx = {s: i for i, s in enumerate(vocab)}
        self.idx2stroke = {i: s for i, s in enumerate(vocab)}
        
        print(f"  笔画符号: {all_strokes}")
    
    def char_to_strokes(self, char):
        """将单个字符转换为笔画序列"""
        return self.char2stroke.get(char, self.unk_token)
    
    def text_to_strokes(self, text):
        """将文本转换为笔画序列"""
        stroke_chars = []
        
        for char in text:
            strokes = self.char_to_strokes(char)
            stroke_chars.extend(list(strokes))  # 展开为单个笔画
        
        return stroke_chars
    
    def __call__(self, text, max_length=None, padding='max_length', 
                 truncation=True, return_tensors='pt'):
        """编码文本为笔画 token ids"""
        max_len = max_length or self.max_length
        
        # 1. 文本 -> 笔画序列
        stroke_chars = self.text_to_strokes(text)
        
        # 2. 笔画 -> ids
        ids = [self.stroke2idx.get(s, self.stroke2idx[self.unk_token]) 
               for s in stroke_chars]
        
        # 3. 添加 [CLS] 和 [SEP]
        ids = [self.stroke2idx[self.cls_token]] + ids + [self.stroke2idx[self.sep_token]]
        
        # 4. 截断
        if truncation and len(ids) > max_len:
            ids = ids[:max_len]
        
        attention_mask = [1] * len(ids)
        
        # 5. 填充
        if padding == 'max_length':
            pad_len = max_len - len(ids)
            ids += [self.stroke2idx[self.pad_token]] * pad_len
            attention_mask += [0] * pad_len
        
        result = {
            'input_ids': torch.tensor(ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long)
        }
        
        if return_tensors == 'pt':
            result['input_ids'] = result['input_ids'].unsqueeze(0)
            result['attention_mask'] = result['attention_mask'].unsqueeze(0)
        
        return result
    
    def decode(self, ids):
        """解码（用于调试）"""
        return [self.idx2stroke.get(i, '[UNK]') for i in ids]
    
    @property
    def vocab_size(self):
        return len(self.stroke2idx)
    
    def save_vocab(self, filepath):
        """保存词表"""
        data = {
            'stroke2idx': self.stroke2idx,
            'char2stroke_size': len(self.char2stroke)
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)



if __name__ == "__main__":
    # 测试 tokenizer
    print("="*60)
    print("笔画级 Tokenizer 测试")
    print("="*60)
    
    # 创建 tokenizer
    stroke_tokenizer = StrokeTokenizer('zh2text.txt', max_length=256)
    
    # 测试编码
    text = "我得了糖尿病"
    print(f"\n原文: {text}")
    
    # 查看笔画转换
    print("\n字符 -> 笔画:")
    for char in text:
        strokes = stroke_tokenizer.char_to_strokes(char)
        print(f"  {char} -> {strokes}")
    
    # 编码
    encoded = stroke_tokenizer(text, return_tensors=None)
    print(f"\n笔画序列长度: {sum(encoded['attention_mask'])}")
    print(f"Token IDs (前20个): {encoded['input_ids'][:20]}")
    
    # 解码验证
    decoded = stroke_tokenizer.decode(encoded['input_ids'][:20])
    print(f"解码 (前20个): {decoded}")


