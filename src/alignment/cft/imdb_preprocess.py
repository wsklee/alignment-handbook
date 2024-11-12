from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
import random

class IMDBPreprocess:
    def __init__(self, model_name):
        # Load IMDB dataset
        self.ds = load_dataset("imdb")['train']
        
        # Load tokenizer from HuggingFace
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Group reviews by sentiment
        self.positive_reviews = [item['text'] for item in self.ds if item['label'] == 1]
        self.negative_reviews = [item['text'] for item in self.ds if item['label'] == 0]
        
        self._create_pairs()
        self._preprocess()
    
    def _create_pairs(self):
        """Create positive and negative pairs from IMDB reviews"""
        pairs = []
        for review in self.positive_reviews:
            # Get another positive review as positive pair
            pos_candidates = [r for r in self.positive_reviews if r != review]
            pos_pair = random.choice(pos_candidates)
            # Get a negative review as hard negative
            hard_neg = random.choice(self.negative_reviews)
            
            pairs.append({
                'sent0': review,
                'sent1': pos_pair,
                'hard_neg': hard_neg
            })
        
        self.ds = pairs
    
    def _tokenize(self, text, id):
        out = self.tokenizer(text, 
                    padding='max_length', 
                    truncation=True, 
                    return_tensors="pt", 
                    max_length=512)
        
        print(f"DEBUG - Tokenizer output for {id}:", {k: v.shape for k, v in out.items()})
        
        out[id + '_input_ids'] = out.pop('input_ids')
        out[id + '_attention_mask'] = out.pop('attention_mask')
        return out
    
    def _preprocess(self):
        print("DEBUG - Starting preprocessing...")
        # Convert to HF dataset
        self.ds = Dataset.from_list(self.ds)
        
        # Tokenize all texts
        self.ds = self.ds.map(
            lambda x: self._tokenize(x['sent0'], 'sent0'), batched=True)
        self.ds = self.ds.map(
            lambda x: self._tokenize(x['sent1'], 'sent1'), batched=True)
        self.ds = self.ds.map(
            lambda x: self._tokenize(x['hard_neg'], 'hard_neg'), batched=True)
        
        # Set format for training
        self.ds.set_format(
            type="torch", 
            columns=["sent0_input_ids", "sent0_attention_mask",
                     "sent1_input_ids", "sent1_attention_mask",
                     "hard_neg_input_ids", "hard_neg_attention_mask"]
        )
        print("DEBUG - Final dataset features:", self.ds.features)
        print("DEBUG - First example:", self.ds[0])

if __name__ == "__main__":
    imdb_prep = IMDBPreprocess()
    imdb_prep.ds.save_to_disk("./processed/")