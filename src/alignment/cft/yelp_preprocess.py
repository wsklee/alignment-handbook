from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
import random
from tqdm import tqdm

class YelpPreprocess:
    def __init__(self, model_name):
        # Load Yelp Polarity dataset
        self.raw_ds = load_dataset("yelp_polarity", split="train")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Split reviews by sentiment
        self.positive_reviews = self.raw_ds.filter(lambda x: x['label'] == 1)['text']
        self.negative_reviews = self.raw_ds.filter(lambda x: x['label'] == 0)['text']
        
        # Create pairs and preprocess the dataset
        self._create_pairs()
        self._preprocess()

    def _create_pairs(self):
        """Efficiently create positive and negative pairs"""
        pairs = []
        print("DEBUG - Generating pairs...")
        for review in tqdm(self.positive_reviews):
            # Randomly sample a positive pair
            pos_pair = random.choice(self.positive_reviews)
            while pos_pair == review:  # Ensure it's not the same review
                pos_pair = random.choice(self.positive_reviews)
            
            # Randomly sample a negative review
            hard_neg = random.choice(self.negative_reviews)
            
            pairs.append({
                'sent0': review,
                'sent1': pos_pair,
                'hard_neg': hard_neg
            })
        
        # Convert to Hugging Face Dataset for efficient processing
        self.ds = Dataset.from_list(pairs)

    def _tokenize(self, examples, text_column, prefix):
        """Tokenize a batch of texts and return as a dictionary with prefixed keys"""
        tokenized = self.tokenizer(
            examples[text_column],
            padding="max_length",
            truncation=True,
            max_length=256
        )
        return {
            f"{prefix}_input_ids": tokenized['input_ids'],
            f"{prefix}_attention_mask": tokenized['attention_mask']
        }

    def _preprocess(self):
        """Tokenize dataset and prepare for training"""
        print("DEBUG - Tokenizing dataset...")
        
        # Tokenize sent0
        self.ds = self.ds.map(
            lambda x: self._tokenize(x, "sent0", "sent0"),
            batched=True
        )
        
        # Tokenize sent1
        self.ds = self.ds.map(
            lambda x: self._tokenize(x, "sent1", "sent1"),
            batched=True
        )
        
        # Tokenize hard_neg
        self.ds = self.ds.map(
            lambda x: self._tokenize(x, "hard_neg", "hard_neg"),
            batched=True
        )
        
        # Set format for training
        self.ds.set_format(
            type="torch",
            columns=[
                "sent0_input_ids", "sent0_attention_mask",
                "sent1_input_ids", "sent1_attention_mask",
                "hard_neg_input_ids", "hard_neg_attention_mask"
            ]
        )
        print("DEBUG - Dataset tokenized and ready for training.")
        print("DEBUG - First example:", self.ds[0])

if __name__ == "__main__":
    yelp_prep = YelpPreprocess("distilbert-base-uncased")
    yelp_prep.ds.save_to_disk("./processed_yelp_dataset")
