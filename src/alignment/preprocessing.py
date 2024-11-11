def preprocess_function(examples, tokenizer, max_seq_length):
    texts = examples.get("sentence", examples.get("text", []))
    labels = examples["label"]
    
    # Debug prints for input
    print("\nDEBUG INPUT:")
    print(f"Texts length: {len(texts)}")
    print(f"Labels length: {len(labels)}")
    print(f"First few texts: {texts[:2]}")
    print(f"First few labels: {labels[:2]}")
    
    # Create pairs within the same batch
    texts_1 = []
    texts_2 = []
    pair_labels = []
    
    # Create pairs from consecutive examples regardless of their labels
    for i in range(len(texts) - 1):
        texts_1.append(texts[i])
        texts_2.append(texts[i + 1])
        pair_labels.append(1 if labels[i] == labels[i + 1] else 0)
    
    # Debug prints for pairs
    print("\nDEBUG PAIRS:")
    print(f"Pairs created: {len(texts_1)}")
    if texts_1:
        print(f"First pair example: {texts_1[0]} <-> {texts_2[0]} (label: {pair_labels[0]})")
    
    # Skip if no pairs were created
    if not texts_1:
        return {"input_ids": [], "attention_mask": [], "input_ids_pair": [], "attention_mask_pair": [], "labels": []}
    
    # Tokenize both sequences
    result_1 = tokenizer(
        texts_1,
        padding=True,
        max_length=max_seq_length,
        truncation=True,
    )
    
    result_2 = tokenizer(
        texts_2,
        padding=True,
        max_length=max_seq_length,
        truncation=True,
    )

    # Debug print tokenizer outputs
    print("\nDEBUG TOKENIZER:")
    print(f"Tokenizer output keys: {result_1.keys()}")
    print(f"First sequence shape: {len(result_1['input_ids'])}")
    print(f"Second sequence shape: {len(result_2['input_ids'])}")
    
    # Create result with all necessary fields
    result = {}
    for key in result_1.keys():
        result[key] = result_1[key]
        result[f"{key}_pair"] = result_2[key]
    result["labels"] = pair_labels

    # Final debug print
    print("\nDEBUG FINAL OUTPUT:")
    print(f"Final output keys: {result.keys()}")
    print("Final output shapes:", {key: len(value) for key, value in result.items()})
    
    return result