def get_bert_embeddings(texts, tokenizer, model, device, max_length=512):
    embeddings = []
    for text in texts:
        if not isinstance(text, str) or len(text.strip()) == 0:
            embeddings.append(np.zeros(768))
            continue
        tokens = tokenizer.tokenize(text)
        if len(tokens) > max_length - 2:
            print(f"Warning: Truncating long text (original tokens: {len(tokens)})")
            tokens = tokens[:max_length - 2]
        chunk_text = tokenizer.convert_tokens_to_string(tokens)
        inputs = tokenizer(chunk_text, return_tensors='pt', truncation=True, padding=True, max_length=max_length).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
        embeddings.append(embedding)
    return np.array(embeddings)
