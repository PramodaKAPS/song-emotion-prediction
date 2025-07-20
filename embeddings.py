def get_bert_embeddings(texts, tokenizer, model, device, max_length=512):
    embeddings = []
    for text in texts:
        if not isinstance(text, str) or len(text.strip()) == 0:
            embeddings.append(np.zeros(768))
            continue
        tokens = tokenizer.tokenize(text)
        if len(tokens) > max_length - 2:
            print(f"Warning: Truncating long text (original tokens: {len(tokens)})")
            tokens = tokens[:max_length - 2]  # Explicit truncation
        chunks = [tokens[i:i + (max_length - 2)] for i in range(0, len(tokens), max_length - 2)]
        sentence_embeddings = []
        for chunk in chunks:
            chunk_text = tokenizer.convert_tokens_to_string(chunk)
            inputs = tokenizer(chunk_text, return_tensors='pt', truncation=True, padding=True, max_length=max_length)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs)
            embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
            sentence_embeddings.append(embedding)
        avg_embedding = np.mean(sentence_embeddings, axis=0) if sentence_embeddings else np.zeros(768)
        embeddings.append(avg_embedding)
    return np.array(embeddings)
