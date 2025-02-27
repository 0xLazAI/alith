from alith import RemoteModelEmbeddings

print(
    RemoteModelEmbeddings(
        model="your embeddings model name",
        api_key="your API key",
        base_url="base url",
    ).embed_texts(["Hello", "World"])
)
