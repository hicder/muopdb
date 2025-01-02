import muopdb_client as mp
import ollama
import time

if __name__ == "__main__":
    # Example usage for IndexServer
    muopdb_client = mp.IndexServerClient()
    query = "baby goes to school"
    query_vector = ollama.embeddings(model='nomic-embed-text', prompt=query)["embedding"]

    # Read back the raw data to print the responses
    with open("/mnt/muopdb/raw/1m_sentences.txt", "r") as f:
        sentences = [line.strip() for line in f]

    i = 0
    while i < 5:
        start = time.time()
        search_response = muopdb_client.search(
            index_name="test-collection-1",
            vector=query_vector,
            top_k=5,
            ef_construction=50,
            record_metrics=True
        )
        end = time.time()
        print(f"Time taken for search: {end - start} seconds")

        print(f"Number of results: {len(search_response.ids)}")
        print("================")
        for id in search_response.ids:
            print(f"RESULT: {sentences[id - 1]}")
        print("================")
        i += 1
