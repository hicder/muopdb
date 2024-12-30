import muopdb_client as mp
import google.generativeai as genai
import os
import ollama
import logging
import h5py

def insert_all_documents(muopdb_client: mp.IndexServerClient, collection_name, embeddings):
    logging.info("Inserting documents...")
    batch_size = 10000
    total_embeddings = len(embeddings)
    
    for start_idx in range(0, total_embeddings, batch_size):
        end_idx = min(start_idx + batch_size, total_embeddings)
        batch_embeddings = embeddings[start_idx:end_idx]
        batch_ids = list(range(start_idx + 1, end_idx + 1))
        
        # Flatten the batch embeddings into a single list of floats
        flattened_vectors = [float(value) for embedding in batch_embeddings for value in embedding]
        muopdb_client.insert(
            collection_name=collection_name,
            ids=batch_ids,
            vectors=flattened_vectors
        )
        
        if end_idx % (batch_size * 10) == 0 or end_idx == total_embeddings:
            logging.info(f"Inserted documents up to id {end_idx}")

    logging.info("Start indexing documents...") 
    muopdb_client.flush(collection_name=collection_name)
    logging.info("Indexing documents completed. Documents are ready to be queried.")

# main function
def main():
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )
    
    muopdb_client = mp.IndexServerClient()
    logging.info("=========== Inserting documents ===========")

    # Read the source embedding file
    with h5py.File("/mnt/muopdb/raw/1m_embeddings.hdf5", "r") as f:
        embeddings = f['embeddings'][:]

    # Insert the embeddings into MuopDB
    insert_all_documents(muopdb_client, "test-collection-1", embeddings)
    logging.info("=========== Inserted all documents ===========")


if __name__ == "__main__":
    main()
