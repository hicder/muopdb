import ollama
import time
import sys

if __name__ == "__main__":
    # Read the first args
    input_file = sys.argv[1]
    with open(input_file, "r") as f:
        sentences = f.readlines()

    start = time.time()
    output_file = f"{input_file}_embeddings.txt"
    with open(output_file, "w") as f:
        for sentence in sentences:
            result = ollama.embeddings(model='nomic-embed-text', prompt=sentence)
            f.write(f"{result['embedding']}\n")
        
    end = time.time()
    print(f"Time taken: {end - start} seconds")
