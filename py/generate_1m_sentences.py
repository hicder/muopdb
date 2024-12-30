import datasets

if __name__ == "__main__":
    ds = datasets.load_dataset("agentlans/high-quality-english-sentences")
    train = ds["train"]
    with open("/tmp/1m_sentences.txt", "w") as f:
        for ln in train:
            f.write(ln["text"] + "\n")
    f.flush() 
