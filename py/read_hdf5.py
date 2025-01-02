import h5py

def read_first_and_last_five_points():
    with h5py.File("/mnt/muopdb/raw/1m_embeddings.hdf5", "r") as f:
        # Assuming the dataset is named 'embeddings'
        dataset = f['embeddings']
        first_five = dataset[:5]
        last_five = dataset[-5:]
        print("First 5 points:")
        print(first_five)
        print("\nLast 5 points:")
        print(last_five)

if __name__ == "__main__":
    read_first_and_last_five_points()