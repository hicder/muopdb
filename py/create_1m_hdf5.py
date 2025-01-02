import h5py
import numpy as np

def create_hdf5(input_path, output_path, batch_size=100000):
    # Create HDF5 file
    with h5py.File(output_path, 'w') as hf:
        # Initialize dataset with unknown size
        dataset = hf.create_dataset('embeddings', 
                                  shape=(0, 768), 
                                  maxshape=(None, 768),
                                  dtype='float32')
        
        # Process file in batches
        with open(input_path, 'r') as f:
            while True:
                # Read batch of lines
                lines = [f.readline() for _ in range(batch_size)]
                if not lines[0]:  # End of file
                    break
                    
                # Parse lines to numpy array
                embeddings = np.array([np.fromstring(line.strip('[]\n'), sep=',') 
                                     for line in lines if line.strip()])
                
                # Resize dataset and append new data
                dataset.resize(dataset.shape[0] + embeddings.shape[0], axis=0)
                dataset[-embeddings.shape[0]:] = embeddings

if __name__ == '__main__':
    input_file = '/mnt/muopdb/raw/1m_embeddings.txt'
    output_file = '/mnt/muopdb/1m_embeddings.hdf5'
    create_hdf5(input_file, output_file)
