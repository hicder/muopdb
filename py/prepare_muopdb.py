import argparse
import json
import os
import shutil

"""
Run this script to prepare a MUOPDB collection.
Example usage:
python3 py/prepare_muopdb.py --indices_path=/mnt/muopdb/indices --data_path=/mnt/muopdb/data --collection_name=test-collection-4 --config_path=py/collection_config.json
"""
def main():
    # Add command line arguments --indices and --data with argparse
    parser = argparse.ArgumentParser(description='Prepare MUOPDB')
    parser.add_argument('--indices_path', type=str, help='Path to the indices')
    parser.add_argument('--data_path', type=str, help='Path to the data')
    parser.add_argument('--collection_name', type=str, help='Name of the collection')
    parser.add_argument('--config_path', type=str, help='Path to the config file')
    args = parser.parse_args()

    # Check if the indices and data directories exist. If not, create them.
    if not os.path.exists(args.indices_path):
        os.makedirs(args.indices_path)
    if not os.path.exists(args.data_path):
        os.makedirs(args.data_path)

    # If collection_name is not specified, exit
    if args.collection_name is None:
        return

    # Check if collection exists in the data directory. If it does, delete it. Then create it.
    if os.path.exists(os.path.join(args.data_path, args.collection_name)):
        raise Exception('Collection already exists')
    os.makedirs(os.path.join(args.data_path, args.collection_name))

    # Copy the config file to the collection directory
    shutil.copy(args.config_path, os.path.join(args.data_path, args.collection_name))
    # Add a version_0 file to the collection directory with the content
    # {
    #     "segments": []
    # }
    with open(os.path.join(args.data_path, args.collection_name, 'version_0'), 'w') as f:
        json.dump({'segments': []}, f, indent=2)

    # Find the latest version of the indices (each will have the name version_<version_number>)
    versions = [int(x.split('_')[1]) for x in os.listdir(args.indices_path) if x.startswith('version')]
    data = {}
    if len(versions) == 0:
        latest_version = 0
        data['collections'] = [{'name': args.collection_name}]
    else:
        # Get the latest version number
        latest_version = max(versions)
        with open(os.path.join(args.indices_path, 'version_{}'.format(latest_version)), 'r') as f:
            # parse the json file
            data = json.load(f)
            # add the new collection to the list of collections
            data['collections'].append({'name': args.collection_name})

    # write the updated json file
    with open(os.path.join(args.indices_path, 'version_{}'.format(latest_version + 1)), 'w') as f:
        json.dump(data, f, indent=2)



if __name__ == '__main__':
    main()
