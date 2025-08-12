import json
import os
import shutil
from tqdm import tqdm

def process_json(input_file, output_directory, meta_directory):
    # Read the JSON file
    with open(input_file, 'r') as json_file:
        data = json.load(json_file)

    # Create the output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)

    # Process each entry in the "data" field
    for entry in tqdm(data["data"]):
        old_path = entry["wav"]
        label = entry["labels"]

        # Construct the new path in the output directory
        new_filename = os.path.basename(old_path)
        new_path = os.path.join(output_directory, new_filename)

        # Move the file to the new directory
        shutil.copy(old_path, new_path)

        # Update the "wav" field in the entry
        entry["wav"] = new_path

    # Write the updated data back to the JSON file
    output_file = os.path.join(meta_directory, "train.json" if "train" in input_file else "val.json")
    with open(output_file, 'w') as json_out:
        json.dump(data, json_out, indent=2)

if __name__ == "__main__":
    # Replace 'input.json' with the actual name of your JSON file
    DIR = 'coral_sound_indo_location_30min_few_shot'
    json_input_file = f'{DIR}/train.json'
    json_input_file_val = f'{DIR}/val.json'
    meta_dir = f"/mnt/data/tungtran/eval_data/{DIR}"

    # Replace 'output_directory' with the desired output directory path
    output_directory = f'/mnt/data/tungtran/eval_data/{DIR}/raw_audio'

    process_json(json_input_file, output_directory, meta_dir)
    process_json(json_input_file_val, output_directory, meta_dir)
    shutil.copy(os.path.join(DIR, 'class_labels_indices.csv'), os.path.join(meta_dir, 'class_labels_indices.csv'))
