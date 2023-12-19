import json
import os

for file in os.listdir():
    if file.endswith('.json'):
        with open(file) as f:
            data = json.load(f)
            data = data['data']
            for i in range(len(data)):
                data[i]['wav'] = data[i]['wav'].replace('/mnt/w2g2/data_CRS', '/mnt/data/tungtran/data/data_CRS')
            
        # Create the final JSON structure
        json_data = {
            "data": data
        }

        with open(file, 'w') as json_file:
            json.dump(json_data, json_file, indent=2)
