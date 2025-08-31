import json
import os

def load_config(config_path="src/detection_config.json"):
    """
    Loads the detection configuration from a JSON file.
    """
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config['detection_config']
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return None