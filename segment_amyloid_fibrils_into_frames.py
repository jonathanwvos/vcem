from PIL import Image, ImageSequence
from os import listdir
import json
from pathlib import Path


TARGET_DIRECTORY = 'amyloid-fibrils'
DATA_PATH = Path('data')
RAW_PATH = DATA_PATH / 'raw'
INPUT_PATH = RAW_PATH / TARGET_DIRECTORY
OUTPUT_PATH = DATA_PATH / 'frame-segmented' / TARGET_DIRECTORY
MANIFEST_PATH = DATA_PATH / f'{TARGET_DIRECTORY}-manifest.json'


with open(MANIFEST_PATH, 'w') as manifest_file:
  manifest = {
    'files': [],
    'metadata': {
      'total_no_files': 0,
      'total_no_frames': 0
    }
  }

  total_no_files = 0
  total_no_frames = 0

  for file_idx, file_name in enumerate(listdir(INPUT_PATH)):
    total_no_files += 1
    file_path = INPUT_PATH / file_name
    manifest['files'].append({
      'raw_file_name': file_name,
      'index_file_name': file_idx,
      'raw_file_path': str(file_path),
      'segmented_files_path': str(OUTPUT_PATH)
    })

    print(f"Processing {file_name}")
    with Image.open(file_path) as img:
      for frame_idx, frame in enumerate(ImageSequence.Iterator(img)):
        frame.save(OUTPUT_PATH / f"{file_idx}_frame_{frame_idx:03d}.tiff")
        total_no_frames += 1

  manifest['metadata']['total_no_files'] = total_no_files
  manifest['metadata']['total_no_frames'] = total_no_frames

  manifest_file.write(json.dumps(manifest))
