# Image Complexity Segmentation

Small Python framework for segmenting images based on their local structural complexity.

Instead of detecting objects, the program classifies image regions into different complexity levels.

## Features

The following complexity metrics are currently supported:

- Shannon Entropy
- Delentropy
- Mean Frequency
- Median Frequency

The metric values are combined and segmented using K-Means.

The resulting mask contains:

- `0` = Low complexity
- `1` = Medium complexity
- `2` = High complexity

## Usage

Start the user interface:

```bash
python run_ui.py