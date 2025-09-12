# Histopathology Cancer Detection

This project is focused on computer vision techniques for histopathologic cancer detection using image data. It leverages Python and popular libraries for image processing and machine learning.

## Project Structure

- `main.py` — Main script for running experiments and analysis.
- `cvenv2/` — Python virtual environment for package management.
- `histopathologic-cancer-detection/`
  - `sample_submission.csv` — Example submission file for competitions.
  - `train_labels.csv` — Labels for training images.
  - `test/` — Test images in `.tif` format.
  - `train/` — Training images.
  - `train_data/`, `train_small/` — Additional training datasets.

## Setup Instructions

1. **Clone the repository**
2. **Create and activate the virtual environment** (already present as `cvenv2`):
   - Windows PowerShell:
     ```powershell
     .\cvenv2\Scripts\Activate.ps1
     ```
3. **Install required packages**:
   - Use pip to install dependencies (add your requirements to `requirements.txt` if needed):
     ```powershell
     pip install -r requirements.txt
     ```

## Usage

Run the main script:
```powershell
python main.py
```

## Data
- The dataset consists of histopathology images in `.tif` format, with corresponding labels for training.
- For more details on the dataset, refer to the [Kaggle competition page](https://www.kaggle.com/c/histopathologic-cancer-detection).

## Contributing
Feel free to open issues or submit pull requests for improvements.

## License
This project is licensed under the MIT License.
