# PyTorch Image Classification - Cats vs Dogs

<hr />

A PyTorch-based binary image classification project that distinguishes between cats and dogs (dataset from Kaggle) using both custom CNN architecture and transfer learning with VGG16.



![Result](result.jpg)


**Supporting YouTube Videos:**

1. [Image Classification (from project Structure to Inference)](https://www.youtube.com/watch?v=I84qsrJMq7Y)
2. [Enhancing Image Classification with Transfer Learning](https://www.youtube.com/watch?v=CRua07FmjQM)

**Increase the audiospeed to 1.25X for better experience**



## Project Structure

```
master-pytorch-image-classification/
├── data/
│   ├── training_set/
│   │   └── training_set/
│   │       ├── cats/
│   │       └── dogs/
│   └── test_set/
│       └── test_set/
│           ├── cats/
│           └── dogs/
├── models/
│   ├── imclfv1-0.pth
│   └── imclfv2-0.pth
├── src/
│   ├── data_setup.py      # Custom dataset and data loading utilities
│   ├── engine.py          # Training, testing, and prediction functions
│   ├── model.py           # Model architectures (Custom CNN and VGG16 transfer learning)
│   ├── train.py           # Main training script
│   └── utils.py           # Utility functions for saving models and plotting
├── inference.py           # Script for making predictions on new images
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Features

- **Two Model Architectures:**
  - Custom CNN (`SimpleBinaryClassifier`): A lightweight 3-layer convolutional neural network
  - VGG16 Transfer Learning: Pre-trained VGG16 with custom classifier head

- **Custom Dataset Handling:** Efficient data loading with PyTorch DataLoader
- **Training Pipeline:** Complete training loop with validation
- **Model Persistence:** Automatic model saving during training
- **Inference Pipeline:** Easy prediction on new images
- **Visualization:** Training metrics plotting and inference result display

## Installation

1. Clone or download this repository
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Dataset Setup

The project expects the following directory structure for your data:

```
data/
├── training_set/
│   └── training_set/
│       ├── cats/          # Cat images (.jpg files)
│       └── dogs/          # Dog images (.jpg files)
└── test_set/
    └── test_set/
        ├── cats/          # Cat test images (.jpg files)
        └── dogs/          # Dog test images (.jpg files)
```

Check the **YouTube Video** for more details on how to download your dataset from **Kaggle** and prepare for training.

## Usage

### Training a Model

Run the training script with default parameters:

```bash
python src/train.py
```

Or customize the training parameters:

```bash
python src/train.py --batch_size 64 --learning_rate 0.0001 --num_epochs 30 --model_name my_model.pth
```

**Available Arguments:**
- `--batch_size`: Batch size for training (default: 32)
- `--learning_rate`: Learning rate for optimizer (default: 0.001)
- `--num_epochs`: Number of training epochs (default: 20)
- `--train_dir`: Custom training directory path
- `--test_dir`: Custom test directory path
- `--model_name`: Name for the saved model file (default: 01-trained-model.pth)
- `--verbose`: Enable verbose output

### Making Predictions

Use the inference script to make predictions on new images:

```bash
python inference.py
```

The script will:
1. Load a pre-trained model from the `models/` directory
2. Make predictions on test images
3. Display the results with visualization

### Model Architectures

#### Custom CNN (SimpleBinaryClassifier)
- 3 convolutional layers with ReLU activation and MaxPooling
- Dropout layers for regularization
- Final sigmoid activation for binary classification
- Input: 224x224x3 RGB images
- Output: Single probability score (0-1)

#### VGG16 Transfer Learning
- Pre-trained VGG16 backbone (frozen weights)
- Custom classifier head with dropout
- Adaptive average pooling
- Input: 224x224x3 RGB images
- Output: Single probability score (0-1)

## Training Process

1. **Data Loading:** Images are automatically resized to 224x224 and normalized
2. **Model Selection:** VGG16 transfer learning is used by default
3. **Training Loop:** Model trains for specified epochs with validation
4. **Model Saving:** Best models are saved based on validation loss
5. **Metrics:** Training/validation accuracy and loss are tracked and saved

## Model Performance

The trained models achieve binary classification between cats (0) and dogs (1):
- **Threshold:** 0.5 (predictions > 0.5 = Dog, ≤ 0.5 = Cat)
- **Loss Function:** Binary Cross Entropy (BCE)
- **Optimizer:** Adam with weight decay
- **Metrics:** Accuracy and loss tracked for both training and validation

## Files Description

- **`src/train.py`**: Main training script with command-line arguments
- **`src/model.py`**: Contains both CNN architectures
- **`src/data_setup.py`**: Custom dataset class and data loading utilities
- **`src/engine.py`**: Training, validation, and prediction functions
- **`src/utils.py`**: Helper functions for model saving and plotting
- **`inference.py`**: Script for making predictions on new images

## Requirements

- Python 3.7+
- PyTorch 1.12.0+
- torchvision 0.13.0+
- OpenCV 4.6.0+
- PIL (Pillow) 9.2.0+
- matplotlib 3.5.0+
- tqdm 4.64.0+
- numpy 1.21.0+

## GPU Support

The code automatically detects and uses CUDA if available:
- Training will use GPU if CUDA is available
- Falls back to CPU if no GPU is detected
- Model and data are automatically moved to the appropriate device

## Output Files

After training, the following files are generated:
- **Model file**: `models/{model_name}.pth` - Trained model weights
- **Metrics file**: `{model_name}-metrics.npz` - Training/validation metrics
- **Plots**: Accuracy and loss curves (displayed during training)

## Customization

### Using Your Own Dataset
1. Organize your images in the expected directory structure
2. Ensure images are in `.jpg` format
3. Update the directory paths in the training script if needed

### Modifying Model Architecture
- Edit `src/model.py` to add new architectures
- Update the model instantiation in `src/train.py`
- Adjust loss function and optimizer as needed

### Changing Training Parameters
- Modify default values in `src/train.py`
- Add new command-line arguments as needed
- Adjust data transforms in `src/data_setup.py`

## Troubleshooting

**Common Issues:**
1. **CUDA out of memory**: Reduce batch size
2. **File not found**: Check data directory structure
3. **Import errors**: Ensure all dependencies are installed
4. **Model loading errors**: Verify model file path and architecture match

## License

This project is for educational purposes. Please ensure you have appropriate licenses for any datasets used.
