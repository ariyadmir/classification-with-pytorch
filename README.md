# German Traffic Sign Recognition Benchmark with PyTorch

This project implements a convolutional neural network (CNN) for the German Traffic Sign Recognition Benchmark using PyTorch. The CNN is trained to recognize traffic signs from images with a specific focus on resizing images to 30x30x3, which is a standard size for image classification tasks.

## Dataset
The dataset is organized with different classes representing various traffic signs. The project includes code for filtering and splitting the dataset into training and validation sets.

## Preprocessing
Images are resized to the specified dimensions, and a train-validation split is performed. The data loading process is enhanced with additional functionalities like filtering CSV files and creating a pickle file for efficient data loading.

## Neural Network Architecture
The CNN architecture consists of convolutional layers, max-pooling layers, batch normalization, and a final linear layer for classification. The network is designed to predict the class of the traffic sign from the resized images.

## Training
The model is trained using hyperparameters such as learning rate, batch size, contrast, and rotation. Training progress and validation accuracy are monitored during the training process. The trained model is saved for later use.

## Test Accuracy
The project evaluates the trained model on a test dataset, providing insights into the performance of the model on unseen data. Test accuracy is calculated and reported.

## Dependencies
- Python 3.x
- PyTorch
- Torchvision
- NumPy
- Matplotlib
- Pillow
- scikit-learn

## Usage
1. Clone the repository: `git clone https://github.com/ariyadmir/classification-with-pytorch.git`
2. Install the required dependencies: `pip install -r requirements.txt`
3. Run the training script: `python main.py`

## Note
- The complete code, including hyperparameter tuning, results, confusion matrix, and classification report, can be found in the Jupyter Notebook file. The notebook provides a detailed walkthrough of the training process, model evaluation, and analysis of results.
