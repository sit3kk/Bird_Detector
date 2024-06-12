# 🐦 Project: Bird Detection in Images

The goal of the project was to create a model that checks whether a bird is present in a photo or not. The solution was applied in the project [Bird Species Recognition](https://github.com/bubiasz/team-project).

## 📊 1. Data Set

The dataset used is a mix of CIFAR-10, COCO, Birdsnap, and CUB-200-2011, resulting in a dataset containing about 150,000 images in two categories: bird and non-bird.

The biggest problem in finding the right dataset was finding appropriate datasets that are both diverse and do not contain birds. CIFAR-10 and COCO were chosen because they are very large datasets divided into categories.

## 🚀 2. Data Loader

To speed up model training, a data loader was created to split and shuffle the images into `train`, `val`, and `test` folders. After this operation, the images are rescaled to the appropriate dimensions.

## 🧠 3. Model

### MobileNetV2 (CNN, Convolutional Neural Network)

Designed for mobile devices.

**Pros:**
- Computational efficiency
- Memory efficiency
- Versatility
- Scalability

**Cons:**
- Lower accuracy than larger models
- Difficulties with tuning

#### Preparation process:

```python
def build_model(num_classes):
    # Load the pre-trained MobileNetV2 model without the top layers (no classification layers)
    base_model = MobileNetV2(
        input_shape=(224, 224, 3),  # Define the input shape of the images (224x224x3)
        include_top=False,          # Exclude the top classification layer of the model
        weights="imagenet"          # Use pre-trained weights from the ImageNet dataset
    )

    # Add custom layers on top of the base model
    x = base_model.output  # Output from the pre-trained model
    x = GlobalAveragePooling2D()(x)  # Add a global average pooling layer to reduce data dimensionality
    x = Dense(128, activation="relu")(x)  # Add a dense layer with 128 neurons and ReLU activation
    predictions = Dense(num_classes, activation="softmax")(x)  # Add the output layer with softmax activation (num_classes is the number of classes)

    # Create the final model with the original input and new output
    model = Model(inputs=base_model.input, outputs=predictions)
    return model

# Define the number of classes (in this case 2: bird and non-bird)
num_classes = 2 

# Build the model
model = build_model(num_classes)

# Display the model summary to see its architecture
model.summary()
```
During training, batch_size was set to 32 due to low computing power (at higher values ​​the program crashed).

### 🧩 Result

In the model, the validation accuracy started to decrease from around the 6th epoch. Finally, after the entire training process, the most favorable version was selected, which achieved an accuracy of 74% on the test set, which was an unsatisfactory result.

**Errors in this iteration:**
- Too small dataset - only 10% of the entire dataset was used to save training time.
- Training on CPU - lack of installed CUDA and cuDNN caused the model to train on the processor, significantly prolonging the training time.
- Lack of regularization - the drop in performance might have been caused by overfitting.
- Imbalanced data - there were several tens of percent more bird photos, which might have caused the model to prefer this class more.

## 🔄 Change of Approach

After obtaining results far below expectations, we decided to change the approach.

### ResNet (Residual Neural Network)
A family of deep neural networks

**Pros:**
- High accuracy
- Efficient training
- Easier adjustment

**Cons:**
- Computational complexity
- Higher risk of overfitting
- Architectural complexity

We chose the pre-trained ResNet50 model (a variant consisting of 50 layers) as a compromise between complexity and efficiency.

**Fixed errors compared to the previous solution:**
- Training the model on GPU to speed up the process.
- Increasing the dataset size to about 50%.
- Balancing the class sizes between bird and nonbird.

### 💡 Solution Description

The EarlyStopping class is used to monitor the model training process and automatically stop training if the indicator stops improving. If the validation loss improves, the model is saved, and the best score and minimum validation loss are updated. After all 10 epochs, the final model is saved in `saved_models`.

```python
class EarlyStopping:
    def __init__(self, patience=5, verbose=False, delta=0):
        self.patience = patience          # Number of epochs without improvement after which training will be stopped
        self.verbose = verbose            # Flag indicating whether to print messages
        self.counter = 0                  # Counter of epochs without improvement
        self.best_score = None            # Best score (lowest validation loss)
        self.early_stop = False           # Flag indicating whether to stop training
        self.val_loss_min = np.Inf        # Minimum validation loss (initialized as infinity)
        self.delta = delta                # Minimum difference required to consider improvement

    def __call__(self, val_loss, model, path):
        score = -val_loss                 # Negative value of validation loss to turn minimization into maximization

        if self.best_score is None:       # First call
            self.best_score = score       # Set the initial best score
            self.save_checkpoint(val_loss, model, path)  # Save the model
        elif score < self.best_score + self.delta:  # No improvement
            self.counter += 1             # Increase the counter of epochs without improvement
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:  # If the counter exceeds the patience value
                self.early_stop = True    # Set the early stopping flag
        else:                             # Improvement
            self.best_score = score       # Update the best score
            self.save_checkpoint(val_loss, model, path)  # Save the model
            self.counter = 0              # Reset the counter

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ..."
            )
        torch.save(model.state_dict(), path)  # Save the model state to a file
        self.val_loss_min = val_loss          # Update the minimum validation loss
```

### 🔧 Preparing the ResNet50 Model

```python
# Loading a pre-trained ResNet50 model
resnet50 = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

# Freezing all parameters in the model
for param in resnet50.parameters():
    param.requires_grad = False

# Modifying the last layer to adapt it to the number of classes
num_features = resnet50.fc.in_features
resnet50.fc = nn.Linear(num_features, 2)
```

## ❄️ Freezing Weights
Freezing the weights of a pre-trained model halts the process of updating those weights during training, reducing the risk of overfitting and speeding up training. We focus solely on training new or modified layers of the model.

The original final layer of ResNet50 is tailored for classifying into 1000 ImageNet classes. However, in our task, we only have two classes (bird and non-bird), so we needed to adapt the final layer to this number of classes. By replacing it with a new layer with two neurons, the model can generate appropriate predictions for our specific task.

## 🏋️‍♂️ Model Training
```python
def train_model(
    model,                     # Model to be trained
    dataloaders,               # Training and validation data loaders as a dictionary
    criterion,                 # Loss function
    optimizer,                 # Optimizer for updating model weights
    num_epochs=25,             # Number of training epochs (default is 25)
    patience=5,                # Number of epochs without improvement before stopping (early stopping)
    checkpoint_path="checkpoint.pth",  # Path to save the best model
):
    early_stopping = EarlyStopping(patience=patience, verbose=True)  # Initialize early stopping
    history = {"train_loss": [], "val_loss": [], "val_acc": []}       # Training history

    print("Starting training...")  # Start training

    for epoch in range(num_epochs):  # Loop through all epochs
        model.train()  # Set the model to training mode
        running_loss = 0.0  # Initialize loss for the current epoch
        running_corrects = 0  # Initialize number of correct predictions for the current epoch

        print(f"Epoch {epoch+1}/{num_epochs}")  # Current epoch information

        for inputs, labels in dataloaders["train"]:  # Loop through all training data batches
            inputs = inputs.to(device)  # Move input data to device (CPU/GPU)
            labels = labels.to(device)  # Move labels to device (CPU/GPU)

            optimizer.zero_grad()  # Zero the gradients
            outputs = model(inputs)  # Forward pass
            loss = criterion(outputs, labels)  # Calculate loss
            loss.backward()  # Backward pass
            optimizer.step()  # Update model weights

            _, preds = torch.max(outputs, 1)  # Get predictions
            running_loss += loss.item() * inputs.size(0)  # Update running loss
            running_corrects += torch.sum(preds == labels.data)  # Update number of correct predictions

        epoch_loss = running_loss / len(dataloaders["train"].dataset)  # Calculate average loss for the epoch
        epoch_acc = running_corrects.double() / len(dataloaders["train"].dataset)  # Calculate accuracy for the epoch

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}")  # Epoch results
        history["train_loss"].append(epoch_loss)  # Save training loss to history

        model.eval()  # Set the model to evaluation mode
        val_running_loss = 0.0  # Initialize validation loss
        val_running_corrects = 0  # Initialize number of correct validation predictions

        with torch.no_grad():  # Disable gradient computation during validation
            for inputs, labels in dataloaders["val"]:  # Loop through all validation data batches
                inputs = inputs.to(device)  # Move input data to device (CPU/GPU)
                labels = labels.to(device)  # Move labels to device (CPU/GPU)

                outputs = model(inputs)  # Forward pass
                loss = criterion(outputs, labels)  # Calculate loss

                _, preds = torch.max(outputs, 1)  # Get predictions
                val_running_loss += loss.item() * inputs.size(0)  # Update running validation loss
                val_running_corrects += torch.sum(preds == labels.data)  # Update number of correct validation predictions

        val_epoch_loss = val_running_loss / len(dataloaders["val"].dataset)  # Calculate average validation loss
        val_epoch_acc = val_running_corrects.double() / len(dataloaders["val"].dataset)  # Calculate validation accuracy

        print(f"Validation Loss: {val_epoch_loss:.4f}, Validation Acc: {val_epoch_acc:.4f}")  # Validation results
        history["val_loss"].append(val_epoch_loss)  # Save validation loss to history
        history["val_acc"].append(val_epoch_acc.item())  # Save validation accuracy to history

        # Early stopping and checkpoint saving
        early_stopping(val_epoch_loss, model, checkpoint_path)  # Check early stopping condition
        if early_stopping.early_stop:
            print("Early stopping")  # Early stopping notification
            break

    # Load the best model weights
    model.load_state_dict(torch.load(checkpoint_path))
    return model, history  # Return the trained model and training history

# Assign DataLoader to variable 'dataloaders'
dataloaders = {"train": train_loader, "val": val_loader}

```

#### In order to observe training history, the ability to save results and then visualize them using a chart has been added.

![image](https://github.com/sit3kk/Bird_Detector/assets/69002597/906e7ab9-f10a-4bb9-b2c2-07578b8a9014)

## 📈 Training Progress
From the very beginning, the effectiveness of training was increasing (excluding one drop), while the training time was significantly accelerated (probably due to the use of GPU). Ultimately, a test set accuracy of 99% was achieved, which exceeded expectations.

## 🏆 Final Decision

The use of the ResNet architecture significantly outperformed the results of MobileNetV2 and concluded the iteration in search of the best solution.

### Why Did This Happen?

- Methodological errors in the first model (unbalanced classes, poor model adjustment).
- The deeper architecture of ResNet, thanks to the use of residual blocks, which better capture complex data patterns (e.g., comparing a bird image to a bat image).
- ResNet uses more parameters, allowing for modeling more complex functions.

The model was intended to work on the server side, so the increased computational power was not as critical as in the case of mobile applications.

## 🛠️ Model Usage

In the `models_predictions.ipynb` file, you can see examples of using the models on images found on the internet and their execution times, confirming the effectiveness of the final solution.

In the `utils` folder, we have defined the files `mn_utils.py` and `rn_utils.py`, from which we can import functions to predict images (which take a path as input).

## 📦 Installation

To run the project, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/sit3kk/Bird_Detector
    ```

2. Activate the virtual environment:
    ```bash
    python -m venv venv
    source ./venv/bin/activate
    ```

3. Install the required libraries:
    ```bash
    pip install -r requirements.txt
    ```
