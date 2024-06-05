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
