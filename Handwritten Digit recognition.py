import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from PIL import Image
import io
from IPython.display import display
import ipywidgets as widgets
import cv2

# Step 1: Load MNIST dataset
(X_train, y_train), (X_test, y_test) = datasets.mnist.load_data()

# Step 2: Normalize pixel values
X_train, X_test = X_train / 255.0, X_test / 255.0

# Step 3: Reshape to fit CNN input (28x28x1)
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

# Step 4: Build CNN Model
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    layers.MaxPooling2D((2,2)),
   
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
   
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Step 5: Compile Model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Step 6: Train Model
history = model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test))

# Step 7: Evaluate Model
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"✅ Test Accuracy: {test_acc:.4f}")

# Step 8: Plot Training and Validation Accuracy
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Step 9: Confusion Matrix
y_pred = np.argmax(model.predict(X_test), axis=1)
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))


# --- HANDWRITTEN DIGIT INPUT PART ---

# Helper function to preprocess user's drawing (28x28 grayscale)
def preprocess_user_image(img_data):
    # img_data is 280x280 pixels (canvas size)
    # Convert to grayscale, resize to 28x28, invert colors
    img = Image.open(io.BytesIO(img_data)).convert('L')
    img = img.resize((28, 28))
   
    img_array = np.array(img)
   
    # Invert colors: background white (255), digit black (0)
    img_array = 255 - img_array
   
    # Normalize
    img_array = img_array / 255.0
   
    # Reshape to match model input
    img_array = img_array.reshape(1, 28, 28, 1)
   
    return img_array

# Display canvas widget for user to draw digits
canvas = widgets.Output()
print("Draw a digit (0-9) in the box below, then click 'Predict'.")

# Using ipywidgets Drawing Widget from ipycanvas or any other
# However, Google Colab doesn't support ipycanvas or direct drawing widgets easily.
# So we use an alternative approach: upload an image of the digit.

upload = widgets.FileUpload(accept='image/*', multiple=False)
display(upload)

def predict_uploaded_image(change):
    if upload.value:
        # Get the uploaded image content
        for name, file_info in upload.value.items():
            img_bytes = file_info['content']
            img_preprocessed = preprocess_user_image(img_bytes)
           
            prediction = model.predict(img_preprocessed)
            predicted_digit = np.argmax(prediction)
           
            plt.imshow(img_preprocessed.reshape(28,28), cmap='gray')
            plt.title(f"Predicted digit: {predicted_digit}")
            plt.axis('off')
            plt.show()

upload.observe(predict_uploaded_image, names='value')
