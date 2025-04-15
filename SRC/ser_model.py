
import os
import numpy as np
import librosa
import librosa.display
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import glob

# 📂 Load dataset path
DATASET_PATH = r"C:\Users\dhivy\Downloads\archive"

# 🏷️ Labels and Features Storage
labels = []
features = []

# 🎤 Extract Mel Spectrogram Features
def extract_features(file_path, max_pad_len=128):
    try:
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sample_rate)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        # Normalize to [0,1]
        mel_spec_db = (mel_spec_db - np.min(mel_spec_db)) / (np.max(mel_spec_db) - np.min(mel_spec_db))

        # 🛠️ Pad/Trim Mel Spectrograms to fixed size
        if mel_spec_db.shape[1] < max_pad_len:
            pad_width = max_pad_len - mel_spec_db.shape[1]
            mel_spec_db = np.pad(mel_spec_db, ((0, 0), (0, pad_width)), mode='constant')
        else:
            mel_spec_db = mel_spec_db[:, :max_pad_len]

        return mel_spec_db
    except Exception as e:
        print(f"❌ Error loading {file_path}: {e}")
        return None

# 📥 Load dataset
for file in glob.glob(os.path.join(DATASET_PATH, "**", "*.wav"), recursive=True):
    try:
        # Extract label from filename (RAVDESS format: 03-01-XX-XX-XX-XX-XX.wav)
        file_name = os.path.basename(file)
        parts = file_name.split("-")

        if len(parts) >= 3:  # Ensure correct format
            label = int(parts[2]) - 1  # Convert to integer, adjusting for zero-based indexing
            labels.append(label)

            # Extract features
            feature = extract_features(file)
            if feature is not None:
                features.append(feature)
    except Exception as e:
        print(f"Skipping {file}: {e}")

# 🔄 Convert lists to NumPy arrays
X = np.array(features)
y = np.array(labels)

# 🛑 Check if dataset is loaded correctly
if X.shape[0] == 0 or y.shape[0] == 0:
    print("❌ Error: Dataset is empty. Check dataset path and file structure.")
    exit()

print(f"✅ Dataset Loaded: X shape: {X.shape}, y shape: {y.shape}")

# 🎭 Encode labels
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# 🧪 Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# 🔄 Reshape for CNN input
X_train = X_train[..., np.newaxis]  # Add channel dimension
X_test = X_test[..., np.newaxis]

# 🏗️ Build the CNN Model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(X_train.shape[1], X_train.shape[2], 1)),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.3),

    Flatten(),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),

    Dense(len(set(y_encoded)), activation='softmax')  # Output layer
])

# ⚙️ Compile the Model
model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 🎓 Train the Model
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=30, batch_size=32)

# 💾 Save the Model
model.save("ser_model.h5")
print("✅ Model saved as 'ser_model.h5'")

# 📊 Plot Accuracy and Loss
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title("Model Accuracy")

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title("Model Loss")

plt.show()
