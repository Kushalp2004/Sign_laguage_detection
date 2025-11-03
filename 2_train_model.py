import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import config  # Import config file

# --- 1. Load Data and Create Labels ---
label_map = {label: num for num, label in enumerate(config.ACTIONS)}

sequences, labels = [], []
print("Loading data...")
for action in config.ACTIONS:
    print(f"Loading action: {action}")
    for sequence in range(config.NO_SEQUENCES):
        window = [] # To hold 30 frames
        for frame_num in range(config.SEQUENCE_LENGTH):
            res = np.load(os.path.join(config.DATA_PATH, action, str(sequence), f"{frame_num}.npy"))
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])

print(f"\nTotal sequences loaded: {len(sequences)}")

X = np.array(sequences)
y = to_categorical(labels).astype(int)

print(f"Shape of X (input data): {X.shape}")
print(f"Shape of y (labels): {y.shape}")

# --- 2. Split Data ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, stratify=y, random_state=42)

# --- 3. Build LSTM Model (Dynamically) ---
model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=config.INPUT_SHAPE))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
# This layer is now DYNAMIC. It creates as many neurons as you have actions.
model.add(Dense(config.ACTIONS.shape[0], activation='softmax'))

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# --- 4. Train Model ---
print("\n--- Starting Model Training ---")

# Stop training if validation loss doesn't improve for 15 epochs
early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
# Reduce learning rate if training plateaus
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)

history = model.fit(X_train, y_train, 
                    epochs=150, # More epochs for a larger dataset
                    batch_size=32,
                    validation_data=(X_test, y_test),
                    callbacks=[early_stopping, reduce_lr])

print(f"\nModel training finished.")

# --- 5. Evaluate and Save Model ---
print("\nEvaluating model on test data...")
val_loss, val_acc = model.evaluate(X_test, y_test)
print(f"Test Loss: {val_loss:.4f}")
print(f"Test Accuracy: {val_acc*100:.2f}%")

model.save(config.MODEL_NAME)
print(f"\nModel saved as '{config.MODEL_NAME}'")
