"""
model.py — CIFAR-10 CNN model (load cached or train from scratch)
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

MODEL_PATH = "cifar10_cnn.keras"


def build_model():
    """
    Build an improved CNN for CIFAR-10.
    Expected accuracy: ~82–88% depending on training duration and hardware.
    """

    inputs = tf.keras.Input(shape=(32, 32, 3))

    # Data augmentation
    x = layers.RandomFlip("horizontal")(inputs)
    x = layers.RandomRotation(0.1)(x)
    x = layers.RandomZoom(0.1)(x)

    # Block 1
    x = layers.Conv2D(32, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(32, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)

    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)

    # Block 2
    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)

    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)

    # Block 3
    x = layers.Conv2D(128, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(128, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)

    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)

    # Better classifier head
    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dense(256, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)

    outputs = layers.Dense(10, activation="softmax")(x)

    model = models.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def load_or_train_model(epochs=30, retrain=False):
    """
    Load saved model if available.
    Otherwise train from scratch.
    """

    if os.path.exists(MODEL_PATH) and not retrain:
        print(f"[INFO] Loading saved model from {MODEL_PATH}")
        return tf.keras.models.load_model(MODEL_PATH)

    print("[INFO] Training model from scratch...")

    # Load CIFAR-10 directly from TensorFlow
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    # Normalize
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    model = build_model()

    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=7,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            MODEL_PATH,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
    ]

    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_test, y_test),
        epochs=epochs,
        batch_size=64,
        callbacks=callbacks,
        verbose=1,
    )

    loss, acc = model.evaluate(x_test, y_test, verbose=0)

    print("\n==============================")
    print(f"Test Accuracy : {acc * 100:.2f}%")
    print(f"Test Loss     : {loss:.4f}")
    print("==============================\n")

    model.save(MODEL_PATH)

    return model


def predict_image(model, image):
    """
    Predict a single CIFAR-sized image.

    image shape:
    (32,32,3)
    """

    image = image.astype("float32") / 255.0
    image = np.expand_dims(image, axis=0)

    prediction = model.predict(image, verbose=0)

    class_idx = np.argmax(prediction)

    return CLASSES[class_idx], prediction[0][class_idx]


if __name__ == "__main__":

    model = load_or_train_model(
        epochs=30,
        retrain=False
    )

    # Final evaluation
    (_, _), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    x_test = x_test.astype("float32") / 255.0

    loss, acc = model.evaluate(
        x_test,
        y_test,
        verbose=0
    )

    print(f"Final Accuracy: {acc * 100:.2f}%")
