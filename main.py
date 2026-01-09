import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

RESNET_V2_50_URL = "https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/5"

def build_model(num_classes):
    # Input layer
    inputs = tf.keras.Input(shape=(224, 224, 3))

    # Hub feature extractor (feature_vector, not classification)
    base_model = hub.KerasLayer(
        "https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/5",
        trainable=False
    )

    # Apply feature extractor to inputs
    x = base_model(inputs)

    # Dense classification layer
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    # Build the functional model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    # Compile
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def plot_history(history):
    plt.figure(figsize=(12, 4))

    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training vs Validation Accuracy')

    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training vs Validation Loss')

    plt.show()

def main():
   
    dataset_path = "C:\BCA_Project_PlantVillage\PlantVillageDataset\PlantVillage\Subset_Tomato_Potato\PlantVillage"

    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

    train_data = datagen.flow_from_directory(
        dataset_path,
        target_size=(224, 224),
        batch_size=32,
        subset='training'
    )

    val_data = datagen.flow_from_directory(
        dataset_path,
        target_size=(224, 224),
        batch_size=32,
        subset='validation'
    )

    num_classes = train_data.num_classes
    model = build_model(num_classes)

    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=5
    )

    plot_history(history)

    # Save model
    model.save("plant_disease_model.h5")
    print("Model saved as plant_disease_model.h5")

if __name__ == "__main__":
    main()
