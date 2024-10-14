# helpers/build_model.py

from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam

def build_model(input_shape=(300, 300, 3), num_classes=6):
    # Pre-trained InceptionV3 model without the top layers
    base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=input_shape)
    
    # Freeze the base model
    base_model.trainable = False
    
    # Add custom layers on top
    x = base_model.output
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    
    # Define the model
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # Compile the model
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

if __name__ == "__main__":
    model = build_model()
    model.summary()
    
    # Agar tumhare paas training data hai, toh yahan model ko train karo
    # Example:
    # from tensorflow.keras.preprocessing.image import ImageDataGenerator
    # train_datagen = ImageDataGenerator(rescale=1./255)
    # train_generator = train_datagen.flow_from_directory(
    #     'path_to_training_data',
    #     target_size=(300, 300),
    #     batch_size=32,
    #     class_mode='categorical'
    # )
    # model.fit(train_generator, epochs=10)
    
    # Model ko save karo
    model.save("model_correct.h5")
    print("Model successfully saved as 'model_correct.h5'")
