from tensorflow.keras.applications import VGG19
from tensorflow.keras_vggface import utils

def load_model():
    # Load the VGGFace model with the ResNet50 architecture
    model = vGG19(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')
    return model
