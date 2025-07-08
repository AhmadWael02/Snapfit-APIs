import tensorflow as tf
import numpy as np
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.preprocessing import image as keras_image
import threading
import os

# Path to the VSE checkpoint (update as needed)
VSE_CHECKPOINT_PATH = r"D:\snapfit_v1\snapfit_v1\assets\models\model.ckpt-34865"
INCEPTION_WEIGHTS_PATH = r"C:\Users\lenovo\Downloads\inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5"

# Thread-safe singleton pattern for model loading
_inception_model = None
_vse_weights = None
_model_lock = threading.Lock()
_vse_lock = threading.Lock()


def get_inception_model():
    global _inception_model
    with _model_lock:
        if _inception_model is None:
            if os.path.exists(INCEPTION_WEIGHTS_PATH):
                _inception_model = InceptionV3(
                    weights=INCEPTION_WEIGHTS_PATH,
                    include_top=False,
                    pooling='avg',
                    input_shape=(299, 299, 3)
                )
            else:
                _inception_model = InceptionV3(
                    weights='imagenet',
                    include_top=False,
                    pooling='avg',
                    input_shape=(299, 299, 3)
                )
        return _inception_model

def get_vse_weights():
    global _vse_weights
    with _model_lock:
        if _vse_weights is None:
            try:
                tf.compat.v1.disable_eager_execution()
                tf.compat.v1.reset_default_graph()
                with tf.compat.v1.Session() as sess:
                    with tf.compat.v1.variable_scope('image_embedding'):
                        vse_weights_var = tf.compat.v1.get_variable('weights', shape=[2048, 512])
                    saver = tf.compat.v1.train.Saver([vse_weights_var])
                    saver.restore(sess, VSE_CHECKPOINT_PATH)
                    _vse_weights = sess.run(vse_weights_var)
            except Exception as e:
                raise RuntimeError(f"Failed to load VSE weights: {e}")
        return _vse_weights

def get_vse_embedding(image_path: str) -> list:
    """
    Given an image path, extract the InceptionV3 embedding, transform it using VSE weights, normalize, and return as a list.
    """
    try:
        with _vse_lock:
            # Step 1: InceptionV3 embedding (same as get_inception_embedding)
            model = get_inception_model()
            img = keras_image.load_img(image_path, target_size=(299, 299))
            img_array = keras_image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)
            features = model.predict(img_array, verbose=0)
            features = features.flatten()  # (2048,)

            # Step 2: VSE transformation (separate session)
            # Get VSE weights using the existing function
            vse_weights = get_vse_weights()
            vse_features = np.dot(features, vse_weights)  # (512,)

            # Step 3: Normalize
            vse_features = vse_features / (np.linalg.norm(vse_features) + 1e-8)
            return vse_features.astype(np.float32).tolist()
    except Exception as e:
        raise RuntimeError(f"Failed to extract VSE embedding: {e}")

def get_inception_embedding(image_path: str) -> list:
    """
    Given an image path, load the image and return the InceptionV3 embedding as a list of floats (for DB storage).
    """
    try:
        model = get_inception_model()
        img = keras_image.load_img(image_path, target_size=(299, 299))
        img_array = keras_image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        features = model.predict(img_array, verbose=0)
        return features.flatten().astype(np.float32).tolist()  # Shape: (2048,)
    except Exception as e:
        raise RuntimeError(f"Failed to extract InceptionV3 embedding: {e}")