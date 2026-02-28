from tensorflow.keras import backend as K

# Weight for [background, person] - higher weight for person to handle class imbalance
weights = [1, 3]


def class_weighted_pixelwise_crossentropy(target, output):
    output = K.clip(output, 1e-7, 1.0 - 1e-7)
    return -K.sum(target * weights * K.log(output), axis=-1)
