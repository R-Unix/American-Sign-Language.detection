import tensorflow as tf 

model = tf.keras.models.load_model("/Users/ishan/Downloads/model.chkpt.data-00001-of-00002.pb")
var = tf.lite.TFLiteConverter.from_keras_model(model)
conv = var.convert()
with open('ASL.tflite','wb') as f:
    f.write(conv)


