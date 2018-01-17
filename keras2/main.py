from models import vanillaCnn
from cifarTrainer import cifarTrainer
import tensorflow as tf


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

classes = [1, 2]
model = vanillaCnn.vanillaCnn([32, 32, 3], 10).model
trainer = cifarTrainer()
model = trainer.trainModel(model, classes, 256, 50)
trainer.evalModel(model, classes)
