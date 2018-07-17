import tensorflow as tf

def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    """
    Freezes the state of a session into a pruned computation graph.

    Creates a new computation graph where variable nodes are replaced by
    constants taking their current value in the session. The new graph will be
    pruned so subgraphs that are not necessary to compute the requested
    outputs are removed.
    @param session The TensorFlow session to be frozen.
    @param keep_var_names A list of variable names that should not be frozen,
                          or None to freeze all the variables in the graph.
    @param output_names Names of the relevant graph outputs.
    @param clear_devices Remove the device directives from the graph for better portability.
    @return The frozen graph definition.
    """
    from tensorflow.python.framework.graph_util import convert_variables_to_constants
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = convert_variables_to_constants(session, input_graph_def,
                                                      output_names, freeze_var_names)
        return frozen_graph


mnist = tf.keras.datasets.mnist
tf.keras.backend.set_learning_phase(0)

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train = x_train.reshape((-1, 28*28))
x_test = x_test.reshape((-1, 28*28))

model = tf.keras.models.Sequential()
#model.add(tf.keras.layers.Reshape((784,), input_shape=(28, 28, )))
model.add(tf.keras.layers.Dense(512, activation=tf.nn.relu))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax, name="output_node"))


model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print(x_test.shape)
print(x_train.shape)
model.fit(x_train, y_train, epochs=1)
model.evaluate(x_test, y_test)


from tensorflow.contrib.keras import backend as K

# Create, compile and train model...

frozen_graph = freeze_session(K.get_session(),
    output_names=[out.op.name for out in model.outputs])
tf.train.write_graph(frozen_graph, "some_directory", "my_model_text.pb", as_text=True)
tf.train.write_graph(K.get_session().graph_def, "some_directory", "my_model3.pb", as_text=True)

saver = tf.train.Saver()
tf.train.get_or_create_global_step()
sess = tf.keras.backend.get_session()
tf.train.write_graph(sess.graph_def, 'tmp', 'model.pbtxt')
save_path = saver.save(sess, 'tmp/model.cpkt')



# import cv2
# import numpy as np
#
# image = x_test[1, :, :]#np.zeros((28, 28), dtype=np.float32)
# print(image.shape)
# blob = cv2.dnn.blobFromImage(image)
#
# print(blob.shape)
# #print(blob)
#
# print("[INFO] loading model...")
# net = cv2.dnn.readNetFromTensorflow("some_directory/my_model.pb")
#
# print('set input')
# net.setInput(blob)
# print('forward')
# print(net.empty())
# #print(net.getLayersCount())
# preds = net.forward()
# print(preds)
# print('finished')