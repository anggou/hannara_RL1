import tensorflow as tf

action = [534, 433
    , 2, 3, 4]

# for i in action:
code = tf.one_hot([action], 5)

print(code)
