from keras import backend as K
from transformer import TranformerEncoder


input_tensor1 = K.variable(np.array([[0.1, 0.2, 0.3, 0.4], [0.9, 0.5, 0.6, 0.7], [0.1, 0.2, 0.3, 0.4], [0.1, 0.1, 0.1, 0.1]]), dtype='float32')
opt = TransformerEncoder(4, 2, 4)(input_tensor1)

print("input tensor")
print(K.eval(input_tensor1))
print("output tensor")
print(K.eval(opt))
