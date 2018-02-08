import time
import constants as const
import classification.neural_network as nn

start_time = time.time()

print('running neural network...')
# nn.run_NN(const.SIZE, const.PREPROCESSED_DATASET_DIR)
nn.run_CNN(const.SIZE, const.PREPROCESSED_DATASET_DIR)

end_time = time.time()
print('done in {:.2f}s'.format(end_time - start_time))
