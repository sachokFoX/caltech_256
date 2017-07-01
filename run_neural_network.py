import time
import constants as const
import classification.neural_network as nn

start_time = time.time()

print('running neural network...')
nn.run(const.size, const.preprocessed_dataset_dir)

end_time = time.time()
print('done in {:.2f}s'.format(end_time - start_time))
