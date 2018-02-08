import os
import csv
import time
import collections
import constants as cons
import matplotlib.pyplot as plt

start_time = time.time()

print('reading results...')

results = {}
results = collections.defaultdict(lambda: 0, results)

with open(os.path.join('../output', 'nn_results.csv'), newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for i, row in enumerate(reader):
        if i is not 0:
            results[row[1]] += 1

classes = [None] * len(results)
samples = [None] * len(results)

for i, image_class in enumerate(sorted(results)):
    classes[i] = int(image_class)
    samples[i] = results[image_class]

plt.bar(classes, samples)
plt.show()

end_time = time.time()
print('done in {:.2f}s'.format(end_time - start_time))