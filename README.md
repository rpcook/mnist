# mnist
Ground-up neural-network to classify MNIST handwritten digits, not bleeding edge by any definition

## Dependencies
- Python 3.x
- Python libraries: numpy
- MNIST database from http://yann.lecun.com/exdb/mnist/ (plus uncompress) in local folder to code

## .nn file format
Describes a fully connected neural network, N layers (1<N<256). First layer has no weights and biases, input from external only.

Binary data, little endian

| type | length | description |
| --- | --- | --- |
| unsigned char | 1 | number of layers (N) |
| unsigned short | 2 | number of neurons in layer 0 |
| unsigned short | 2 | number of neurons in layer 1 |
| ... | ... | ... |
| unsigned short | 2 | number of neurons in layer N-1 |
| float | 4 | weight for neuron 0 in layer 1 to neuron 0 in layer 0 |
| float | 4 | weight for neuron 0 in layer 1 to neuron 1 in layer 0 |
| ... | ... | ... |
| float | 4 | weight for neuron 0 in layer 1 to last neuron in layer 0 |
| float | 4 | weight for neuron 1 in layer 1 to neuron 0 in layer 0 |
| float | 4 | weight for neuron 1 in layer 1 to neuron 1 in layer 0 |
| ... | ... | ... |
| float | 4 | weight for neuron 1 in layer 1 to last neuron in layer 0 |
| ... | ... | ... |
| float | 4 | weight for last neuron in layer 1 to neuron 0 in layer 0 |
| float | 4 | weight for last neuron in layer 1 to neuron 1 in layer 0 |
| ... | ... | ... |
| float | 4 | weight for last neuron in layer 1 to last neuron in layer 0 |
| float | 4 | bias for neuron 0 in layer 1 |
| float | 4 | bias for neuron 1 in layer 1 |
| ... | ... | ... |
| float | 4 | bias for last neuron in layer 1 |
| float | 4\*x | above structure (weights, then biases) follows for each subsequent layer |

## Inspiration / References
- 3Blue1Brown's excellent series on [Neural Networks](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi) (my original motivation to start this project)
- Standford University's Andrew Ng's full course on [Machine Learning](https://www.youtube.com/playlist?list=PLLssT5z_DsK-h9vYZkQkYNWcItqhlRJLN)
- Cambridge University's Mike Gordon's [write up](https://www.cl.cam.ac.uk/archive/mjcg/plans/Backpropagation.pdf) of going through a similar process
- Johannes Langelaar's exemplar [Matlab code](https://uk.mathworks.com/matlabcentral/fileexchange/73010-mnist-neural-network-training-and-testing) on the same topic, particularly helpful around the vector notation of the accumulators in the backpropagation algorithm
