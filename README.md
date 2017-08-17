# deeplearning4d

Deep learning library for Dart based on Tensorflow Playground Neural Net implementation.

## Example
```dart

import 'package:deeplearning4d/deeplearning4d.dart';
import 'dataset.dart';

List<List<Node>> network = null;
int iterations = 400;
List<Example2D> trainData = classifyTwoGaussData(500.0, 0.1);
List<Example2D> testData = classifyTwoGaussData( 50.0, 0.1);
double learningRate = 0.03;
double regularizationRate = 0.0;
double lossTrain = 0.0;
double lossTest = 0.0;
int batchSize = 30;


void main() {
  network = buildNetwork([2, 1, 1], Activations.RELU, Activations.TANH, RegularizationFunctions.L1, ["x", "y"], initZero: false);
  lossTrain = getLoss(network, trainData);
  lossTest = getLoss(network, testData);
  for (int i = 0; i < iterations; i ++) {
    oneStep();
  }
}

oneStep() {
  trainData.asMap().forEach((i, example2D) {
    List<double> input = constructInput(example2D.x, example2D.y);
    forwardProp(network, input);
    backProp(network, example2D.label, Errors.SQUARE);
    if ((i + 1) % batchSize == 0) {
      updateWeights(network, learningRate, regularizationRate);
    }
  });
// Compute the loss.
  lossTrain = getLoss(network, trainData);
  lossTest = getLoss(network, testData);
  print(lossTest.toString() + ',       ' + lossTrain.toString());
}


List<double> constructInput(double x, double y) {
  List<double> input = <double>[];
  input.add(x);
  input.add(y);
  return input;
}

double getLoss(List<List<Node>> network, List<Example2D> dataPoints) {
  double loss = 0.0;
  for (int i = 0; i < dataPoints.length; i++) {
    Example2D dataPoint = dataPoints[i];
    List<double> input = constructInput(dataPoint.x, dataPoint.y);
    double output = forwardProp(network, input);
    loss += Errors.SQUARE.error(output, dataPoint.label);
  }
  return loss / dataPoints.length;
}
```
