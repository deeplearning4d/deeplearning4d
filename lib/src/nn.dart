/* Copyright 2017. Marat Gubaidullin. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
=============================================================================*/

import 'dart:core';
import 'dart:math';

/**
 * Builds a neural network.
 *
 * @param networkShape The shape of the network. E.g. [1, 2, 3, 1] means
 *   the network will have one input node, 2 nodes in first hidden layer,
 *   3 nodes in second hidden layer and 1 output node.
 * @param activation The activation function of every hidden node.
 * @param outputActivation The activation function for the output nodes.
 * @param regularization The regularization function that computes a penalty
 *     for a given weight (parameter) in the network. If null, there will be
 *     no regularization.
 * @param inputIds List of ids for the input nodes.
 */
List<List<Node>> buildNetwork(List<int> networkShape, ActivationFunction activation,
    ActivationFunction outputActivation, RegularizationFunction regularization,
    List<String> inputIds, {bool  initZero : true}) {
  int numLayers = networkShape.length;
  int id = 1;
  /** List of layers, with each layer being a list of nodes. */
  List<List<Node>> network = new List<List<Node>>();
  for (int layerIdx = 0; layerIdx < numLayers; layerIdx++) {
    bool isOutputLayer = layerIdx == numLayers - 1;
    bool isInputLayer = layerIdx == 0;
    List<Node> currentLayer = <Node>[];
    network.add(currentLayer);
    int numNodes = networkShape[layerIdx];
    for (int i = 0; i < numNodes; i++) {
      String nodeId = id.toString();
      if (isInputLayer) {
        nodeId = inputIds[i];
      } else {
        id++;
      }
      Node node = new Node(nodeId, isOutputLayer ? outputActivation : activation, initZero: initZero);
      currentLayer.add(node);
      if (layerIdx >= 1) {
        // Add links from nodes in the previous layer to this node.
        for (int j = 0; j < network[layerIdx - 1].length; j++) {
          Node prevNode = network[layerIdx - 1][j];
          Link link = new Link(prevNode, node, regularization, initZero: initZero);
          prevNode.outputs.add(link);
          node.inputLinks.add(link);
        }
      }
    }
  }
  return network;
}

/**
 * Runs a forward propagation of the provided input through the provided
 * network. This method modifies the internal state of the network - the
 * total input and output of each node in the network.
 *
 * @param network The neural network.
 * @param inputs The input array. Its length should match the number of input
 *     nodes in the network.
 * @return The final output of the network.
 */
double forwardProp(List<List<Node>> network, List<double> inputs) {
  List<Node> inputLayer = network[0];
  if (inputs.length != inputLayer.length) {
    throw new StateError("The number of inputs must match the number of nodes in  the input layer");
  }
  // Update the input layer.
  for (int i = 0; i < inputLayer.length; i++) {
    Node node = inputLayer[i];
    node.output = inputs[i];
  }
  for (int layerIdx = 1; layerIdx < network.length; layerIdx++) {
    List<Node> currentLayer = network[layerIdx];
    // Update all the nodes in this layer.
    for (int i = 0; i < currentLayer.length; i++) {
      Node node = currentLayer[i];
      node.updateOutput();
    }
  }
  return network[network.length - 1][0].output;
}

/**
 * Runs a backward propagation using the provided target and the
 * computed output of the previous call to forward propagation.
 * This method modifies the internal state of the network - the error
 * derivatives with respect to each node, and each weight
 * in the network.
 */
backProp(List<List<Node>> network, double target, ErrorFunction errorFunc) {
  // The output node is a special case. We use the user-defined error
  // function for the derivative.
  Node outputNode = network[network.length - 1][0];
  outputNode.outputDer = errorFunc.der(outputNode.output, target);

  // Go through the layers backwards.
  for (int layerIdx = network.length - 1; layerIdx >= 1; layerIdx--) {
    List<Node> currentLayer = network[layerIdx];
    // Compute the error derivative of each node with respect to:
    // 1) its total input
    // 2) each of its input weights.
    for (int i = 0; i < currentLayer.length; i++) {
      Node node = currentLayer[i];
      node.inputDer = node.outputDer * node.activation.der(node.totalInput);
      node.accInputDer += node.inputDer;
      node.numAccumulatedDers++;
    }

    // Error derivative with respect to each weight coming into the node.
    for (int i = 0; i < currentLayer.length; i++) {
      Node node = currentLayer[i];
      for (int j = 0; j < node.inputLinks.length; j++) {
        Link link = node.inputLinks[j];
        if (link.isDead) {
          continue;
        }
        link.errorDer = node.inputDer * link.source.output;
        link.accErrorDer += link.errorDer;
        link.numAccumulatedDers++;
      }
    }
    if (layerIdx == 1) {
      continue;
    }
    List<Node> prevLayer = network[layerIdx - 1];
    for (int i = 0; i < prevLayer.length; i++) {
      Node node = prevLayer[i];
      // Compute the error derivative with respect to each node's output.
      node.outputDer = 0.0;
      for (int j = 0; j < node.outputs.length; j++) {
        Link output = node.outputs[j];
        node.outputDer += output.weight * output.dest.inputDer;
      }
    }
  }
}

/**
 * Updates the weights of the network using the previously accumulated error
 * derivatives.
 */
updateWeights(List<List<Node>> network, double learningRate, double regularizationRate) {
  for (int layerIdx = 1; layerIdx < network.length; layerIdx++) {
    List<Node> currentLayer = network[layerIdx];
    for (int i = 0; i < currentLayer.length; i++) {
      Node node = currentLayer[i];
      // Update the node's bias.
      if (node.numAccumulatedDers > 0) {
        node.bias -= learningRate * node.accInputDer / node.numAccumulatedDers;
        node.accInputDer = 0.0;
        node.numAccumulatedDers = 0.0;
      }
      // Update the weights coming into this node.
      for (int j = 0; j < node.inputLinks.length; j++) {
        Link link = node.inputLinks[j];
        if (link.isDead) {
          continue;
        }
        double regulDer = link.regularization != null ? link.regularization.der(link.weight) : 0;
        if (link.numAccumulatedDers > 0) {
          // Update the weight based on dE/dw.
          link.weight = link.weight -
              (learningRate / link.numAccumulatedDers) * link.accErrorDer;
          // Further update the weight based on regularization.
          double newLinkWeight = link.weight - (learningRate * regularizationRate) * regulDer;
          if (link.regularization == RegularizationFunctions.L1 &&
              link.weight * newLinkWeight < 0) {
            // The weight crossed 0 due to the regularization term. Set it to 0.
            link.weight = 0.0;
            link.isDead = true;
          } else {
            link.weight = newLinkWeight;
          }
          link.accErrorDer = 0.0;
          link.numAccumulatedDers = 0.0;
        }
      }
    }
  }
}

/** Iterates over every node in the network/ */
forEachNode(List<List<Node>> network, bool ignoreInputs, accessor(Node node)) {
  for (int layerIdx = ignoreInputs ? 1 : 0; layerIdx < network.length; layerIdx++) {
    List<Node> currentLayer = network[layerIdx];
    for (int i = 0; i < currentLayer.length; i++) {
      Node node = currentLayer[i];
      accessor(node);
    }
  }
}

/** Returns the output node in the network. */
getOutputNode(List<List<Node>> network) {
  return network[network.length - 1][0];
}

/**
 * A node in a neural network. Each node has a state
 * (total input, output, and their respectively derivatives) which changes
 * after every forward and back propagation run.
 */
class Node {
  String id;

  /** List of input links. */
  List<Link> inputLinks = <Link>[];
  double bias = 0.1;

  /** List of output links. */
  List<Link> outputs = <Link>[];
  double totalInput;
  double output;

  /** Error derivative with respect to this node's output. */
  double outputDer = 0.0;

  /** Error derivative with respect to this node's total input. */
  double inputDer = 0.0;

  /**
   * Accumulated error derivative with respect to this node's total input since
   * the last update. This derivative equals dE/db where b is the node's
   * bias term.
   */
  double accInputDer = 0.0;

  /**
   * Number of accumulated err. derivatives with respect to the total input
   * since the last update.
   */
  double numAccumulatedDers = 0.0;

  /** Activation function that takes total input and returns node's output */
  ActivationFunction activation;

  /**
   * Creates a new node with the provided id and activation function.
   */
  Node(String id, ActivationFunction activation, {bool  initZero : true}) {
    this.id = id;
    this.activation = activation;
    if (initZero) {
      this.bias = 0.0;
    }
  }

  /** Recomputes the node's output and returns it. */
  double updateOutput() {
    // Stores total input into the node.
    this.totalInput = this.bias;
    for (int j = 0; j < this.inputLinks.length; j++) {
      Link link = this.inputLinks[j];
      this.totalInput += link.weight * link.source.output;
    }
    this.output = this.activation.output(this.totalInput);
    return this.output;
  }

  @override
  String toString() {
    String result = this.id + ' ';
    this.outputs.forEach((Link link){
      result += link.id + ' w=' + link.weight.toString() +',';
    });
    return result;
  }


}

/**
 * A link in a neural network. Each link has a weight and a source and
 * destination node. Also it has an internal state (error derivative
 * with respect to a particular input) which gets updated after
 * a run of back propagation.
 */
class Link {
  String id;
  Node source;
  Node dest;
  double weight = new Random().nextDouble() - 0.5;
  bool isDead = false;

  /** Error derivative with respect to this weight. */
  double errorDer = 0.0;

  /** Accumulated error derivative since the last update. */
  double accErrorDer = 0.0;

  /** Number of accumulated derivatives since the last update. */
  double numAccumulatedDers = 0.0;
  RegularizationFunction regularization;

  /**
   * Constructs a link in the neural network initialized with random weight.
   *
   * @param source The source node.
   * @param dest The destination node.
   * @param regularization The regularization function that computes the
   *     penalty for this weight. If null, there will be no regularization.
   */
  Link(Node source, Node dest, RegularizationFunction regularization, {bool initZero}) {
    this.id = source.id + "-" + dest.id;
    this.source = source;
    this.dest = dest;
    this.regularization = regularization;
    if (initZero) {
      this.weight = 0.0;
    }
  }
}

/** Built-in error functions */
class _SquareErrorFunction implements ErrorFunction {

  @override
  double error(double output, double target) => 0.5 * pow(output - target, 2);

  @override
  double der(double output, double target) => output - target;
}

class Errors {
  static ErrorFunction SQUARE = new _SquareErrorFunction();
}

/** Built-in activation functions */
class _TANHctivationFunction implements ActivationFunction {
  double output(double input) {
    if (input == double.INFINITY) {
      return 1.0;
    } else if (input == double.NEGATIVE_INFINITY) {
      return -1.0;
    } else {
      double e2x = exp(2 * input);
      return (e2x - 1) / (e2x + 1);
    }
  }

  double der(double input) {
    double output = this.output(input);
    return 1 - output * output;
  }
}

class _RELUActivationFunction implements ActivationFunction {
  double output(double input) => max(0.0, input);

  double der(double input) => input <= 0.0 ? 0.0 : 1.0;
}

class _SIGMOIDActivationFunction implements ActivationFunction {
  double output(double input) => 1 / (1 + exp(-input));

  double der(double input) {
    double output = this.output(input);
    return output * (1 - output);
  }
}

class _LINEARActivationFunction implements ActivationFunction {
  double output(double input) => input;

  double der(double input) => 1.0;
}

class Activations {
  static ActivationFunction TANH = new _TANHctivationFunction();
  static ActivationFunction RELU = new _RELUActivationFunction();
  static ActivationFunction SIGMOID = new _SIGMOIDActivationFunction();
  static ActivationFunction LINEAR = new _LINEARActivationFunction();
}

/** Build-in regularization functions */
class _L1RegularizationFunction implements RegularizationFunction {

  @override
  double output(double weight) => weight.abs();

  @override
  double der(double weight) => weight < 0.0 ? -1.0 : (weight > 0.0 ? 1.0 : 0.0);
}

class _L2RegularizationFunction implements RegularizationFunction {

  @override
  double output(double weight) => 0.5 * weight * weight;

  @override
  double der(double weight) => weight;
}

class RegularizationFunctions {
  static RegularizationFunction L1 = new _L1RegularizationFunction();
  static RegularizationFunction L2 = new _L2RegularizationFunction();
}

/**
 * Interfaces
 */

/**
 * An error function and its derivative.
 */
abstract class ErrorFunction {
  double error(double output, double target);

  double der(double output, double target);
}

/** A node's activation function and its derivative. */
abstract class ActivationFunction {
  double output(double input);

  double der(double input);
}

/** Function that computes a penalty cost for a given weight in the network. */
abstract class RegularizationFunction {
  double output(double weight);

  double der(double weight);
}
