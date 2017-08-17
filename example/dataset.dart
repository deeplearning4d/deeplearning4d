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
 * A two dimensional example: x and y coordinates with the label.
 */
class Example2D {
  double x;
  double y;
  double label;

  Example2D(this.x, this.y, this.label);

  @override
  String toString() {
//    return 'x=' + x.toString() + ', y=' + y.toString() + ', label=' + label.toString();
    return label.toString() + ',' + x.toString() + ',' + y.toString();
  }


}

class Point {
  double x;
  double y;
}


List<Example2D> classifyTwoGaussData(double numSamples, double noise) {
  List<Example2D> points = <Example2D>[];

  var varianceScale = new ScaleLinear(0.0, .5, 0.5, 4.0);
  double variance = varianceScale.scale(noise);

  genGauss(double cx, double cy, double label) {
    for (int i = 0; i < numSamples / 2; i++) {
      double x = normalRandom(mean: cx, variance: variance);
      double y = normalRandom(mean: cx, variance: variance);
      points.add(new Example2D(x, y, label));
    }
  }

  genGauss(2.0, 2.0, 1.0); // Gaussian with positive examples.
  genGauss(-2.0, -2.0, -1.0); // Gaussian with negative examples.
  return points;
}


/**
 * Samples from a normal distribution. Uses the seedrandom library as the
 * random generator.
 *
 * @param mean The mean. Default is 0.
 * @param variance The variance. Default is 1.
 */
double normalRandom({double mean: 0.0, double variance: 1.0}) {
  double v1, v2, s;
  do {
    v1 = 2 * new Random().nextDouble() - 1;
    v2 = 2 * new Random().nextDouble() - 1;
    s = v1 * v1 + v2 * v2;
  } while (s > 1);

  double result = sqrt(-2 * log(s) / s) * v1;
  return mean + sqrt(variance) * result;
}

/** Returns the eucledian distance between two points in space. */
double dist(Point a, Point b) {
  double dx = a.x - b.x;
  double dy = a.y - b.y;
  return sqrt(dx * dx + dy * dy);
}

class ScaleLinear {
  double r1;
  double r2;
  double d1;
  double d2;

  ScaleLinear(this.d1, this.d2, this.r1, this.r2);

  double scale (double x){
    double step = (r2 -r1)/(d2 -d1);
    return r1 + (x - d1) * step;

  }
}