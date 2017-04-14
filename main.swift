/* Daniel Seitz
 * Assignment 1
 * CS445 - Machine Learning, Spring 2017
 * 4/6/17
 *
 * USAGE: hw1 <eta> [training_file test_file]
 *
 * Tested with: Apple Swift version 3.1 (swiftlang-802.0.48 clang-802.0.38)
 * Compile with:
 *    make
 * For faster running time compile with:
 *    make release
 *
 * Example:
 *    > make
 *    > ./hw1 0.001
 */

import Foundation

let inputSize: Int = 785
let numEpochs: Int = 50

extension Array {
  /** Shuffles an array in place with the Fisher-Yates shuffle algorithm
   */
  mutating func shuffle() {
    let c = self.count
    guard c > 1 else { return }

    for (firstUnshuffled, unshuffledCount) in zip(self.indices, stride(from: c, to: 1, by: -1)) {
      let d: IndexDistance = numericCast(arc4random_uniform(numericCast(unshuffledCount)))
      guard d != 0 else { continue }
      let i = self.index(firstUnshuffled, offsetBy: d)
      swap(&self[firstUnshuffled], &self[i])
    }
  }

  /** Returns the element at the specified index iff it is within bounds, otherwise nil.
   */
  subscript (safe index: Index) -> Element? {
    return self.indices.contains(index) ? self[index] : nil
  }
}

/** Struct representing a single input image. It contains an array of `Double` values in the range
 *  [0,1] representing each pixel which will be used as the input values. It also stores the
 *  expected result as an `Int`. So an input that should be seen as a '3' would have an expected
 *  value of `3`.
 */
struct Input {
  let input: [Double]
  let expected: Int

  /** Initialize the input struct with a given array of data. The input parameter must have the
   *  expected output as the first element of the array, and each element of the input array must
   *  be in the range [0,255].
   */
  init(_ input: [Double]) {
    // There should only be values in the range of [0,255] in the input array
    assert(input.filter({ 0 > $0 || $0 > 255 }).isEmpty)
    self.expected = Int(input[0])
    // Normalize the values
    var input = input.map({ $0 / 255 })

    // bias should always have weight 1
    input[0] = 1
    self.input = input
  }

  /** Display input array along with expected value.
   *
   *  For debugging
   */
  func display() {
    print("Input { input: \(self.input), expected: \(self.expected) }")
  }
}

/** Struct representing a single perceptron.
 *
 *  A perceptron has a set of weights given for each input parameter. The data we are woking with
 *  contains 785 inputs, so the `weights` array will have length 785. Each perceptron should have
 *  a target number that it is trying to recognize. If it is passed data that it thinks matches its
 *  target number, then it should output a positive number, otherwise it should output a negative
 *  number. The greater the output value the more confidence the perceptron has that it is correct.
 */
struct Perceptron {
  var weights: [Double]
  let target: Int

  /** Initialize the Perceptron with a given input size and target number.
   *
   *  The weights for the perceptron will be randomly generated in the range [-0.05, 0.05]
   */
  init(inputSize: Int, target: Int) {
    // Initialize weights to random values in range [-0.05, 0.05]
    self.weights = [Double](repeating: 0, count: inputSize).map { _ in
      // arc4random returns a value in range [0, 2^32]
      (Double(Double(arc4random()) / Double(UInt32.max)) - 0.5) / 10
    }
    self.target = target
  }

  /** Initialize a perceptron with an already calculated array of weights.
   */
  init(weights: [Double], target: Int) {
    self.weights = weights
    self.target = target
  }

  /** Calculate the output value for a given input array.
   *
   *  The length of the input array MUST be the same length as the weights array.
   */
  func calculate(inputs: [Double]) -> Double {
    assert(self.weights.count == inputs.count)

    // Calculate dot product
    let result = zip(self.weights, inputs)
      .map({(w, x) in
        w * x
      })
      .reduce(0, +)

    return result
  }

  /** Update the weights of the perceptron based on the given input array.
   *
   *  This updates the perceptron in place
   *
   *  The length of the input array MUST be the same length as the weights array
   */
  mutating func update(eta: Double, expectedTarget: Int, inputs: [Double]) {
    // We should always have a same airity input vector and weight vector
    assert(self.weights.count == inputs.count)
    let result = self.calculate(inputs: inputs) > 0 ? 1 : 0

    // If our target is the same as expected, then we want to output a 1
    let expected = self.target == expectedTarget ? 1 : 0

    // Small optimization, if the result is what we expected, we don't need to update on this pass
    // since `delta w` will just be 0
    if result != expected {
      var newWeights = [Double](repeating: 0, count: self.weights.count)
      for (i, weight) in self.weights.enumerated() {
        // w_i = w_i + eta(t - y)x_i
        // where w_i is the current weight,
        //       eta is a constant
        //       t is the expected value
        //       y is the calculated value
        //       x_i is the input associated with the current weight
        newWeights[i] = weight + eta * Double(expected - result) * inputs[i]
      }
      self.weights = newWeights
    }
  }

  /** Update the weights of the perceptron based on the given input array.
   *
   *  This returns a new perceptron with the modified weights, the original perceptron stays
   *  unchanged.
   *
   *  The length of the input array MUST be the same length as the weights array
   */
  func updated(eta: Double, expectedTarget: Int, inputs: [Double]) -> Perceptron {
    var newPerceptron = Perceptron(weights: self.weights, target: self.target)
    newPerceptron.update(eta: eta, expectedTarget: expectedTarget, inputs: inputs)
    return newPerceptron
  }

  /** Print the weights of the perceptron.
   *
   *  For debugging
   */
  func printWeights() {
    for (i, weight) in self.weights.enumerated() {
      print("[\(i)] \(weight)")
    }
  }
}

/** Load a given filename and return the contents as a `String`
 *
 *  Throws an error if the file cannot be opened
 */
func load(file: String) throws -> String {
  let fileName = URL(fileURLWithPath: file)

  let contents = try Data(contentsOf: fileName)
  return String(data: contents, encoding: .utf8)!
}

/** Parse a CSV formatted string as a matrix of `Double`s where each row will represent one set of
 *  input data
 */
func parse(csv: String) -> [[Double]] {
  // Parse the csv string into an array of arrays of doubles in the range [0,255].
  // 255 is the max value in the file
  return csv.components(separatedBy: "\n")
            .filter({ !$0.isEmpty })
            .map({ $0.components(separatedBy: ",").map({ Double($0)! }) })
}

/** Predict what value a given input is based on an array of Perceptrons.
 *
 *  This will return the target value of the Perceptron with the highest confidence, as multiple
 *  perceptrons may output a true value on a given input.
 */
func predict(data: Input, withPerceptrons perceptrons: [Perceptron]) -> Int {
  var prediction: (Double, Int)? = nil
  for perceptron in perceptrons {
    let result = perceptron.calculate(inputs: data.input)

    if let (max, _) = prediction {
      if result > max {
        prediction = (result, perceptron.target)
      }
    }
    else {
      prediction = (result, perceptron.target)
    }
  }

  return prediction!.1
}

/** Calculate the accuracy for a given set of inputs using a given array of perceptrons.
 *
 *  This will return a value in the range [0,1], with 1 being 100% accurate, meaning all inputs
 *  were successfully predicted
 */
func calculateAccuracy(forData inputData: [[Double]], withPerceptrons perceptrons: [Perceptron]) -> Double {
  var correct = 0
  for rawInput in inputData {
    let input = Input(rawInput)
    let target = predict(data: input, withPerceptrons: perceptrons)
    if input.expected == target {
      correct += 1
    }
  }

  return Double(correct) / Double(inputData.count)
}

/** Calculate a confusion matrix for a given set of inputs using a given array of perceptrons.
 */
func calculateConfusionMatrix(forData inputData: [[Double]], withPerceptrons perceptrons: [Perceptron]) -> [[Int]] {
  var confusionMatrix: [[Int]] = [[Int]](repeating: [Int](repeating: 0, count: 10), count: 10)
  for rawInput in inputData {
    let input = Input(rawInput)
    let predicted = predict(data: input, withPerceptrons: perceptrons)
    confusionMatrix[input.expected][predicted] += 1
  }
  return confusionMatrix
}

/*** Entry ***/

if CommandLine.arguments.count < 2 {
  print("USAGE: \(CommandLine.arguments[0]) <eta-value> [<training-file> <test-file>]")
  exit(1)
}

guard let eta = Double(CommandLine.arguments[1]) else {
  print("<eta-value> must be a number!")
  exit(1)
}

// Default file names, configurable on the command line
let trainingFile = CommandLine.arguments[safe: 2] ?? "mnist_train.csv"
let testFile = CommandLine.arguments[safe: 3] ?? "mnist_test.csv"

print("Using file: \(trainingFile) for training data")
print("Using file: \(testFile) for test data")

// Load the file data into memory
let trainingFileData: String
let testFileData: String
do {
  trainingFileData = try load(file: trainingFile)
  testFileData = try load(file: testFile)
}
catch {
  print(error.localizedDescription)
  exit(1)
}

// Parse the CSV data into a matrix of Doubles
var trainingData = parse(csv: trainingFileData)
let testData = parse(csv: testFileData)

var perceptrons = [Perceptron(inputSize: inputSize, target: 0),
                   Perceptron(inputSize: inputSize, target: 1),
                   Perceptron(inputSize: inputSize, target: 2),
                   Perceptron(inputSize: inputSize, target: 3),
                   Perceptron(inputSize: inputSize, target: 4),
                   Perceptron(inputSize: inputSize, target: 5),
                   Perceptron(inputSize: inputSize, target: 6),
                   Perceptron(inputSize: inputSize, target: 7),
                   Perceptron(inputSize: inputSize, target: 8),
                   Perceptron(inputSize: inputSize, target: 9)]

// Arrays to store the result of each epoch
var trainingOutput: [Double] = []
var testOutput: [Double] = []
trainingOutput.reserveCapacity(numEpochs)
testOutput.reserveCapacity(numEpochs)

// Calculate initial accuracy for randomized weights
print("Finding accuracy on initial data")
var trainingAccuracy = calculateAccuracy(forData: trainingData, withPerceptrons: perceptrons)
var testAccuracy = calculateAccuracy(forData: testData, withPerceptrons: perceptrons)
print("Training Accuracy: \(trainingAccuracy)")
trainingOutput.append(trainingAccuracy)
print("Test Accuracy: \(testAccuracy)")
testOutput.append(testAccuracy)

// Update weights if needed and recalculate accuracy
for _ in 0..<numEpochs {
  // Update weights
  for rawData in trainingData {
    let input = Input(rawData)

    for i in perceptrons.indices {
      perceptrons[i] = perceptrons[i].updated(eta: eta, expectedTarget: input.expected, inputs: input.input)
    }
  }

  // Calculate new accuracy
  print("Finding accuracy after updated weights")
  trainingAccuracy = calculateAccuracy(forData: trainingData, withPerceptrons: perceptrons)
  testAccuracy = calculateAccuracy(forData: testData, withPerceptrons: perceptrons)
  print("Training Accuracy: \(trainingAccuracy)")
  trainingOutput.append(trainingAccuracy)
  print("Test Accuracy: \(testAccuracy)")
  testOutput.append(testAccuracy)

  trainingData.shuffle()
}

let confusionMatrix = calculateConfusionMatrix(forData: testData, withPerceptrons: perceptrons)

// Save data in CSV format to output files
do {
  let trainingOutputData = trainingOutput.map({ String($0) }).joined(separator: ",")
  let testOutputData = testOutput.map({ String($0) }).joined(separator: ",")
  let confusionMatrixData = confusionMatrix.map({ String($0.map({ String($0) }).joined(separator: ",")) }).joined(separator: "\n")

  let trainingFile = URL(fileURLWithPath: "training_out_eta_\(eta).csv")
  let testFile = URL(fileURLWithPath: "test_out_eta_\(eta).csv")
  let confusionFile = URL(fileURLWithPath: "confusion_out_eta_\(eta).csv")

  print("Writing data to \(trainingFile)")
  try trainingOutputData.write(to: trainingFile, atomically: false, encoding: .utf8)
  print("Writing data to \(testFile)")
  try testOutputData.write(to: testFile, atomically: false, encoding: .utf8)
  print("Writing data to \(confusionFile)")
  try confusionMatrixData.write(to: confusionFile, atomically: false, encoding: .utf8)
}
catch {
  print(error.localizedDescription)
  exit(1)
}
