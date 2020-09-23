# Onnx runtime module

## Status
Proposed

Proposed by: Adam Gibson (23-09-2020)

Discussed with: raver119

## Context

We need a way of providing nd4j a way of running onnx modules
that is easily compatible with the onnx community. The gold standard for this
is is using [onnxruntime](https://github.com/microsoft/onnxruntime/blob/master/docs/Java_API.md).


## Decision

We will use javacpp's onnxruntime bindings in a similar manner to [nd4j-tensorflow](../nd4j-tensorflow)
allowing nd4j to be used as an ndarray format that interops with onnxruntime.

We will implement a simple api similar to the [GraphRunner](../nd4j-tensorflow/src/main/java/org/nd4j/tensorflow/conversion/graphrunner/GraphRunner.java)
This will sit on top of javacpp's lower level onnxruntime bindings.

This module will follow a similar structure to the nd4j-tensorflow module
focusing on INDArrays as a data interchange format, but otherwise pass execution
down to onnxruntime.


The main api to the graph runner works as follows:

```java
try(GraphRunner runner = new GraphRunner(...)) {
   Map<String,INDArray> inputs = new HashMap<>();
   // ..initialize inputs
 Map<String,INDArray> outputs = runner.run(inputs);
// process outputs...
}
```

The core logic will contain the following components:

1. Loading onnx pb files
2. A graph runner in similar nature to nd4j-tensorflow
3. Interop with onnxruntime's version of an ndarray/tensor