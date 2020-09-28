# Import IR

## Status
Proposed

Proposed by: Adam Gibson (28-09-2020)

Discussed with: N/A

## Context

Currently, there is a gap in the way samediff/nd4j operations are 
implemented vs how other frameworks represent their models.

Keras, Tensorflow, and Pytorch use an attribute based format with names.
Interop between Onnx,Tensorflow, and Keras tends to follow the following formula:

1. Map names to equivalent names in other framework
for each operation configuration. Names being both op names
and associated attributes of the operations such as in Conv2D
where you have strides, kernel sizes.

2. Map input/output tensors to the equivalent tensor type
in each framework.

3. Setup the complete graph in the equivalent framework.
Sometimes the framework's concepts don't map 1 to 1.
They should output equivalent results regardless though. 
In order to do this, sometimes the framework needs to add/remove
operations in order to produce equivalent output in a different graph.
The [tensorflow onnx import](https://github.com/onnx/tensorflow-onnx#how-tf2onnx-works) 
is a good example of this.


Currently, samediff/nd4j have their internal op representations as 
a set of ordered arguments for execution in the form of:

1. t arguments: floating point arguments (float, double,..)

2. integer arguments: integer arguments (long, integer)

3. boolean argument: boolean arguments

4. data type arguments: data types for input/output

5. input arguments: ndarrays for input

6. output arguments: often optional (dynamically created) output ndarray
arguments. If the user wants to pass in outputs to control memory, they are allowed
to do so.

7. axis arguments: Integer arguments that represent the dimension(s)
for an operation to be executed on.

[Reference implementation](https://github.com/KonduitAI/deeplearning4j/blob/master/nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/api/ops/DynamicCustomOp.java#L58)

This maps well enough for execution, but not for file formats.
A bridge/intermediary format is proposed to handle making
writing import/interop code easier in the future.

This could be a future file format depending on how the framework evolves.
For now, this is considered a work around for making writing import code
easier/more portable.

## Decision

Similar to [ONNX](https://onnx.ai/) and  [Tensorflow](https://tensorflow.org/)
we will use protobuf for expressing an attribute based file format
mapping samediff/nd4j operations to this format.
We will have a translation layer that handles mapping
from attributes to the ordered arguments approach reflected
in samediff/nd4j.

For each operation, we will define a mapper to/from this attribute
format to the order based execution format.

This attribute based format will be an Intermediary Representation
that we then "compile" to the equivalent calls in libnd4j.

We can derive the existing attributes from the existing conventions of the 
libnd4j code base. An [example cpp file](https://github.com/KonduitAI/deeplearning4j/blob/master/libnd4j/include/ops/declarable/generic/nn/convo/conv1d.cpp#L104)
containing the following declaration:
```c++
auto inputShapeInfo   = inputShape->at(0);
auto weightsShapeInfo = inputShape->at(1);
Nd4jLong const* biasShapeInfo    = block.width() > 2 ? inputShape->at(2) : nullptr;

int kW = INT_ARG(0) > 0 ? INT_ARG(0) : static_cast<int>(shape::sizeAt(weightsShapeInfo, 0)); // filter(kernel) width
int sW = INT_ARG(1);                                                        // strides width
int pW = INT_ARG(2);                                                        // paddings width
int dW = INT_ARG(3);                                                        // dilations width
int paddingMode = INT_ARG(4);                                               // 0-VALID, 1-SAME
int isNCW  = block.getIArguments()->size() > 5 ? !INT_ARG(5) : 1;           // INT_ARG(4): 1-NWC, 0-NCW
int wFormat = block.getIArguments()->size() > 6 ? INT_ARG(6) : 0;           // 0 - [kW, iC, oC], 1 - [oC, iC, kW], 2 - [oC, kW, iC]
```

We can see that there are macros in the libnd4j code base reflecting
how each argument is accessed. Each list of arguments has an expected
ordering of arguments we need to explicitly map to a parsable
structure.


Comparing this to [onnx's Convolution operator](https://github.com/onnx/onnx/blob/master/docs/Operators.md#Conv), we see:

attributes with various types such as lists of ints and named tensors.
These concepts exist internally in the operations and layers themselves
in nd4j/samediff, but not are exposed directly to the user.

The proposal is simple: Expose a similar symbol based mapping
in libnd4j in a protobuf format to make it easier to interop with other
frameworks. 

Also store metadata about how to map each operation
to calls within libnd4j.

This bridge would allow for the following benefits:
 1. straightforward mapping of arguments for import
 2. provide an easy bridge to existing libnd4j
 3. allow automation of op descriptors
 in any language that would understand how to pass data to the
 c++ library.
 
 
 The format would have a defined schema as the following:
   1. FLOAT: Floating point values (float32, 64,..)
   2. INT:   Integer values (int32, int64)
   3. STRING: UTF8 String
   4. TENSOR: An ndarray
   5. NODE: An operation in a deeplearning pipeline
   6. GRAPH: A DAG of NODES
   7. SPARSE TENSOR: Sparse ndarray
   8. FLOATS: List of floats
   9. INTS: List of Integers
   10. STRINGS: List of strings
   11. TENSORS: List of Tensors
   12. GRAPHS: List of GRAPHS
   13. SPARSE_TENSORS: List of SPARSE_TENSOR
   
   4. TENSOR, with the following format:
         a. shape: INTS
         b. strides: INTS
         c. data type: enum  
   
   5. Attribute: A key/value pair with metadata describing type,
   comments, etc. Since the goal of this IR Is interop, we will target
   similarities to onnx while extending [the ONNX IR](https://github.com/onnx/onnx/blob/25fd2c332cf854fd38a92aa8f60d232530ab0065/onnx/onnx.proto#L113) for libnd4j execution
   use cases
    Attributes have the following values:
         
          a. name: the name of the attribute (STRING)
         
          b. type: the type of the attribute, types can have any value listed in line 124 :
            
          c. value: one of the above types
          
          d.  description: description of the attribute
          
    6. Node: The node in a sorted graph. This node will have 1 input and 1 output as well as additonal possible attributes. More in depth:
     
        a. input: string as input name
        b. output: string as output name
        c. name: the name of the node itself
        d. operation type: the operator referenced by name
        e. description: the description for the node (optional)
        
     
     
    7. Graph: A sorted in order directed acyclic graph representing a sequential
    set of operations to run. Details below:
    
     1. List of nodes
     2. name of note
     3. list of named tensors as input
     4. list of named sparse tensors as inputs
     5. the names of inputs to the graph
     6. the names of outputs to the graph
     
    