# Import IR

## Status

Proposed

Proposed by: Adam Gibson (28-09-2020)

Discussed with: Paul Dubs

## Context

Currently, there is a gap in the way samediff/nd4j operations are  implemented vs how other frameworks represent their models.

Keras, Tensorflow, and Pytorch use an attribute based format with names. Interop between Onnx ,Tensorflow, and Keras tends to follow the following formula:

1. Map names to equivalent names in other framework for each operation configuration. Names being both op names and associated attributes of the operations such as in Conv2D where you have strides, kernel sizes.
2. Map input/output tensors to the equivalent tensor type in each framework.
3. Setup the complete graph in the equivalent framework. Sometimes the framework's concepts don't map 1 to 1. They should output equivalent results regardless though.  In order to do this, sometimes the framework needs to add/remove operations in order to produce equivalent output in a different graph. The [tensorflow onnx import](https://github.com/onnx/tensorflow-onnx#how-tf2onnx-works)  is a good example of this.

Samediff/nd4j have their internal op representations as  a set of ordered arguments for execution in the form of:

1. t arguments: floating point arguments (float, double,..)
2. integer arguments: integer arguments (long, integer)
3. boolean argument: boolean arguments
4. data type arguments: data types for input/output
5. input arguments: ndarrays for input
6. output arguments: often optional (dynamically created) output ndarray arguments. If the user wants to pass in outputs to control memory, they are allowed to do so.
7. axis arguments: Integer arguments that represent the dimension(s) for an operation to be executed on.

[Reference implementation](https://github.com/KonduitAI/deeplearning4j/blob/master/nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/api/ops/DynamicCustomOp.java#L58)

This maps well enough for execution, but not for file formats. A bridge/intermediary format is proposed to handle making writing import/interop code easier in the future.

This could be a future file format depending on how the framework evolves. For now, this is considered a work around for making writing import code easier/more portable.

An example can be found under: src/main/proto/nd4j/nd4j.proto

## Proposal

Similar to [ONNX](https://onnx.ai/) and  [Tensorflow](https://tensorflow.org/) we will use protobuf for expressing an attribute based file format mapping samediff/nd4j operations to this format. We will have a translation layer that handles mapping from attributes to the ordered arguments approach reflected in samediff/nd4j.

For each operation, we will define a mapper to/from this attribute format to the order based execution format.

This attribute based format will be an Intermediary Representation that we then "compile" to the equivalent calls in libnd4j.

We can derive the existing attributes from the existing conventions of the  libnd4j code base. An [example cpp file](https://github.com/KonduitAI/deeplearning4j/blob/master/libnd4j/include/ops/declarable/generic/nn/convo/conv1d.cpp#L104) containing the following declaration:

```
`auto inputShapeInfo   = inputShape->at(0);
auto weightsShapeInfo = inputShape->at(1);
Nd4jLong const* biasShapeInfo    = block.width() > 2 ? inputShape->at(2) : nullptr;

int kW = INT_ARG(0) > 0 ? INT_ARG(0) : static_cast<int>(shape::sizeAt(weightsShapeInfo, 0)); // filter(kernel) width
int sW = INT_ARG(1);                                                        // strides width
int pW = INT_ARG(2);                                                        // paddings width
int dW = INT_ARG(3);                                                        // dilations width
int paddingMode = INT_ARG(4);                                               // 0-VALID, 1-SAME
int isNCW  = block.getIArguments()->size() > 5 ? !INT_ARG(5) : 1;           // INT_ARG(4): 1-NWC, 0-NCW
int wFormat = block.getIArguments()->size() > 6 ? INT_ARG(6) : 0;           // 0 - [kW, iC, oC], 1 - [oC, iC, kW], 2 - [oC, kW, iC]
`
```

We can see that there are macros in the libnd4j code base reflecting how each argument is accessed. Each list of arguments has an expected ordering of arguments we need to explicitly map to a parseable structure.

Comparing this to [onnx's Convolution operator](https://github.com/onnx/onnx/blob/master/docs/Operators.md#Conv), we see:

attributes with various types such as lists of ints and named tensors. These concepts exist internally in the operations and layers themselves in nd4j/samediff, but not are exposed directly to the user.

The proposal is simple: 
Expose a similar symbol based mapping in libnd4j in a protobuf format to make it 
easier to interop with other frameworks adding proper information for being able to map
attributes to arguments in each list present in an op.

This bridge would allow for the following benefits:

1.    straightforward mapping of arguments for import  
2. provide an easy bridge to existing libnd4j  
3.  allow automation of op descriptors  in any language that would understand how to pass data to the  c++ library.



## Format Example

Similar to onnx and tensorflow, the goals are as follows:

1. Define an attribute based op schema allowing interop between tensorflow/onnx and nd4j
2. Within the same op schema, define mappings from attributes to indexed arguments in each list present within libnd4j.

 We can do that with an  op definition schema similar to tensorflow located at:
https://github.com/KonduitAI/deeplearning4j/blob/b5f0ec072f3fd0da566e32f82c0e43ca36553f39/nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/resources/ops.proto#L1742

In addition to the attribute based mapping we need for each op, we also have descriptions of the operation indexing.
An add op in tensorflow looks like:

```
op {
  name: "Add"
  input_arg {
    name: "x"
    type_attr: "T"
  }
  input_arg {
    name: "y"
    type_attr: "T"
  }
  output_arg {
    name: "z"
    type_attr: "T"
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_BFLOAT16
        type: DT_HALF
        type: DT_FLOAT
        type: DT_DOUBLE
        type: DT_UINT8
        type: DT_INT8
        type: DT_INT16
        type: DT_INT32
        type: DT_INT64
        type: DT_COMPLEX64
        type: DT_COMPLEX128
        type: DT_STRING
      }
    }
  }
}
```

Onnx’s add can be found here https://github.com/onnx/onnx/blob/master/docs/Operators.md#Add

In nd4j an add would simply be 2 input ndarrays. This metadata was added to the attribute based information already found in the tensorflow and onnx file formats.
An nd4j IR extension would look like:

```


op {
  name: "Add"
  input_arg {
    name: "x"
    type_attr: "T"
  }
  input_arg {
    name: "y"
    type_attr: "T"
  }
  output_arg {
    name: "z"
    type_attr: "T"
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_BFLOAT16
        type: DT_HALF
        type: DT_FLOAT
        type: DT_DOUBLE
        type: DT_UINT8
        type: DT_INT8
        type: DT_INT16
        type: DT_INT32
        type: DT_INT64
        type: DT_COMPLEX64
        type: DT_COMPLEX128
        type: DT_STRING
      }
    }
  }
  
  repeated int64 ints;
  repeated bool booleans;
  repeated int32 axisArguments;
  
  
}
```


Note above that we add list information to the attribute based declaration from onnx.

###Op Descriptor

An op descriptor from libnd4j is as follows:
```java
 private String name;
    private int nIn,nOut,tArgs,iArgs;
    private boolean inplaceAble;
    private List<String> inArgNames;
    private List<String> outArgNames;
    private List<String> tArgNames;
    private List<String> iArgNames;
    private List<String> bArgNames;
    private OpDeclarationType opDeclarationType;

    public enum OpDeclarationType {
        CUSTOM_OP_IMPL,
        BOOLEAN_OP_IMPL,
        LIST_OP_IMPL,
        LOGIC_OP_IMPL,
        OP_IMPL,
        DIVERGENT_OP_IMPL,
        CONFIGURABLE_OP_IMPL,
        REDUCTION_OP_IMPL,
        BROADCASTABLE_OP_IMPL,
        BROADCASTABLE_BOOL_OP_IMPL
    }
```
These contain all the op declarations and fields associated with a descriptor.
Validation for what can be present in the various names can eb found 
[here](https://github.com/KonduitAI/deeplearning4j/blob/master/libnd4j/include/ops/declarable/impl/DeclarableOp.cpp#L734-L765)

A declaration in libnd4j is a set of macros that can be found
[here](https://github.com/eclipse/deeplearning4j/blob/master/libnd4j/include/system/op_boilerplate.h)

All the macros contain various declarations that are easy to find
for automatically extracting out what properties are declared with what variable names.

We use this to create automatic attribute mappings that can be serialized
as a protobuf file for interpretation by an interpreter.

###Interpreter

An interpreter will take a tensorflow or pytorch model and figure out how to map
various ops. Their attributes and op names will be mapped to libnd4j
using information from the above op descriptor.

An interpreter can take in an individual op from tensorflow, onnx or
another framework and translate it to an equivalent op in libnd4j represented
as the equivalent op descriptor.

The usage will be as follows:

```java
Interpreter interpreter = ...;

OpDescriptor descriptor = interpreter.interpret(nodeFromOtherFramework);

//proceed to use descriptor to map for model import...
```


## Consequences

Migration to an attribute based import format makes working with other deep learning frameworks easier in the future.

This may encourage future work to be done to the samediff file format.
Of note here for prior work is the current code generation
https://github.com/KonduitAI/dl4j-dev-tools/blob/master/codegen/src/main/ops/org/nd4j/codegen/ops/CNN.kt#L28

While it does have the intended description, it’s kotlin specific and is only available for a very small subset of the ops where pre-created objects were created for specific operations. The goal of this ADR is to expand upon that and make it language agnostic by providing this information in a neutral file format that has code generation with it.

Current code generation efforts can be augmented using this file format. More on this decision making can be found https://github.com/KonduitAI/dl4j-dev-tools/blob/master/codegen/adr/0007-configuration_objects.md

### Drawbacks

1. Yet another file format.
2. Risk migrating to new file format in the future.
3. A lot of up front manual work to index set of current operations.
4. Backwards compatibility: yet another thing to maintain. We will need to write a converter for any forward compatibility. We can address this with an opset schema scheme similar to onnx.

### Advantages

1. Easy to maintain.
2. Backwards compatible.
3. Easily interops with existing other deep learning frameworks.
4. No additional dependencies from what's already normal.
5. Protobuf allows easy code generation for other languages.
6. Industry standard conventions being used over proprietary tooling reducing friction for adoption for people coming from other frameworks

