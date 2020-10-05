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


## Op def files

For each framework in tensorflow/onnx, we have inbuilt definition files
for each tensorflow and pytorch.

For onnx, we have an onnx.pbtxt generated by the dl4j-dev tools submodule 
onnx-defs. This definition file has each op serialized as an [onnx NodeProto](https://github.com/onnx/onnx/blob/25fd2c332cf854fd38a92aa8f60d232530ab0065/onnx/onnx-ml.proto#L193)
For tensorflow, we have an ops.proto pulled from tensorflow's official repo.

We will use these files to map operation attributes serialized by 
nd4j's generated operation definition tool found in dl4j-dev-tools
to their equivalents in tensorflow and pytorch.

An interpreter will have 2 methods:


```java
Interpreter interpreter = ...;

OpDescriptor descriptor = interpreter.interpretTensorflow(nodeFromOtherFramework);
OpDescriptor descriptor = interpreter.interpretOnnx(nodeFromOtherFramework);

//proceed to use descriptor to map for model import...
```  

##Interpreter file format

An interpreter will be language neutral. We will have a mini syntax
for mapping attributes from one format to another.

Through indexing every attribute and input/output in libnd4j,
we will maintain an index of operation names and attributes
with a mapping syntax. If we want to map a trivial operation like say:
Abs, let's compare tensorflow, onnx and the descriptor in nd4j.

Tensorflow:
```prototext
op {
  name: "Floor"
  input_arg {
    name: "x"
    type_attr: "T"
  }
  output_arg {
    name: "y"
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
      }
    }
  }
}
```

Onnx:
```prototext
input: "X"
output: "Y"
name: "Floor"
op_type: "Floor"
attribute {
  name: "X-types"
  strings: "float"
  strings: "float16"
  strings: "double"
  type: STRINGS
}
doc_string: "\nFloor takes one input data (Tensor<T>) and produces one output data\n(Tensor<T>) where the floor is, y = floor(x), is applied to\nthe tensor elementwise.\n"
```

The op descriptor for libnd4j is:
```
OpDeclarationDescriptor(name=Floor, nIn=1, nOut=1, tArgs=0, iArgs=0, inplaceAble=true, inArgNames=[first], outArgNames=[z], tArgNames=[], iArgNames=[], bArgNames=[], opDeclarationType=OP_IMPL)
```

Floor is a fairly simple op with 1 input and 1 output.
Inputs and outputs are implicitly tensors.
This is true for both onnx and tensorflow.

Tensorflow has an attribute defined for valid types.
The way we generated the onnx schema proto, we have something equivalent
that allows for a list of types presented as a string.


Mapping a descriptor happens based on attribute.
An example of abs below:

```prototext
floor {
  tensorflow_mapping: {
     input_mappings: {
       input_mapping {
        first: "x"
     }
         
     }
     output_mappings: {
        z: "y"
     }
  
      attribute_mapping_functions: {

     }

  }
  onnx_mapping {
    input_mappings {
       first: "X"
    }
    output_mappings {
        z: "Y"
    }

     attribute_mapping_functions {

     }
  }
}
```

Now we can compare this to Convolution.
In tensorflow, the convolution op is represented as:
```prototext
op {
  name: "Conv2D"
  input_arg {
    name: "input"
    type_attr: "T"
  }
  input_arg {
    name: "filter"
    type_attr: "T"
  }
  output_arg {
    name: "output"
    type_attr: "T"
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_HALF
        type: DT_BFLOAT16
        type: DT_FLOAT
        type: DT_DOUBLE
      }
    }
  }
  attr {
    name: "strides"
    type: "list(int)"
  }
  attr {
    name: "use_cudnn_on_gpu"
    type: "bool"
    default_value {
      b: true
    }
  }
  attr {
    name: "padding"
    type: "string"
    allowed_values {
      list {
        s: "SAME"
        s: "VALID"
      }
    }
  }
  attr {
    name: "data_format"
    type: "string"
    default_value {
      s: "NHWC"
    }
    allowed_values {
      list {
        s: "NHWC"
        s: "NCHW"
      }
    }
  }
  attr {
    name: "dilations"
    type: "list(int)"
    default_value {
      list {
        i: 1
        i: 1
        i: 1
        i: 1
      }
    }
  }
}
```
In onnx, it's represented as:
```prototext
input: "X"
input: "W"
input: "B"
output: "Y"
name: "Conv"
op_type: "Conv"
attribute {
  name: "auto_pad"
  s: "NOTSET"
  type: STRING
}
attribute {
  name: "dilations"
  s: ""
  type: INTS
}
attribute {
  name: "group"
  i: 1
  type: INT
}
attribute {
  name: "kernel_shape"
  s: ""
  type: INTS
}
attribute {
  name: "pads"
  s: ""
  type: INTS
}
attribute {
  name: "strides"
  s: ""
  type: INTS
}
attribute {
  name: "X-types"
  strings: "double"
  strings: "float"
  strings: "float16"
  type: STRINGS
}
attribute {
  name: "W-types"
  strings: "double"
  strings: "float"
  strings: "float16"
  type: STRINGS
}
attribute {
  name: "B-types"
  strings: "double"
  strings: "float"
  strings: "float16"
  type: STRINGS
}
doc_string: "\nThe convolution operator consumes an input tensor and a filter, and\ncomputes the output."
```

The libnd4j OpDescriptor:
```
OpDeclarationDescriptor(name=conv2d, nIn=2, nOut=1, tArgs=0, iArgs=9, inplaceAble=false, inArgNames=[input, weights, bias, gradO, gradIShape], outArgNames=[output, gradI, gradW, gradB], tArgNames=[], iArgNames=[sH, sW, pH, pW, dH, dW, isSameMode, isNCHW, wFormat, kH, kW], bArgNames=[], opDeclarationType=CUSTOM_OP_IMPL)
```

A few challenges stand out when trying to map all these formats
to nd4j:

1. Different conventions for the same concept. One example that stands out from conv
is padding. Padding can be represented as a string or have a boolean that says what a string equals.
In nd4j, we represent this as a boolean: isSameMode. We need to do a conversion inline in order
to invoke nd4j correctly.

2. Another issue is implicit concepts. Commonly, convolution requires you to configure a layout
of NWHC (Batch size, Height, Width, Channels) 
or NCHW (Batch size, Channels,Height, Width). Tensorflow allows you to specify it,
nd4j also allows you to specify it. Onnx does not.
 
 A more in depth conversation on this specific issue relating to the 
 2 frameworks can be found [here](https://github.com/onnx/onnx-tensorflow/issues/31)

In order to bypass this issue, a mapping rule is needed.
Defining this example can be found as MappingRule
and Mapper found in [./nd4j-backends/nd4j-api-parent/nd4j-api/src/main/protobuf/nd4j/nd4j.proto]

We use these 2 core concepts to define a set of rules
allowing translation for each op within each framework.

A mapper for nd4j and tensorflow can be found as follows:

```prototext
MappingRule {
  name: "nchwConversion"
  functionName: "StringEqualsConversion"  
}

MappingRule {
  name: "strideIndexLookup"
  functionName: "IndexLookup"
}
MappingRule {
  name: "kernelIndexLookup"
  functionName: "IndexLookup"
}


Mapper {
  name: "tensorflowConversion"
  opName: "conv2d"
  rules {
    list {
     MappingRule {
        name: "nchwConversion"
        functionName: "StringEqualsConversion"  
    }

    MappingRule {
      name: "strideIndexLookup"
      functionName: "IndexLookup"
    }

    MappingRule {
      name: "kernelIndexLookup"
      functionName: "IndexLookup"
    }

    }
  }
 

 # attribute mappings..
}
```





Convolution is significantly more complicated and does not map
1 to 1. In order to properly map both tensorflow and onnx
to libnd4j, we need to have adapter functions built in.

Adapter functions need to be built in and defined if they are
to be language neutral. It also might be possible to add custom functions
depending on the use case. 

Adapter functions are simple implementations in the nd4j
side (and would need to be implemented for other languages if added)
where the mapper will specify by name what adapter functions to invoke.
This allows definition of rules and functions in protobuf
and clear mappings from declaration to implementation (that could be picked up by code generation later)

Adapter function invocations are handled by declaring rules.
Rules specify how to lookup arguments and what function to run on them.

An individual mapper declaration for an op consists of a list of rules
to apply for a given op.The declaration contains the framework 
name meant to map for, and the name of the nd4j op to map.

This allows the user to lookup which mapper to use for what framework
at runtime. This also unifies the mapper framework from the current
tensorflow import which has similar metadata, but only in java.





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

