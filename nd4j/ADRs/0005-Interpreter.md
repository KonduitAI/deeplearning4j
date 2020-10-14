# Import IR

## Status
Proposed

Proposed by: Adam Gibson (28-09-2020)

Discussed with: N/A

## Context
 

## Decision

Below, the mapping rule format is discussed.
All of the various aspects and challenges this


## MappingRule Implementation


Adapter functions are simple implementations in the nd4j
side (and would need to be implemented for other languages if added)
where the mapper specifies by name what adapter functions to invoke.
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


In order to map inputs, we need to be able to take the descriptors
mentioned above and do 1 to 1 mappings to the names of each op.

In order to do this, an attribute mapping is needed. Attributes are
simply named types. Both tensorflow and onnx have this format.
The op descriptor format also aligns with this pattern.


## Transformation definitions

For defining and implementing transforms, a transform is just a name.
The IR consumption language needs to implement the desired operations.
In the protobuf spec, we call this a MappingRule.

A definition/mapping rule is an annotated function that gets dynamically picked up at runtime.

Its inputs are as follows:
 1. framework specific instantiated op definition
 2. framework specific op descriptor
 
Its outputs are an op descriptor definition from nd4j representing the needed information
to facilitate creation of a custom op.
``
The annotated functions all need to be picked up at runtime and match what is in the desired
framework mapping. This includes any custom operations or transforms discovered during the initial scan.
Validation is done at the beginning of the loading process.

## Mapping Rule

A MappingRule maps attributes of the input framework to the nd4j OpDescriptor
mentioned above.

This allows us to describe a list of rules that map an input framework's operations
to the nd4j OpDescriptor format.

Defining this example can be found as MappingRule
and Mapper found in [./nd4j-backends/nd4j-api-parent/nd4j-api/src/main/protobuf/nd4j/nd4j.proto]

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

A MappingRule contains a named function that shows how to map one attribute
name to another.

In our tensorflow import we implement [examples of these functions](https://github.com/KonduitAI/deeplearning4j/tree/master/nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/imports/descriptors/properties/adapters)
Adapter functions extend this concept, but are intended to be language neutral. 

A separate ADR will cover how we implement these functions.

## Contrasting MappingRules with another implementation

We map names and types to equivalent concepts in each framework.
Onnx tensorflow does this with an [attribute converter](https://github.com/onnx/onnx-tensorflow/blob/08e41de7b127a53d072a54730e4784fe50f8c7c3/onnx_tf/common/attr_converter.py)

This is done by a handler (one for each op).
More can be found [here](https://github.com/onnx/onnx-tensorflow/tree/master/onnx_tf/handlers/backend)



Below we address the challenges in bridging the gap between the 2 formats.

## Challenges when mapping nd4j ops

The above formats are vastly different. Onnx and tensorflow
are purely attribute based. Nd4j is index based.
This challenge is addressed by the IR by adding names to each property.


In order to actually map these properties, we need to define rules for doing so.
Examples of why these mapping rules are needed below:

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
In order to address these challenges, we introduce a MappingRule allowing
us to define a series of steps to map the input format to the nd4j format
in a language neutral way via a protobuf declaration.





## Consequences
### Advantages
* Allows a language neutral way of describing a set of transforms necessary
for mapping an set of operations found in a graph from one framework to the nd4j format.

* Allows a straightforward way of writing an interpreter as well as mappers
for different frameworks in nd4j in a standardized way.

* Replaces the old import and makes maintenance of imports/mappers more straightforward.

### Disadvantages

* More complexity in the code base instead of a more straightforward java implementation.

* Risks introducing new errors due to a rewrite
