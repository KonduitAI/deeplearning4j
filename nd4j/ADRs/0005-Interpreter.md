# Interpreter

## Status
Proposed

Proposed by: Adam Gibson (28-09-2020)

Discussed with: N/A

## Context
 

## Decision

An interpreter uses the [import IR](./0003-Import_IR.md) and the [mapping rule IR](./0004-Mapping_IR.md)
to execute and map operations from one framework to nd4j's file format and back.

This also allows execution of different frameworks via conversion in the nd4j engine.


A combination of the 2 allows a uniform interface to be used for the interpreter.

1 or more MappingRules will be used to transform 1 file format to another.


## Mapping Rules Execution

Mapping Rules are named functions that contain the function signature
(input and outputs). These mapping rules are used by the interpreter
to know which functions to execute.

The interpreter has built in implementations of the defined functions
for the desired transforms.



## Graph creation 

In order for execution to happen, a graph needs to be built.
This happens in java via the samediff builder.

The conversion happens as follows:
input node -> convert node to op descriptor via defined mapping rules -> add op descriptor to graph

The op descriptor is converted to a CustomOp which is then added to the graph via
[addArgsFor](https://github.com/KonduitAI/deeplearning4j/blob/88d3c4867fb87ec760b445c6b9459ecf353cec47/nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/autodiff/samediff/SameDiff.java#L1078)

This handles declarative graph creation setting dependencies up. Delegation of the graph structure
creation to the existing Samediff library enables the scope of this interpreter to be focused on 
mapping operations.

## Consequences
### Advantages
* Uses a common interface across different frameworks making maintenance simple

* Allows an easy to maintain abstraction for interop with different file formats

* Allows an easy entry point in to the framework without knowing much about the framework.

### Disadvantages

* Need to ensure compatibility across different frameworks

* Requires extensive testing to ensure proper compatibility

* May not necessarily support all ops people are expecting. This will be addressed
in a new ADR.
