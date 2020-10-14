# Mapper Scanning

## Status
Proposed

Proposed by: Adam Gibson (28-09-2020)

Discussed with: N/A

## Context
 

## Decision

In order to allow easy extension of the interpreter framework
presented in [the interpreter ADR](./0005-Interpreter.md)
we leverage a widely used concept in the JVM community: [annotation scanning](https://www.baeldung.com/spring-component-scanning).

Annotation scanning allows scanning for registered op mappings and mapping rules.

Dynamic discovery of annotations is needed to allow for custom operations to be specified by the user
in the event that a particular op is not supported for the user's use case.

## Mapper Generation

Given the above configuration used for representing ops and ops transformations,
we also automatically generate those mappings.

We generate mappings from kotlin mappings and outputting the above protobuf format
that can be read/consumed/interpreted by any language supported by protobuf and implements
the spec operations needed for processing op transformations needed for mapping.

In order to implement a mapping, we annotate a generator that generates an op descriptor
for a particular op.

We discover these mapper generators with annotation scanning. The annotation looks like the
following:
```java
@MapperGenerator("opName","frameworkName")
```

where opName is the name of the nd4j op to generate a mapping for and frameworkName
is the framework to generate a mapping for.

This scan dynamically happens at runtime allowing for discovery of custom operation generators
as well. This allows workarounds for missing ops.



## Mapper Consumption

In order to consume the IR and write an interpreter for the IR, a few simple steps are followed:
1. Generate protobuf bindings for the desired language

2. Implement the necessary transforms that manipulate the protobuf to achieve
the desired mapping for nd4j op execution. Remember, a transform is something simple like:
convert tensor to list of ints, convert string to list of ints


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
