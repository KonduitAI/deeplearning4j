# Subgraph Scanning

## Status
Proposed

Proposed by: Adam Gibson (28-09-2020)

Discussed with: N/A

## Context
 

## Decision

Extending from [interpreter scanning](./0006-Interpreter-Scanning.md)
a common use case for import (especially keras)
are custom defined layers that define a sequence of operations
that usually existing in nd4j but may not be explicitly supported in the framework.

These components are usually registered as a custom layer.
Our analog to this is defining a graph mapping.

This allows us to define named sub graphs that act as layers.

This abstraction allows us to write tooling for mapping custom named subgraphs
as sequences of operations. These can be discovered at runtime dynamically with package scanning.

A graph mapping annotation looks as follows:
```java
@GraphMapping(frameworkName = "keras",opList = {""},name = "layerName",frameworkLayerName = "layerNameInFramework")
```
The opList is the list of ops present in the name.
The frameworkName covers the name of the framework being imported (used for discovering mappers).

The frameworkLayerName is the registered layer name in the framework being imported.


## Consequences
### Advantages
* Allows a way to bypass 1 to 1 mappings

* Allows import of custom layers in any framework

* Easy definition by the user rather than having to setup a lambda layer.

### Disadvantages

* Maintenance risks: Edge cases may come up.

* Dependency on other concepts makes maintenance of this concept in isolation hard
