/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 * Copyright (c) 2020 Konduit K.K.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

//
//  @author raver119@gmail.com
//

#include <graph/logic/LogicNextIteration.h>

namespace sd {
namespace graph {

Nd4jStatus LogicNextIeration::processNode(const Node *node, Stack &stack, const OptimizedGraph& graph) {
  auto &frame = stack.back();

  const auto &inputs = node->inputs();
  auto &varSpace = const_cast<VariableProxy&>(frame.variableProxy());

  REQUIRE_TRUE(inputs.size() == 1, 0, "LoopCond: op must have exactly 1 input1");
  REQUIRE_TRUE(frame.variableProxy().hasVariable(inputs[0]), 0, "LoopCond: input Variable doesn't exist");

  // Propagate Variable
  auto var = varSpace.getVariable(inputs[0]);
  varSpace.putVariable({node->id(), 0}, *var->getNDArray());

  return Status::OK();
}

}  // namespace graph
}  // namespace sd