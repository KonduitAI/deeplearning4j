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
// Created by raver119 on 20.10.2017.
//

#ifndef SD_LOGICEXECUTOR_H
#define SD_LOGICEXECUTOR_H

#include <graph/Graph.h>
#include <graph/Node.h>
#include <graph/OptimizedGraph.h>
#include <system/pointercast.h>
#include <graph/execution/Stack.h>

namespace sd {
namespace graph {
/**
 * This class acts as switch for picking logic execution based on opNum, unique
 * for each logical op
 * @tparam T
 */
class LogicExecutor {
 public:
  static Nd4jStatus processNode(const Node* node, Stack &stack, const OptimizedGraph& graph);
};
}  // namespace graph
}  // namespace sd

#endif  // SD_LOGICEXECUTOR_H