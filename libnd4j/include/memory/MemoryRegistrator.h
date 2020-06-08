/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
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
// Created by raver119 on 12.09.17.
//

#ifndef SD_MEMORYREGISTRATOR_H
#define SD_MEMORYREGISTRATOR_H

#include <system/dll.h>
#include <system/op_boilerplate.h>

#include <map>
#include <mutex>
#include <unordered_map>

#include "Workspace.h"

namespace sd {
namespace memory {
class SD_EXPORT MemoryRegistrator {
 protected:

  Workspace* _workspace;
  MAP_IMPL<Nd4jLong, Nd4jLong> _footprint;
  std::mutex _lock;

  MemoryRegistrator();
  ~MemoryRegistrator() = default;

 public:
  static MemoryRegistrator& getInstance();
  bool hasWorkspaceAttached();
  Workspace* getWorkspace();
  void attachWorkspace(Workspace* workspace);
  void forgetWorkspace();

  /**
   * This method allows you to set memory requirements for given graph
   */
  void setGraphMemoryFootprint(Nd4jLong hash, Nd4jLong bytes);

  /**
   * This method allows you to set memory requirements for given graph, ONLY if
   * new amount of bytes is greater then current one
   */
  void setGraphMemoryFootprintIfGreater(Nd4jLong hash, Nd4jLong bytes);

  /**
   * This method returns memory requirements for given graph
   */
  Nd4jLong getGraphMemoryFootprint(Nd4jLong hash);
};
}  // namespace memory
}  // namespace sd

#endif  // SD_MEMORYREGISTRATOR_H
