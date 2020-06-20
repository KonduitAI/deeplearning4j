/*******************************************************************************
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
// @author raver119@gmail.com
//

#ifndef SD_OPSEQUENCE_H
#define SD_OPSEQUENCE_H

#include <graph/execution/ExecutionTask.h>
#include <ops/declarable/DeclarableOp.h>

#include <vector>

namespace sd {
namespace graph {
/**
 * This class represents independent and immutable sequence of operations
 */
class SD_EXPORT OpSequence
    : public std::iterator<std::output_iterator_tag, ExecutionTask> {
  // our internal iterator for OpSequence
  class iterator;

 protected:
  // main thing here. sorted list of operations and their contexts
  std::vector<ExecutionTask> _ops;

  int _deviceId = 0;

  // this map contains Node::id() -> OpSequence index mappings
  MAP_IMPL<int, int> _idToIndex;

  // this map contains OpSequence index -> Node::id() mapping
  MAP_IMPL<int, int> _indexToId;

 public:
  explicit OpSequence(const std::vector<ExecutionTask>& ops, int deviceId = 0);
  OpSequence(int deviceId = 0);
  ~OpSequence() = default;

  OpSequence(const OpSequence& other) noexcept;

  OpSequence& operator=(const OpSequence& other) noexcept;

  // move constructor
  OpSequence(OpSequence&& other) noexcept;

  // move assignment operator
  OpSequence& operator=(OpSequence&& other) noexcept;

  int deviceId() const;

  /**
   * This method blocks until all operations within sequence are processed
   * @return
   */
  Nd4jStatus wait() const;

  /**
   * This method prints out content of the sequence
   */
  void printOut() const;

  /**
   * This method returns number of individual operations within this sequence
   * @return
   */
  uint64_t length() const;

  /**
   * This method returns specific Op/ContextPrototype pair for specified index
   * @param index
   * @return
   */
  const ExecutionTask& at(uint64_t index) const;
  const ExecutionTask& operator[](uint64_t index) const;

  /**
   * This method allows to add DeclarableOp to the end of execution queue
   * @param op - Op to be executed
   * @param ctx - ContextPrototype for this operation with inputs/outputs/args
   * defined
   */
  void append(const Node& node,
              const sd::graph::ContextPrototype& ctx);
  void append(const ExecutionTask& task);
  void append(ExecutionTask&& task);

  /**
   * These two methods provide access to index/id dictionalries
   * @param index
   * @return
   */
  int nodeId(int index) const;
  int nodeIndex(int id) const;
  bool hasNode(int id) const;

  /**
   * Iterator functionality for OpSequence
   * @return
   */

  OpSequence::iterator begin();
  OpSequence::iterator end();

  // additional private section
 private:
  class iterator
      : public std::iterator<std::output_iterator_tag, ExecutionTask> {
   private:
    uint64_t _position = 0;
    OpSequence& _container;

   public:
    explicit iterator(OpSequence& container, uint64_t index = 0);
    const ExecutionTask& operator*() const;
    iterator& operator++();
    iterator& operator++(int);
    bool operator!=(const iterator&) const;
  };
};
}  // namespace graph
}  // namespace sd

#endif  // SD_OPSEQUENCE_H