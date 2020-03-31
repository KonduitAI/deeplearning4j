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
//  @author raver119@gmail.com
//

#include <system/pointercast.h>
#include <system/dll.h>
#include <types/float16.h>
#include <graph/ContextPrototype.h>

namespace sd {
    namespace graph {
        ContextPrototype::ContextPrototype(sd::ops::OpDescriptor* opDescriptor, int nodeId, bool inPlace) {
            _nodeId = nodeId;
            _isInplace = inPlace;
            _opDescriptor = opDescriptor;
        }

        void ContextPrototype::pickInput(std::pair<int, int>& p) {
            this->_inputs.emplace_back(p);
        }

        void ContextPrototype::pickInput(int input, int index) {
            std::pair<int, int> pair(input, index);
            pickInput(pair);
        }

        int ContextPrototype::opNum() const {
            return this->_opNum;
        }

        void ContextPrototype::setOpNum(int opNum) {
            this->_opNum = opNum;
        }

        std::vector<std::pair<int, int>> & ContextPrototype::inputs() const {
            return const_cast<std::vector<std::pair<int, int>> &>(_inputs);
        }

        void ContextPrototype::fillInputs(std::vector<int>& inputs) {
            for (int e = 0; e < inputs.size(); e++) {
                auto v = inputs.at(e);
                pickInput(v);
            }
        }

        samediff::Engine ContextPrototype::engine() const {
            return _engine;
        }

        bool ContextPrototype::hasVariablesFilled() const {
            return this->_inputs.size() > 0;
        }

        bool ContextPrototype::isInplace() const {
            return this->_isInplace;
        }

        const std::vector<double> & ContextPrototype::getTArguments() const {
            return const_cast<std::vector<double>&>(_tArgs);
        }

        const std::vector<int> & ContextPrototype::getIArguments() const {
            return const_cast<std::vector<int>&>(_iArgs);
        }

        const std::vector<bool> & ContextPrototype::getBArguments() const {
            return const_cast<std::vector<bool>&>(_bArgs);
        }

        const std::vector<int> & ContextPrototype::getAxis() const {
            return const_cast<std::vector<int>&>(_axis);
        }

        void ContextPrototype::pickInput(int input) {
            std::pair<int, int> pair(input, 0);
            this->_inputs.emplace_back(pair);
        }

        const std::pair<int, int>& ContextPrototype::input(int idx) const {
            return this->_inputs.at(idx);
        }

        void ContextPrototype::fillInputs(std::initializer_list<int> inputs) {
            for (auto v: inputs) {
                pickInput(v);
            }
        }

        int ContextPrototype::nodeId() const {
            return getNodeId();
        }

        sd::DataType ContextPrototype::dataType() const {
            return dataType(0);
        }

        sd::DataType ContextPrototype::dataType(int index) const {
            return _dataType;
        }

        void ContextPrototype::setDataType(int index, sd::DataType type) {
            // if (_outputs->size() == 0)
            _dataType = type;
        }

        size_t ContextPrototype::numT() const {
            return (int) _tArgs.size();
        }

        size_t ContextPrototype::numI() const {
            return (int) _iArgs.size();
        }

        size_t ContextPrototype::numB() const {
            return (int) _bArgs.size();
        }

        int ContextPrototype::getNodeId() const {
            return this->_nodeId;
        }

        /**
         * This method returns number of inputs available in this block
         * @return
         */
        unsigned long ContextPrototype::width() const {
            return this->_inputs.size();
        };

        void ContextPrototype::markInplace(bool reallyInplace) {
            this->_isInplace = reallyInplace;
        }

        template <typename N>
        ContextPrototype* ContextPrototype::asT() {
            auto clone = new ContextPrototype(_opDescriptor, _nodeId, _isInplace);

            return clone;
        }

        void ContextPrototype::setOpDescriptor(sd::ops::OpDescriptor* opDescriptor) {
            _opDescriptor = opDescriptor;
        }

        ContextPrototype* ContextPrototype::clone() {
            auto clone = new ContextPrototype(_opDescriptor, _nodeId, _isInplace);
            clone->_opNum = _opNum;
            
            for (auto v: _inputs)
                clone->_inputs.emplace_back(v);

            for (auto v: _tArgs)
                clone->_tArgs.emplace_back(v);

            for (auto v: _iArgs)
                clone->_iArgs.emplace_back(v);

            return clone;
        }

        const std::vector<sd::DataType> & ContextPrototype::getDArguments() const {
            return const_cast<std::vector<sd::DataType>&>(_dArgs);
        }

        size_t ContextPrototype::numD() const {
            return _dArgs.size();
        }

        void ContextPrototype::appendI(const std::vector<Nd4jLong> &value) {
            for (auto v:value)
                _iArgs.emplace_back(v);
        }

        void ContextPrototype::appendT(const std::vector<double> &value) {
            for (auto v:value)
                _tArgs.emplace_back(v);
        }

        void ContextPrototype::appendB(const std::vector<bool> &value) {
            for (auto v:value)
                _bArgs.emplace_back(v);
        }

        void ContextPrototype::appendD(const std::vector<DataType> &value) {
            for (auto v:value)
                _dArgs.emplace_back(v);
        }

        void ContextPrototype::appendA(Nd4jLong value) {
            _axis.emplace_back(value);
        }

        void ContextPrototype::appendI(Nd4jLong value) {
            _iArgs.emplace_back(value);
        }

        void ContextPrototype::appendT(double value) {
            _tArgs.emplace_back(value);
        }

        void ContextPrototype::appendB(bool value) {
            _bArgs.emplace_back(value);
        }

        void ContextPrototype::appendD(DataType value) {
            _dArgs.emplace_back(value);
        }
    }
}