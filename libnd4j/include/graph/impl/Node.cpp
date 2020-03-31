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
// @author raver119@gmail.com
//

#include <graph/Node.h>
#include <ops/declarable/OpRegistrator.h>
#include <ops/declarable/LegacyTransformSameOp.h>
#include <ops/declarable/LegacyTransformFloatOp.h>
#include <ops/declarable/LegacyScalarOp.h>
#include <ops/declarable/LegacyReduceSameOp.h>
#include <ops/declarable/LegacyReduceFloatOp.h>
#include <ops/declarable/LegacyIndexReduceOp.h>
#include <ops/declarable/LegacyStatsOp.h>
#include <ops/declarable/LegacyBroadcastOp.h>
#include <ops/declarable/LegacyReduce3Op.h>
#include <ops/declarable/LegacyPairwiseTransformOp.h>
#include <ops/declarable/LegacyRandomOp.h>
#include <ops/declarable/LegacyOp.h>
#include <ops/declarable/LegacyReduceLongOp.h>
#include <ops/declarable/LegacyReduceBoolOp.h>
#include <ops/declarable/LegacyBroadcastBoolOp.h>
#include <ops/declarable/LegacyScalarBoolOp.h>
#include <ops/declarable/LegacyPairwiseTransformBoolOp.h>
#include <ops/declarable/LegacyTransformStrictOp.h>
#include <ops/declarable/LegacyTransformBoolOp.h>
#include <graph/FlatUtils.h>
#include <array/NDArrayFactory.h>

namespace sd {
    namespace graph {
        Node::Node(const std::string &nodeName, const ops::DeclarableOp &opName, const std::vector<double> &tArgs,
                   const std::vector<Nd4jLong> &iArgs, const std::vector<bool> &bArgs,
                   const std::vector<DataType> &dArgs) {
            auto customOp = ops::OpRegistrator::getInstance()->getOperation(opName.getOpHash());

            this->_name = nodeName;
            this->_opType = OpType_CUSTOM;
            this->_opNum = customOp->getOpHash();
            this->_extraParams = nullptr;
            this->_dataType = sd::DataType::FLOAT32; // float as default
            this->_dim = nullptr;
            this->_customOp = customOp;

            _hasExternalInputs = false;
            _hasExternalOutputs = false;
            _hasInternalInputs = false;
            _hasInternalOutputs = false;

            // FIXME: get rid of this!!!
            _scalar = NDArrayFactory::create<int>(0);

            ContextPrototype block(this->customOp()->getOpDescriptor(), this->id(), false);

            block.appendI(iArgs);
            block.appendT(tArgs);
            block.appendB(bArgs);
            block.appendD(dArgs);

            this->setContextPrototype(block);
        }

        Node::Node(const std::string &nodeName, const std::string &opName, const std::vector<double> &tArgs,
                   const std::vector<Nd4jLong> &iArgs, const std::vector<bool> &bArgs,
                   const std::vector<DataType> &dArgs) {

            auto customOp = ops::OpRegistrator::getInstance()->getOperation(opName);

            this->_name = nodeName;
            this->_opType = OpType_CUSTOM;
            this->_opNum = customOp->getOpHash();
            this->_extraParams = nullptr;
            this->_dataType = sd::DataType::FLOAT32; // float as default
            this->_dim = nullptr;
            this->_customOp = customOp;

            _hasExternalInputs = false;
            _hasExternalOutputs = false;
            _hasInternalInputs = false;
            _hasInternalOutputs = false;

            // FIXME: get rid of this!!!
            _scalar = NDArrayFactory::create<int>(0);

            ContextPrototype block(this->customOp()->getOpDescriptor(), this->id(), false);

            block.appendI(iArgs);
            block.appendT(tArgs);
            block.appendB(bArgs);
            block.appendD(dArgs);

            this->setContextPrototype(block);
        }

        void Node::setOuterTime(Nd4jLong time){
//            if (hasBlockAttached())
//                _block->setOuterTime(time);
        }

        void Node::setInnerTime(Nd4jLong time){
//            if (hasBlockAttached())
//                _block->setInnerTime(time);
        }

        void Node::setGraph(Graph* graph) {
            _graph = graph;
        }

        Graph* Node::graph() const {
            return _graph;
        }

        void Node::markInplace(bool reallyInplace) {
            _isInplace = reallyInplace;
            _protoContext.markInplace(reallyInplace);
        }

        bool Node::isRemovable() const {
            return _removable;
        }

        void Node::markRemovable(bool reallyRemovable) const {
            _removable = reallyRemovable;
        }

        OpClass Node::getOpClass() {
            return _opClass;
        }

        bool Node::hasBlockAttached() {
            return true;
        }

        bool Node::isInplace() {
            return _isInplace;
        }

        bool Node::isDivergencePoint() {
            if (hasCustomOp()) {
                return _customOp->getOpDescriptor()->isDivergent();
            } else if (opType() == OpType_LOGIC && opNum() == 30)
                return true;
            else
                return false;
        }

        void Node::setActive(bool reallyActive) {
            _active = reallyActive;
        }

        bool Node::isActive() {
            return _active;
        }

        Nd4jLong Node::getFrameId() {
            return _frameId;
        }

        void Node::setFrameId(Nd4jLong frameId) {
            _frameId = frameId;
        }

        const ContextPrototype& Node::contextPrototype() const {
            return _protoContext;
        }

        void Node::setContextPrototype(const ContextPrototype &block) {
            _protoContext = block;
        }

        void Node::setId(int id) {
            _id = id;
        }

        sd::ops::DeclarableOp* Node::customOp() const {
            return _customOp;
        }

        void Node::setCustomOp(sd::ops::DeclarableOp *customOp) {
            _customOp = customOp;

            // divergent ops (Switch etc) are always inplace, they don't allocate anything
            if (_customOp != nullptr && customOp->getOpDescriptor()->isDivergent())
                _isInplace = true;
        }

        bool Node::hasCustomOp() const {
            return _customOp != nullptr;
        }

        const std::string & Node::name() const {
            return this->getName();
        }

        const std::string & Node::getName() const {
            return _name;
        }

        void Node::setName(const std::string& name) {
            _name = name;
        }

        void Node::setName(std::string *name) {
            _name = *name;
        }

        double Node::scalar() {
            return  _scalar.e<double>(0);
        };

        void Node::pickInput(std::pair<int,int>& pair) {
            _input.push_back(pair);
        }

        void Node::pickInput(const std::string &id) {
            throw std::runtime_error("Node::pickInput - Not implemented yet");
        }

        void Node::pickInput(int inputId, int outputId) {
            std::pair<int,int> p(inputId,outputId);
            pickInput(p);
        }

        void Node::pickInput(int inputId) {
            pickInput(inputId, 0);

            if (inputId < 0)
                _hasExternalInputs = true;
            else
                _hasInternalInputs = true;
        }

        void Node::pickExternalOutput(int outputId) {
            std::pair<int, int> pair(outputId, 0);
            _output.push_back(pair);

            _hasExternalOutputs = true;
        }

        void Node::pickOutputOnce(int outputId) {
            std::pair<int, int> pair(outputId, 0);
            if (std::find(_output.begin(), _output.end(), pair) == _output.end())
                pickOutput(outputId);
        }

        void Node::pickOutput(int nodeId, int outputId) {
            std::pair<int, int> pair(nodeId, outputId);
            _output.emplace_back(pair);
        }

        void Node::pickOutput(int outputId) {
            std::pair<int, int> pair(outputId, 0);
            _output.emplace_back(pair);

            if (outputId < 0)
                _hasExternalOutputs = true;
            else
                _hasInternalOutputs = true;
        }

        int * Node::getDimensionsPtr() {
            return _dim;
        }

        std::vector<int> * Node::getDimensions() {
            return &_dimensions;
        }

        int Node::getLayer() {
            return _layer;
        }

        void Node::setLayer(int layer) {
            _layer = layer;
        }

        bool Node::hasExternalOutputs() {
            return _hasExternalOutputs;
        }

        bool Node::hasExternalInputs() {
            return _hasExternalInputs;
        }

        bool Node::hasInternalOutputs() {
            return _hasInternalOutputs;
        }

        bool Node::hasInternalInputs() {
            return _hasInternalInputs;
        }

        bool Node::isMultiInput() {
            return _input.size() > 1;
        }

        bool Node::isMultiOutput() {
            return _output.size() > 1;
        }

        double * Node::extraParams() {
            return _extraParams;
        }

        int Node::totalReferences() {
            return _referencedBy.size();
        }

        void Node::addReference(int nodeId) {
            _referencedBy.emplace_back(nodeId);
        }

        OpType Node::opType() const {
            return _opType;
        }

        int Node::id() const {
            return _id;
        }

        Nd4jLong Node::opNum() const {
            return _opNum;
        }

        const std::vector<std::pair<int,int>>& Node::input() const {
            return _input;
        }

        const std::vector<std::pair<int, int>>& Node::output() const {
            return _output;
        }

        bool Node::isScoped() {
            return _scope_id != 0;
        }

        void Node::setScopeInfo(int id, const char* name) {
            _scope_id = id;

            if (name != nullptr)
                _scope_name = name;
        }

        int Node::scopeId() {
            return _scope_id;
        }

        std::string* Node::scopeName() {
            return &_scope_name;
        }

        template <typename T>
        Node* Node::asT() {
            auto node = this->clone();
            node->_dataType = DataTypeUtils::fromT<T>();
            return node;
        }
        BUILD_SINGLE_TEMPLATE(template SD_EXPORT Node* Node::asT, (), LIBND4J_TYPES);

        Node::Node(const std::string &opName, const std::string &nodeName, const int id, const std::vector<std::string> &inputs, const std::vector<double> &tArgs, const std::vector<Nd4jLong> &iArgs) {
            auto customOp = ops::OpRegistrator::getInstance()->getOperation(opName);

            this->_opType = OpType_CUSTOM;
            this->_id = id;
            this->_opNum = customOp->getOpHash();
            this->_extraParams = nullptr;
            this->_dataType = sd::DataType::FLOAT32; // float as default
            this->_dim = nullptr;
            this->_customOp = customOp;

            _hasExternalInputs = false;
            _hasExternalOutputs = false;
            _hasInternalInputs = false;
            _hasInternalOutputs = false;

            // FIXME: get rid of this!!!
            _scalar = NDArrayFactory::create<int>(0);

            for (auto i: inputs)
                pickInput(i);

            ContextPrototype block(this->customOp()->getOpDescriptor(), this->id(), false);

            block.appendI(iArgs);
            block.appendT(tArgs);

            this->setContextPrototype(block);
        }

        Node::Node(const std::string &opName, const int id, const std::vector<std::pair<int, int>> &inputs, const std::vector<double> &tArgs, const std::vector<Nd4jLong> &iArgs) {
            auto customOp = ops::OpRegistrator::getInstance()->getOperation(opName);

            this->_opType = OpType_CUSTOM;
            this->_id = id;
            this->_opNum = customOp->getOpHash();
            this->_extraParams = nullptr;
            this->_dataType = sd::DataType::FLOAT32; // float as default
            this->_dim = nullptr;
            this->_customOp = customOp;

            _hasExternalInputs = false;
            _hasExternalOutputs = false;
            _hasInternalInputs = false;
            _hasInternalOutputs = false;

            // FIXME: get rid of this!!!
            _scalar = NDArrayFactory::create<int>(0);

            for (auto i: inputs)
                pickInput(i);

            ContextPrototype block(this->customOp()->getOpDescriptor(), this->id(), false);

            block.appendI(iArgs);
            block.appendT(tArgs);

            this->setContextPrototype(block);
        }

        Node::Node(sd::ops::DeclarableOp *customOp, int id, std::initializer_list<int> input, std::initializer_list<int> output,  std::initializer_list<int> dimensions, float scalar, std::initializer_list<double> tArgs, std::initializer_list<int> iArgs) {
            this->_opType = OpType_CUSTOM;
            this->_id = id;
            this->_opNum = customOp->getOpHash();
            this->_extraParams = nullptr;
            this->_dataType = sd::DataType::FLOAT32; // float as default
            this->_dim = nullptr;
            this->_customOp = customOp;

            _hasExternalInputs = false;
            _hasExternalOutputs = false;
            _hasInternalInputs = false;
            _hasInternalOutputs = false;

            // FIXME: get rid of this!!!
            _scalar = NDArrayFactory::create(scalar);

            for (auto i: input)
                pickInput(i);

            for (auto o: output)
                pickOutput(o);

            if (dimensions.size() > 0) {
                _dim = new int[dimensions.size()];
                int cnt = 0;
                for (auto d: dimensions) {
                    _dimensions.push_back(d);
                    _dim[cnt++] = d;
                }
            }

            ContextPrototype block(this->customOp()->getOpDescriptor(), this->id(), false);

            for (auto v: dimensions)
                block.appendA(v);

            for (auto v: iArgs)
                block.appendI(v);

            for (auto v: tArgs)
                block.appendT(v);

            this->setContextPrototype(block);
        }

        void Node::setOpType(OpType opType) {
            this->_opType = opType;
        }

        Node::Node(OpType opType, int opNum, int id, std::initializer_list<int> input, std::initializer_list<int> output, std::initializer_list<int> dimensions, float scalar, std::initializer_list<double> tArgs, std::initializer_list<int> iArgs) {
            this->_opType = opType;
            this->_id = id;
            this->_opNum = opNum;
            this->_extraParams = nullptr;
            this->_dataType = sd::DataType::FLOAT32; // float as default
            this->_dim = nullptr;

            _hasExternalInputs = false;
            _hasExternalOutputs = false;
            _hasInternalInputs = false;
            _hasInternalOutputs = false;

            _scalar = NDArrayFactory::create(scalar);

            for (auto i: input)
                pickInput(i);

            for (auto o: output)
                pickOutput(o);

            if (dimensions.size() > 0) {
                _dim = new int[dimensions.size()];
                int cnt = 0;
                for (auto d: dimensions) {
                    _dimensions.push_back(d);
                    _dim[cnt++] = d;
                }
            }

            // these ops allow in-place execution by design
            if (opType == OpType_TRANSFORM_SAME || opType == OpType_TRANSFORM_FLOAT || opType == OpType_TRANSFORM_STRICT || opType == OpType_TRANSFORM_BOOL || opType == OpType_SCALAR || opType == OpType_BROADCAST) {
                if (_output.size() <= 1) {
                    _isInplace = true;
                }
                _opClass = OpClass_TRANSFORM;
            } else if (opType == OpType_REDUCE_SAME || opType == OpType_REDUCE_FLOAT || opType == OpType_REDUCE_BOOL || opType == OpType_REDUCE_LONG || opType == OpType_SUMMARYSTATS) {
                _opClass = OpClass_REDUCTION;
            }


            if (opType == OpType_BROADCAST ||
                    opType == OpType_BROADCAST_BOOL ||
                    opType == OpType_INDEX_REDUCE ||
                    opType == OpType_SUMMARYSTATS ||
                    opType == OpType_REDUCE_BOOL ||
                    opType == OpType_REDUCE_SAME ||
                    opType == OpType_REDUCE_FLOAT ||
                    opType == OpType_REDUCE_3 ||
                    opType == OpType_TRANSFORM_STRICT ||
                    opType == OpType_TRANSFORM_SAME ||
                    opType == OpType_TRANSFORM_FLOAT ||
                    opType == OpType_TRANSFORM_BOOL ||
                    opType == OpType_RANDOM ||
                    opType == OpType_PAIRWISE ||
                    opType == OpType_PAIRWISE_BOOL ||
                    opType == OpType_SCALAR_BOOL ||
                    opType == OpType_SCALAR) {

                this->_isDeductable = true;

                ContextPrototype block(nullptr, this->id(), false);

                for (auto v: dimensions)
                    block.appendA(v);

                for (auto v: iArgs)
                    block.appendI(v);

                for (auto v: tArgs)
                    block.appendT(v);

                this->setContextPrototype(block);

                this->setCustomOp(Node::buildOpByType(opType, (int) input.size(), (int) block.getIArguments().size(), (int) block.getTArguments().size(), opNum, &_scalar));
                block.setOpDescriptor(this->customOp()->getOpDescriptor());
            } else if (opType == OpType_CUSTOM) {
                if (this->customOp()) {
                    ContextPrototype block(this->customOp()->getOpDescriptor(), this->id(), false);

                    for (auto v: dimensions)
                        block.appendA(v);

                    for (auto v: iArgs)
                        block.appendI(v);

                    for (auto v: tArgs)
                        block.appendT(v);

                    this->setContextPrototype(block);
                } else throw std::runtime_error("wrong custom operation given");
            }
        };

        Node::Node(const FlatNode *node) {
            _hasExternalInputs = false;
            _hasExternalOutputs = false;
            _hasInternalInputs = false;
            _hasInternalOutputs = false;
            _extraParams = nullptr;
            _dim = nullptr;
            _dataType = sd::DataType::FLOAT32; // float as default
            if (node->scope_id() != 0)
                this->_scope_id = node->scope_id();

            if (node->scope_name() != nullptr && node->scope_name()->size() > 0)
                this->_scope_name = node->scope_name()->str();

            if (node->scalar() != nullptr) {
                auto scalar = FlatUtils::fromFlatArray(node->scalar());
                _scalar = *scalar;
                delete scalar;
            }

            if (node != nullptr) {
                this->_id = node->id();
                //this->_dataType = DataTypeUtils::fromFlatDataType(node->dataType());
                this->_opNum = node->opNum();
                this->_opType = node->opType();

                if (node->name() != nullptr && node->name()->c_str() != nullptr) {
                    this->_name = node->name()->str();
                }

                if (node->inputPaired() != nullptr && node->inputPaired()->size() > 0) {
                    for (int e = 0; e < (int) node->inputPaired()->size(); e++) {
                        auto pair = node->inputPaired()->Get(e);
                        pickInput(pair->first(), pair->second());
                    }
                } else if (node->input() != nullptr && node->input()->size() > 0) {
                    for (int e = 0; e < (int) node->input()->size(); e++)
                        pickInput(node->input()->Get(e));
                } else {
                    if (this->opType() != OpType_LOGIC) {
                        if (this->_name.size() > 0) {
                            nd4j_debug("Node [%i:<%s>] has no inputs defined\n", this->_id, this->_name.c_str());
                        } else {
                            nd4j_debug("Node [%i:<noname>] has no inputs defined\n", this->_id);
                        }
                    }
                }

                /*
                if (node->output() != nullptr)
                    for (int e = 0; e < (int) node->output()->size(); e++) {
                        auto oid = node->output()->Get(e);
                        if (oid != this->_id && oid != 0) {
                            nd4j_verbose("Picking output: %i\n", node->output()->Get(e));
                            pickOutput(oid);
                        }
                    }
                */


                if (node->extraParams() != nullptr && node->extraParams()->size() > 0) {
                    _extraParams = new double[node->extraParams()->size()];
                    for (int e = 0; e < (int) node->extraParams()->size(); e++) {
                        _extraParams[e] = static_cast<double>(node->extraParams()->Get(e));
                    }
                }

                if (node->dimensions() != nullptr && node->dimensions()->size() > 0) {
                    _dim = new int[node->dimensions()->size()];
                    for (int e = 0; e < (int) node->dimensions()->size(); e++) {
                        _dimensions.emplace_back(node->dimensions()->Get(e));
                        _dim[e] = node->dimensions()->Get(e);
                    }
                }

                if (this->opType() == OpType_LOGIC && this->opNum() == 100L) {
                    if (node->extraInteger()->size() < 1) {
                        nd4j_printf("Node_%i is type of Enter, but has no FrameID defined\n", this->id());
                        throw std::runtime_error("Enter node must have FrameID specified");
                    }

                    this->setFrameId(node->extraInteger()->Get(0));
                }


                // these ops allow in-place execution by design
                if (_opType == OpType_BROADCAST ||
                    _opType == OpType_BROADCAST_BOOL ||
                        _opType == OpType_INDEX_REDUCE ||
                        _opType == OpType_SUMMARYSTATS ||
                        _opType == OpType_REDUCE_BOOL ||
                        _opType == OpType_REDUCE_SAME ||
                        _opType == OpType_REDUCE_FLOAT ||
                        _opType == OpType_REDUCE_3 ||
                        _opType == OpType_TRANSFORM_STRICT ||
                        _opType == OpType_TRANSFORM_SAME ||
                        _opType == OpType_TRANSFORM_FLOAT ||
                        _opType == OpType_TRANSFORM_BOOL ||
                        _opType == OpType_RANDOM ||
                        _opType == OpType_PAIRWISE ||
                        _opType == OpType_PAIRWISE_BOOL ||
                        _opType == OpType_SCALAR_BOOL ||
                        _opType == OpType_SCALAR) {

                    if (_output.size() <= 1) {
                        _isInplace = true;
                    }

                    if (node->input() != nullptr && node->input()->size() > 0) {
                        this->_isDeductable = true;

                        ContextPrototype block(nullptr, this->id(), false);


                        for (auto v: _dimensions)
                            block.appendA(v);

                        if (node->extraParams() != nullptr && node->extraParams()->size() > 0)
                            for (int e = 0; e < (int) node->extraParams()->size(); e++) {
                                block.appendT(static_cast<double>(node->extraParams()->Get(e)));
                            }

                        if (node->extraBools() != nullptr && node->extraBools()->size() > 0)
                            for (int e = 0; e < (int) node->extraBools()->size(); e++) {
                                block.appendB(node->extraBools()->Get(e));
                            }

                        if (node->extraInteger() != nullptr && node->extraInteger()->size() > 0)
                            for (int e = 0; e < (int) node->extraInteger()->size(); e++) {
                                block.appendI(node->extraInteger()->Get(e));
                            }

                        if (node->extraTypes() != nullptr && node->extraTypes()->size() > 0) {
                            for (int e = 0; e < (int) node->extraTypes()->size(); e++) {
                                block.appendD((sd::DataType) node->extraTypes()->Get(e));
                            }
                        }

                        this->setContextPrototype(block);
                        this->setCustomOp(Node::buildOpByType(_opType, (int) node->input()->size(), (int) block.getIArguments().size(), (int) block.getTArguments().size(), (int) _opNum, &_scalar));
                        block.setOpDescriptor(this->customOp()->getOpDescriptor());
                    } else if (node->inputPaired() != nullptr && node->inputPaired()->size() > 0) {
                        this->_isDeductable = true;

                        ContextPrototype block(nullptr, this->id(), false);

                        for (int e = 0; e < this->input().size(); e++) {
                            block.inputs().emplace_back(this->input().at(e));
                        }

                        // there's no other IArgs in legacy options, actually
                        for (auto v: _dimensions)
                            block.appendA(v);

                        if (node->extraParams() != nullptr && node->extraParams()->size() > 0)
                            for (int e = 0; e < (int) node->extraParams()->size(); e++) {
                                block.appendT(static_cast<double>(node->extraParams()->Get(e)));
                            }

                        if (node->extraBools() != nullptr && node->extraBools()->size() > 0)
                            for (int e = 0; e < (int) node->extraBools()->size(); e++) {
                                block.appendB(node->extraBools()->Get(e));
                            }

                        if (node->extraInteger() != nullptr && node->extraInteger()->size() > 0)
                            for (int e = 0; e < (int) node->extraInteger()->size(); e++) {
                                block.appendI(node->extraInteger()->Get(e));
                            }

                        if (node->extraTypes() != nullptr && node->extraTypes()->size() > 0) {
                            for (int e = 0; e < (int) node->extraTypes()->size(); e++) {
                                block.appendD((sd::DataType) node->extraTypes()->Get(e));
                            }
                        }

                        this->setContextPrototype(block);

                        this->setCustomOp(Node::buildOpByType(_opType, (int) node->inputPaired()->size(), (int) block.getIArguments().size(), (int) block.getTArguments().size(), (int) _opNum, &_scalar));
                        block.setOpDescriptor(this->customOp()->getOpDescriptor());
                    }
                } else if (this->_opType == OpType_CUSTOM) {
                        auto op = sd::ops::OpRegistrator::getInstance()->getOperation(this->opNum());
                        if (op == nullptr) {
                            nd4j_verbose("Can't find operation: %lld\n", this->opNum());
                            throw std::runtime_error("Can't find requested operation");
                        }

                        ContextPrototype block(nullptr, this->id());

                        for (int e = 0; e < this->input().size(); e++) {
                            block.inputs().emplace_back(this->input().at(e));
                        }

                        if (node->extraInteger() != nullptr)
                            for (uint32_t e = 0; e < node->extraInteger()->size(); e++) {
                                auto v = node->extraInteger()->Get(e);
                                // FIXME: remove this static_cast, iArgs should be Nd4jLong
                                block.appendI(static_cast<int>(v));
                            }

                        if (node->extraParams() != nullptr)
                            for (uint32_t e = 0; e < node->extraParams()->size(); e++)
                                block.appendT(static_cast<double>(node->extraParams()->Get(e)));

                        if (node->extraBools() != nullptr && node->extraBools()->size() > 0)
                            for (int e = 0; e < (int) node->extraBools()->size(); e++) {
                                block.appendB(node->extraBools()->Get(e));
                            }

                        if (node->extraTypes() != nullptr && node->extraTypes()->size() > 0) {
                            for (int e = 0; e < (int) node->extraTypes()->size(); e++) {
                                block.appendD((sd::DataType) node->extraTypes()->Get(e));
                            }
                        }

                        for (auto v: _dimensions)
                            block.appendA(v);

                        this->setContextPrototype(block);
                        this->setCustomOp(op);
                        block.setOpDescriptor(this->customOp()->getOpDescriptor());
                }
            } else {
                // empty dynamic node, tests probably
            }
        }

        sd::DataType Node::dataType() {
            return _dataType;
        }

        const ContextPrototype& Node::protoContext() const {
            return _protoContext;
        }

        Node::~Node() {
            if (_extraParams != nullptr)
                delete[] _extraParams;

            if (_dim != nullptr)
                delete[] _dim;

            if (_isDeductable && _customOp != nullptr) {
                Node::deleteOpByType(_opType, _customOp);
            }
        }

        int Node::getRewindNode() {
            return _rewindNode;
        }

        void Node::setRewindNode(int nodeId) {
            _rewindNode = nodeId;
        }

        std::pair<int, int>& Node::getRewindLayer() {
            return _rewindLayer;
        };

        void Node::setRewindLayer(int layerId, int stepId) {
            _rewindLayer.first = layerId;
            _rewindLayer.second = stepId;
        }

        bool Node::equals(Node *other) const {
            if (_opType == other->_opType && _dataType == other->_dataType && _opNum == other->_opNum)
                return true;

            return false;
        }

        Node::Node(const Node &other) noexcept {

        }

        Node &Node::operator=(const Node &other) noexcept {
            if (this == &other)
                return *this;

            _dataType = other._dataType;
            _opType = other._opType;
            _opClass = other._opClass;
            _opNum = other._opNum;
            _customOp = other._customOp;
            _name = other._name;
            _scope_id = other._scope_id;
            _scope_name = other._scope_name;
            _rewindNode = other._rewindNode;
            _layer = other._layer;

            _hasExternalOutputs = other._hasExternalOutputs;
            _hasExternalInputs = other._hasExternalInputs;
            _hasInternalOutputs = other._hasInternalOutputs;
            _hasInternalInputs = other._hasInternalInputs;
            _isInplace = other._isInplace;
            _isDeductable = other._isDeductable;
            _active = other._active;
            _removable = other._removable;

            _graph = other._graph;
            _customOp = other._customOp;
            _dim = other._dim;
            _extraParams = other._extraParams;
            _protoContext = other._protoContext;

            _input = other._input;
            _output = other._output;
            _dimensions = other._dimensions;
            _rewindLayer = other._rewindLayer;
            _referencedBy = other._referencedBy;
            _scalar = other._scalar;

            return *this;
        }

        Node::Node(Node &&other) noexcept {

        }

        Node &Node::operator=(Node &&other) noexcept {
            if (this == &other)
                return *this;

            _dataType = other._dataType;
            _opType = other._opType;
            _opClass = other._opClass;
            _opNum = other._opNum;
            _customOp = other._customOp;
            _scope_id = other._scope_id;
            _name = std::move(other._name);
            _scope_name = std::move(other._scope_name);
            _rewindNode = other._rewindNode;
            _layer = other._layer;

            _hasExternalOutputs = other._hasExternalOutputs;
            _hasExternalInputs = other._hasExternalInputs;
            _hasInternalOutputs = other._hasInternalOutputs;
            _hasInternalInputs = other._hasInternalInputs;
            _isInplace = other._isInplace;
            _isDeductable = other._isDeductable;
            _active = other._active;
            _removable = other._removable;

            _graph = other._graph;
            _customOp = other._customOp;
            _dim = other._dim;
            _extraParams = other._extraParams;
            _protoContext = other._protoContext;

            _input = std::move(other._input);
            _output = std::move(other._output);
            _dimensions = std::move(other._dimensions);
            _rewindLayer = std::move(other._rewindLayer);
            _referencedBy = std::move(other._referencedBy);
            _scalar = std::move(other._scalar);

            other._customOp = nullptr;

            return *this;
        }

        void Node::deleteOpByType(OpType opType, void *op) {
            switch (opType) {
                case OpType_PAIRWISE:
                    delete reinterpret_cast<sd::ops::LegacyPairwiseTransformOp*>(op);
                    break;
                case OpType_PAIRWISE_BOOL:
                    delete reinterpret_cast<sd::ops::LegacyPairwiseTransformBoolOp*>(op);
                    break;
                case OpType_TRANSFORM_STRICT:
                    delete reinterpret_cast<sd::ops::LegacyTransformStrictOp*>(op);
                    break;
                case OpType_TRANSFORM_SAME:
                    delete reinterpret_cast<sd::ops::LegacyTransformSameOp*>(op);
                    break;
                case OpType_TRANSFORM_FLOAT:
                    delete reinterpret_cast<sd::ops::LegacyTransformFloatOp*>(op);
                    break;
                case OpType_TRANSFORM_BOOL:
                    delete reinterpret_cast<sd::ops::LegacyTransformBoolOp*>(op);
                    break;
                case OpType_SCALAR:
                    delete reinterpret_cast<sd::ops::LegacyScalarOp*>(op);
                    break;
                case OpType_SCALAR_BOOL:
                    delete reinterpret_cast<sd::ops::LegacyScalarBoolOp*>(op);
                    break;
                case OpType_REDUCE_3:
                    delete reinterpret_cast<sd::ops::LegacyReduce3Op*>(op);
                    break;
                case OpType_REDUCE_SAME:
                    delete reinterpret_cast<sd::ops::LegacyReduceSameOp*>(op);
                    break;
                case OpType_REDUCE_FLOAT:
                    delete reinterpret_cast<sd::ops::LegacyReduceFloatOp*>(op);
                    break;
                case OpType_REDUCE_LONG:
                    delete reinterpret_cast<sd::ops::LegacyReduceLongOp*>(op);
                    break;
                case OpType_REDUCE_BOOL:
                    delete reinterpret_cast<sd::ops::LegacyReduceBoolOp*>(op);
                    break;
                case OpType_INDEX_REDUCE:
                    delete reinterpret_cast<sd::ops::LegacyIndexReduceOp*>(op);
                    break;
                case OpType_SUMMARYSTATS:
                    delete reinterpret_cast<sd::ops::LegacyStatsOp*>(op);
                    break;
                case OpType_RANDOM:
                    delete reinterpret_cast<sd::ops::LegacyRandomOp*>(op);
                    break;
                case OpType_BROADCAST:
                    delete reinterpret_cast<sd::ops::LegacyBroadcastOp*>(op);
                    break;
                case OpType_BROADCAST_BOOL:
                    delete reinterpret_cast<sd::ops::LegacyBroadcastBoolOp*>(op);
                    break;
                case OpType_CUSTOM:
                    delete reinterpret_cast<sd::ops::DeclarableOp*>(op);
                    break;
                default:
                    throw std::runtime_error("Bad opType passed in");
            }
        }

        sd::ops::DeclarableOp* Node::buildOpByType(OpType opType, int numInputs,  int numIArgs, int numTArgs, int opNum, NDArray *scalar) {
            switch (opType) {
                case OpType_PAIRWISE:
                    return new sd::ops::LegacyPairwiseTransformOp(opNum);
                case OpType_PAIRWISE_BOOL:
                    return new sd::ops::LegacyPairwiseTransformBoolOp(opNum);
                case OpType_TRANSFORM_STRICT:
                    return new sd::ops::LegacyTransformStrictOp(opNum);
                case OpType_TRANSFORM_SAME:
                    return new sd::ops::LegacyTransformSameOp(opNum);
                case OpType_TRANSFORM_FLOAT:
                    return new sd::ops::LegacyTransformFloatOp(opNum);
                case OpType_TRANSFORM_BOOL:
                    return new sd::ops::LegacyTransformBoolOp(opNum);
                case OpType_SCALAR:
                    return scalar == nullptr ? new sd::ops::LegacyScalarOp(opNum) : new sd::ops::LegacyScalarOp(opNum, *scalar);
                case OpType_SCALAR_BOOL:
                    return scalar == nullptr ? new sd::ops::LegacyScalarBoolOp(opNum) : new sd::ops::LegacyScalarBoolOp(opNum, *scalar);
                case OpType_REDUCE_3:
                    return new sd::ops::LegacyReduce3Op(opNum);
                case OpType_REDUCE_SAME:
                    return new sd::ops::LegacyReduceSameOp(opNum);
                case OpType_REDUCE_FLOAT:
                    return new sd::ops::LegacyReduceFloatOp(opNum);
                case OpType_REDUCE_LONG:
                    return new sd::ops::LegacyReduceLongOp(opNum);
                case OpType_REDUCE_BOOL:
                    return new sd::ops::LegacyReduceBoolOp(opNum);
                case OpType_INDEX_REDUCE:
                    return new sd::ops::LegacyIndexReduceOp(opNum);
                case OpType_SUMMARYSTATS:
                    return new sd::ops::LegacyStatsOp(opNum);
                case OpType_RANDOM:
                    return new sd::ops::LegacyRandomOp(opNum);
                case OpType_BROADCAST:
                    return new sd::ops::LegacyBroadcastOp(opNum);
                case OpType_BROADCAST_BOOL:
                    return new sd::ops::LegacyBroadcastBoolOp(opNum);
                default:
                    throw std::runtime_error("Bad opType passed in");
            }
        }

        bool Node::isDeductable() {
            return _isDeductable;
        }

        void Node::setDeductable(bool reallyDeductable) {
            _isDeductable = reallyDeductable;
        }


        Node* Node::clone() {
            if (this->_customOp && this->_opType == OpType_CUSTOM) {
                auto clone = new Node(this->_customOp, _id);
                clone->pullValues(this);
                return clone;
            }
            else {
            auto clone = new Node(_opType, _opNum, _id);

            clone->pullValues(this);

            // op time
            if (!_isDeductable)
                clone->_customOp = _customOp;
            else {
                auto c = dynamic_cast<sd::ops::LegacyOp*>(_customOp);
                clone->_customOp = c->clone();
            }

            return clone;
            }
        }
    }
}
