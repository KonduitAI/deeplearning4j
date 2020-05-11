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
// Created by raver119 on 16.10.2017.
//

#include <ops/declarable/LegacyReduceLongOp.h>
#include <helpers/TAD.h>
#include <helpers/ShapeUtils.h>
#include <graph/Status.h>
#include <helpers/ConstantTadHelper.h>
#include <array/DataTypeUtils.h>

namespace sd {
    namespace ops {
        LegacyReduceLongOp::LegacyReduceLongOp() : LegacyOp::LegacyOp(1) {
            //
        }

        LegacyReduceLongOp::LegacyReduceLongOp(int opNum) : LegacyOp::LegacyOp(1, opNum) {
            //this->_opNum = opNum;
        }

        LegacyOp* LegacyReduceLongOp::clone() {
            return new LegacyReduceLongOp(this->_opNum);
        }

        Nd4jStatus LegacyReduceLongOp::validateAndExecute(Context &block) {
            auto x = INPUT_VARIABLE(0);

            auto z = OUTPUT_VARIABLE(0);

            NDArray::prepareSpecialUse({z}, {x});

            int opNum = block.opNum() < 0 ? this->_opNum : block.opNum();
            nd4j_debug("Executing LegacyReduceFloatOp: [%i]\n", opNum);

            auto axis = block.getAxis();
            bool allAxes = false;

            ExtraArguments extras(block.getTArguments());
            PointersManager manager(block.launchContext(),"LegacyReduceLongOp");

            if (block.width() == 1) {

                if (axis.size() == x->rankOf())
                    allAxes = true;

                if ((axis.empty()) ||
                    (axis.size() == 1 && axis[0] == sd::DataTypeUtils::max<int>()) || allAxes) {
                    // scalar
                    NativeOpExecutioner::execReduceLongScalar(block.launchContext(), opNum, x->buffer(), x->shapeInfo(), x->specialBuffer(), x->specialShapeInfo(),
                            extras.argumentsAsT(x->dataType()), z->buffer(), z->shapeInfo(), z->specialBuffer(), z->specialShapeInfo());
                } else {
                    // TAD
                    std::vector<int> dims(axis);

                    for (int e = 0; e < dims.size(); e++)
                        if (dims[e] < 0)
                            dims[e] += x->rankOf();

                    if (dims.size() > 1)
                        std::sort(dims.begin(), dims.end());

                    REQUIRE_TRUE(dims.size() > 0, 0, "Some dimensions required for reduction!");

                    auto packX = sd::ConstantTadHelper::getInstance()->tadForDimensions(x->shapeInfo(), dims);

                    auto pTadShape = Environment::getInstance()->isCPU() ? packX.primaryShapeInfo() : packX.specialShapeInfo(); //(Nd4jLong *) manager.replicatePointer(tad.tadOnlyShapeInfo, shape::shapeInfoByteLength(tad.tadOnlyShapeInfo));
                    auto pTadOffsets = Environment::getInstance()->isCPU() ? packX.primaryOffsets() : packX.specialOffsets(); //(Nd4jLong *) manager.replicatePointer(tad.tadOffsets, tad.numTads * sizeof(Nd4jLong));

                    NativeOpExecutioner::execReduceLong(block.launchContext(), opNum, x->buffer(), x->shapeInfo(), x->specialBuffer(), x->specialShapeInfo(),
                            extras.argumentsAsT(x->dataType()),
                            z->buffer(), z->shapeInfo(), z->specialBuffer(), z->specialShapeInfo(),
                            dims.data(), (int) dims.size(), pTadShape, pTadOffsets);
                }

                STORE_RESULT(*z);
            } else {
                auto indices = INPUT_VARIABLE(1);
                if (indices->lengthOf() == x->rankOf())
                    allAxes = true;

                //indices->printIndexedBuffer("indices");

                std::vector<int> dims(indices->lengthOf());
                for (int e = 0; e < indices->lengthOf(); e++) {
                    // lol otherwise we segfault on macOS
                    int f = indices->e<int>(e);
                    dims[e] = f >= 0 ? f : f += x->rankOf();
                }

                if ((block.numI() == 1 && INT_ARG(0) == sd::DataTypeUtils::max<int>()) || allAxes) {
                    // scalar
                    NativeOpExecutioner::execReduceLongScalar(block.launchContext(), opNum, x->buffer(), x->shapeInfo(), x->specialBuffer(), x->specialShapeInfo(), extras.argumentsAsT(x->dataType()), z->buffer(), z->shapeInfo(), z->specialBuffer(), z->specialShapeInfo());
                } else {
                    // TAD
                    REQUIRE_TRUE(dims.size() > 0, 0, "Some dimensions required for reduction!");

                    auto packX = sd::ConstantTadHelper::getInstance()->tadForDimensions(x->shapeInfo(), dims);

                    auto pTadShape = Environment::getInstance()->isCPU() ? packX.primaryShapeInfo() : packX.specialShapeInfo(); //(Nd4jLong *) manager.replicatePointer(tad.tadOnlyShapeInfo, shape::shapeInfoByteLength(tad.tadOnlyShapeInfo));
                    auto pTadOffsets = Environment::getInstance()->isCPU() ? packX.primaryOffsets() : packX.specialOffsets(); //(Nd4jLong *) manager.replicatePointer(tad.tadOffsets, tad.numTads * sizeof(Nd4jLong));

                    NativeOpExecutioner::execReduceLong(block.launchContext(), opNum, x->buffer(), x->shapeInfo(), x->specialBuffer(), x->specialShapeInfo(), extras.argumentsAsT(x->dataType()),
                            z->buffer(), z->shapeInfo(), z->specialBuffer(), z->specialShapeInfo(), dims.data(), (int) dims.size(), pTadShape, pTadOffsets);

                }
            }

            manager.synchronize();
            return Status::OK();
        }

        /**
        *   For all reductions rules are simple: either you return scalar, or you return reduced NDArray.
        *   It solely depends on input shape, and requested dimensions
        */
        ShapeList *LegacyReduceLongOp::calculateOutputShape(ShapeList *inputShape, sd::graph::Context &block) {
            auto inShape = inputShape->at(0);

            Nd4jLong *newShape;

            bool allAxes = false;

            auto keepDims = block.numB() > 0 ? B_ARG(0) : false;
            auto newFormat = block.numB() > 1 ? B_ARG(1) : true;

            auto axis = block.width() > 1 ? INPUT_VARIABLE(1)->asVectorT<int>() : block.getAxis();

            if (axis.size() == shape::rank(inShape))
                allAxes = true;

            // in this case we're building proper shape for reduction
            return SHAPELIST(ShapeUtils::evalReduceShapeInfo(shape::order(inShape), axis, inShape, DataType::INT64, keepDims, !newFormat, block.workspace()));
        }
    }
}