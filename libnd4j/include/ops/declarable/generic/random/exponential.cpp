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

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_random_exponential)

#include <ops/declarable/headers/random.h>
#include <helpers/RandomLauncher.h>

namespace sd {
    namespace ops {
        CUSTOM_OP_IMPL(random_exponential, 1, 1, true, 1, 0) {
            // uniform distribution
            auto rng = block.randomGenerator();

            // FIXME: to be implemented
            /*
            if (rng == nullptr)
                return Status::THROW("RNG is null, aborting...");

            auto x = INPUT_VARIABLE(0);
            auto z = OUTPUT_VARIABLE(0);

            if (block.width() == 1)
                functions::random::RandomFunction<T>::template execTransform<randomOps::ExponentialDistribution<T>>(block.getRNG(), z->getBuffer(), z->getShapeInfo(), block.getTArguments()->data());
            else {
                auto y = INPUT_VARIABLE(1);
                REQUIRE_TRUE(y->isSameShape(z), 0, "ExponentialDistribution: Y shape should be equal to Z shape");

                functions::random::RandomFunction<T>::template execTransform<randomOps::ExponentialDistribution<T>>(block.getRNG(), y->getBuffer(), y->getShapeInfo(), z->getBuffer(), z->getShapeInfo(), block.getTArguments()->data());
            }

            STORE_RESULT(*z);
*/

            auto z = OUTPUT_VARIABLE(0);
            auto lambda = T_ARG(0);

            RandomLauncher::fillExponential(block.launchContext(), rng, z, lambda);

            return Status::OK();
        }


        DECLARE_SHAPE_FN(random_exponential) {
            auto in = INPUT_VARIABLE(0);
            auto shape = in->template asVectorT<Nd4jLong>();

            auto newShape = ConstantShapeHelper::getInstance()->createShapeInfo(DataType::FLOAT32, 'c', shape);
            return SHAPELIST(newShape);
        }

        DECLARE_TYPES(random_exponential) {
            getOpDescriptor()
                    ->setAllowedInputTypes(sd::DataType::ANY)
                    ->setAllowedOutputTypes({ALL_FLOATS});
        }
    }
}

#endif