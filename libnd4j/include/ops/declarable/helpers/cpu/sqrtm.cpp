/*******************************************************************************
 * Copyright (c) Konduit K.K.
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
//  @author GS <sgazeos@gmail.com>
//

#include <ops/declarable/helpers/sqrtm.h>
#include <ops/declarable/helpers/qr.h>
#include <helpers/MmulHelper.h>

namespace sd {
namespace ops {
namespace helpers {

    template <typename T>
    void upperTriangularSqrt(sd::LaunchContext* context, NDArray const* inputTriangular, NDArray* outputTriangular) {
        auto n = inputTriangular->sizeAt(-1);
        auto inputTriangularPart = inputTriangular->allTensorsAlongDimension({-2, -1});
        auto outputTriangularPart = outputTriangular->allTensorsAlongDimension({-2, -1});

        for (auto batch = 0; batch < inputTriangularPart.size(); ++batch) {
            // compute diagonals
            auto input = inputTriangularPart[batch];
            auto output = outputTriangularPart[batch];
            for (auto r = 0; r < n; r++) {
                output->t<T>(r, r) = sd::math::nd4j_sqrt<T,T>(input->t<T>(r, r));
            }

            // compute upper diagonal
            for (auto r = 0; r < n - 1; r++) {
                output->t<T>(r, r + 1) = input->t<T>(r, r + 1) / (output->t<T>(r, r) + output->t<T>(r + 1, r + 1));
            }

            // loop for diagonals
            for (auto d = 2; d < n; d++) {
                for (auto r = 0; r < n - d; r++) {
                    auto sum = T(0.f);
                    for (auto k = r + 1; k < r + d; k++) {
                        sum += output->t<T>(r, k) * output->t<T>(k, d + r);
                    }
                    output->t<T>(r, r + d) = (input->t<T>(r, r + d) - sum) / (output->t<T>(r, r) + output->t<T>(r + d, r + d));
                }
            }
        }
    }

    //
    // Input/output 2D arrays
    //
    template <typename T>
    static void computeTriangulars(sd::LaunchContext* context, NDArray const& input, NDArray& outputPlus, NDArray& outputMinus) {
        outputPlus.nullify();
        outputMinus.nullify();
        auto n = input.sizeAt(-1);
        for (auto r = 0; r < n; r++) {
            outputPlus.t<T>(r,r) = sd::math::nd4j_sqrt<T,T>(input.t<T>(r,r));
            outputMinus.t<T>(r,r) = sd::math::nd4j_sqrt<T,T>(input.t<T>(r,r));
        }
        for (auto r = 0; r < n; r++) {
            for (auto c = r + 1; c < n; c++) {
                auto sumPlus = T(0.f);
                auto sumMinus = T(0.f);
                for (auto j = r + 1; j < c; j++) {
                    sumPlus += outputPlus.t<T>(r, j) * outputPlus.t<T>(j, c);
                    sumMinus += outputMinus.t<T>(r, j) * outputMinus.t<T>(j, c);
                }
                outputPlus.t<T>(r,c) = (input.t<T>(r,c) - sumPlus) / (outputPlus.t<T>(r,r) + outputPlus.t<T>(c,c));
                outputMinus.t<T>(r,c) = (input.t<T>(r,c) - sumMinus) / (outputMinus.t<T>(r,r) + outputMinus.t<T>(c,c));
            }
        }

    }

    template <typename T>
    static void quasyTriangularCompute(sd::LaunchContext* context, NDArray const* inputR, NDArray* outputT) {
        auto inputTriangularPart = inputR->allTensorsAlongDimension({-2, -1});
        auto outputTriangularPart = outputT->allTensorsAlongDimension({-2, -1});

        for (auto batch = 0; batch < inputTriangularPart.size(); ++batch) {
            auto input = inputTriangularPart[batch];
            auto output = outputTriangularPart[batch];
            auto outputPlus = output->ulike();
            auto outputMinus = output->ulike();
            computeTriangulars<T>(context, *input, outputPlus, outputMinus);
        }
    }

    template <typename T>
    void schurDecomposition(sd::LaunchContext* context, NDArray const* input, NDArray* qMatrix, NDArray* tMatrix) {
        qMatrix->setIdentity();
        tMatrix->assign(input);
    }

    template <typename T>
    int sqrtMatrixFunctor_(sd::LaunchContext* context, NDArray const* input, NDArray* output) {
        auto pInput = const_cast<NDArray*>(input);
        auto outputQ = pInput->ulike();
        auto outputT = outputQ.ulike();

        schurDecomposition<T>(context, pInput, &outputQ, &outputT);

//        auto outputT = outputR.ulike();

        upperTriangularSqrt<T>(context, &outputT, output);
        MmulHelper::matmul(&outputQ, output, &outputT, false, false);
        MmulHelper::matmul(&outputT, &outputQ, output, false, true);

        return Status::OK();
    }

    int sqrtMatrixFunctor(sd::LaunchContext* context, NDArray const* input, NDArray* output) {
        BUILD_SINGLE_SELECTOR(input->dataType(), return sqrtMatrixFunctor_, (context, input, output), FLOAT_TYPES);
    }
}
}
}
