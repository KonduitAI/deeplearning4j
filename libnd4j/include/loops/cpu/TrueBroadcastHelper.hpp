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
 // @author Yurii Shyrma (iuriish@yahoo.com)
 //

#include <loops/TrueBroadcastHelper.h>
#include <ops/ops.h>
#include <execution/Threads.h>

using namespace simdOps;

namespace nd4j {
    namespace helpers {

        ////////////////////////////////////////////////////////////////////////
        template <typename X, typename  Y, typename Z>
        template<typename OpType>
        void TrueBroadcastHelper<X, Y, Z>::exec(const NDArray& xArr, const NDArray& yArr, NDArray& zArr) {


            const X* x = reinterpret_cast<X*>(xArr.getBuffer());
            const Y* y = reinterpret_cast<Y*>(yArr.getBuffer());
            Z* z = reinterpret_cast<Z*>(zArr.getBuffer());

            const auto xShapeInfo = xArr.getShapeInfo();
            const auto yShapeInfo = yArr.getShapeInfo();
            const auto zShapeInfo = zArr.getShapeInfo();

            const int xRank = xArr.rankOf();
            const int yRank = yArr.rankOf();
            const int zRank = zArr.rankOf();

            bool bSpecialCase = (1 == xArr.ews() && 'c' == xArr.ordering() &&
                1 == yArr.ews() && 'c' == yArr.ordering() &&
                1 == zArr.ews() && 'c' == zArr.ordering());

            if (bSpecialCase && yArr.isColumnVector() && 1 == xArr.sizeAt(-1) ) {
                auto yLen = yArr.lengthOf();
                auto func = PRAGMA_THREADS_FOR{
                   for (auto i = start; i < stop; i++) {
                       auto rZ = z + (i * yLen);
                       auto v = x[i];
                       for (Nd4jLong j = 0; j < yLen; j++) {
                            rZ[j] = OpType::op(v, y[j]);
                       }
                   }
                };
                samediff::Threads::parallel_tad(func, 0, xArr.lengthOf());
                return;
            }


            auto yShapeInt = yArr.getShapeAsVectorInt();
            auto xShapeInt = xArr.getShapeAsVectorInt();
            auto nCountY = std::count_if(yShapeInt.cbegin(), yShapeInt.cend(), [](int i) { return i == 1; });
            auto nCountX = std::count_if(xShapeInt.cbegin(), xShapeInt.cend(), [](int i) { return i == 1; });

            bool bSpecialCase2 = (xRank == zRank && yRank == zRank && 1 == xArr.sizeAt(-1) && 1 == yArr.sizeAt(-2) && 1 == nCountY && 1 == nCountX);

            if (bSpecialCase && bSpecialCase2) {

                uint32_t zDim1 = zArr.sizeAt(-2);
                uint32_t zDim2 = zArr.sizeAt(-1);

                uint32_t nLen = zArr.lengthOf() / yArr.sizeAt(-1);

                auto func = PRAGMA_THREADS_FOR{
                     for (auto total = start; total < stop; total++) {

                        uint32_t i = total / zDim1;
                        uint32_t j = total % zDim1;

                        uint32_t index = (i * zDim1) + j;
                        auto rZ = z + (index * zDim2);
                        auto rY = y + (i * zDim2);
                        auto rX = x[index];

                        for (uint32_t n = 0; n < zDim2; n++) {
                             rZ[n] = OpType::op(rX, rY[n]);
                        }
                    }
                };
                samediff::Threads::parallel_tad(func, 0, nLen, 1);
                return;
            }


            const Nd4jLong zLen = zArr.lengthOf();
            auto func = PRAGMA_THREADS_FOR{
                std::vector<Nd4jLong> xCoords(xArr.rankOf()), yCoords(yArr.rankOf()), zCoords(zArr.rankOf());

                for (auto i = start; i < stop; ++i) {

                    shape::index2coords(i, zShapeInfo, zCoords.data());

                    for (int ix = xRank - 1, iy = yRank - 1, iz = zRank - 1; iz >= 0; --iz) {

                        if (ix >= 0) {
                            if (xShapeInfo[ix + 1] == zShapeInfo[iz + 1]) {
                                xCoords[ix--] = zCoords[iz];
                            }
                             else {
                              xCoords[ix--] = 0;
                             }
                        }
                    
                        if (iy >= 0) {
                              if (yShapeInfo[iy + 1] == zShapeInfo[iz + 1]) {
                                  yCoords[iy--] = zCoords[iz];
                              }
                              else {
                                  yCoords[iy--] = 0;
                              }
                          }
                    }

                        const auto xOffset = shape::getOffset(xShapeInfo, xCoords.data());
                        const auto yOffset = shape::getOffset(yShapeInfo, yCoords.data());
                        const auto zOffset = shape::getOffset(zShapeInfo, zCoords.data());
                        
                        z[zOffset] = OpType::op(x[xOffset], y[yOffset]);
                }
            };

            samediff::Threads::parallel_for(func, 0, zLen);
        }

        template <typename X, typename  Y, typename Z>
        void TrueBroadcastHelper<X, Y, Z>::exec(const nd4j::broadcast::Ops opNum, const NDArray& xArr, const NDArray& yArr, NDArray& zArr) {
            DISPATCH_BY_OPNUM_TTT(exec, PARAMS(xArr, yArr, zArr), BROADCAST_OPS);
        }

        ////////////////////////////////////////////////////////////////////////
        template <typename X, typename  Z>
        template<typename OpType>
        void TrueBroadcastBoolHelper<X, Z>::exec(const NDArray& xArr, const NDArray& yArr, NDArray& zArr) {

            const X* x = reinterpret_cast<X*>(xArr.getBuffer());
            const X* y = reinterpret_cast<X*>(yArr.getBuffer());
            Z* z = reinterpret_cast<Z*>(zArr.getBuffer());

            const auto xShapeInfo = xArr.getShapeInfo();
            const auto yShapeInfo = yArr.getShapeInfo();
            const auto zShapeInfo = zArr.getShapeInfo();

            const int xRank = xArr.rankOf();
            const int yRank = yArr.rankOf();
            const int zRank = zArr.rankOf();

            const Nd4jLong zLen = zArr.lengthOf();

            auto func = PRAGMA_THREADS_FOR{
                std::vector<Nd4jLong> xCoords(xArr.rankOf()), yCoords(yArr.rankOf()), zCoords(zArr.rankOf());

                for (auto i = start; i < stop; ++i) {

                    shape::index2coords(i, zShapeInfo, zCoords.data());

                    for (int ix = xRank - 1, iy = yRank - 1, iz = zRank - 1; iz >= 0; --iz) {

                        if (ix >= 0) {
                            if (xShapeInfo[ix + 1] == zShapeInfo[iz + 1]) {
                                xCoords[ix--] = zCoords[iz];
                            }
                            else {
                                xCoords[ix--] = 0;
                            }
                        }
                
                        if (iy >= 0) {
                            if (yShapeInfo[iy + 1] == zShapeInfo[iz + 1]) {
                                yCoords[iy--] = zCoords[iz];
                            }
                            else {
                                 yCoords[iy--] = 0;
                            }
                        }
                }
                
                const auto xOffset = shape::getOffset(xShapeInfo, xCoords.data());
                const auto yOffset = shape::getOffset(yShapeInfo, yCoords.data());
                const auto zOffset = shape::getOffset(zShapeInfo, zCoords.data());
                
                z[zOffset] = OpType::op(x[xOffset], y[yOffset], nullptr);
                }
            };

            samediff::Threads::parallel_for(func, 0, zLen);
        }

        template <typename X, typename  Y>
        void TrueBroadcastBoolHelper<X, Y>::exec(const nd4j::broadcast::BoolOps opNum, const NDArray& xArr, const NDArray& yArr, NDArray& zArr) {
            DISPATCH_BY_OPNUM_TT(exec, PARAMS(xArr, yArr, zArr), BROADCAST_BOOL_OPS);
        }

        ////////////////////////////////////////////////////////////////////////
        template <typename X>
        template<typename OpType>
        void TrueBroadcastIntHelper<X>::exec(const NDArray& xArr, const NDArray& yArr, NDArray& zArr) {

            const X* x = reinterpret_cast<X*>(xArr.getBuffer());
            const X* y = reinterpret_cast<X*>(yArr.getBuffer());
            X* z = reinterpret_cast<X*>(zArr.getBuffer());

            const auto xShapeInfo = xArr.getShapeInfo();
            const auto yShapeInfo = yArr.getShapeInfo();
            const auto zShapeInfo = zArr.getShapeInfo();

            const int xRank = xArr.rankOf();
            const int yRank = yArr.rankOf();
            const int zRank = zArr.rankOf();

            const Nd4jLong zLen = zArr.lengthOf();

            auto func = PRAGMA_THREADS_FOR{
                std::vector<Nd4jLong> xCoords(xArr.rankOf()), yCoords(yArr.rankOf()), zCoords(zArr.rankOf());

                for (auto i = start; i < stop; ++i) {

                    shape::index2coords(i, zShapeInfo, zCoords.data());

                    for (int ix = xRank - 1, iy = yRank - 1, iz = zRank - 1; iz >= 0; --iz) {

                        if (ix >= 0) {
                            if (xShapeInfo[ix + 1] == zShapeInfo[iz + 1]) {
                                xCoords[ix--] = zCoords[iz];
                            }
                            else {
                                xCoords[ix--] = 0;
                            }
                        }
                
                        if (iy >= 0) {
                            if (yShapeInfo[iy + 1] == zShapeInfo[iz + 1]) {
                               yCoords[iy--] = zCoords[iz];
                            }
                            else {
                               yCoords[iy--] = 0;
                            }
                        }
                    }
                
                    const auto xOffset = shape::getOffset(xShapeInfo, xCoords.data());
                    const auto yOffset = shape::getOffset(yShapeInfo, yCoords.data());
                    const auto zOffset = shape::getOffset(zShapeInfo, zCoords.data());
                    
                    z[zOffset] = OpType::op(x[xOffset], y[yOffset]);
                }
            };

            samediff::Threads::parallel_for(func, 0, zLen);
        }

        template <typename X>
        void TrueBroadcastIntHelper<X>::exec(const nd4j::broadcast::IntOps opNum, const NDArray& xArr, const NDArray& yArr, NDArray& zArr) {
            DISPATCH_BY_OPNUM_T(exec, PARAMS(xArr, yArr, zArr), BROADCAST_INT_OPS);
        }

        /*
        BUILD_PAIRWISE_TEMPLATE(template class ND4J_EXPORT TrueBroadcastHelper, , PAIRWISE_TYPES_0);
        BUILD_PAIRWISE_TEMPLATE(template class ND4J_EXPORT TrueBroadcastHelper, , PAIRWISE_TYPES_1);
        BUILD_PAIRWISE_TEMPLATE(template class ND4J_EXPORT TrueBroadcastHelper, , PAIRWISE_TYPES_2);
        BUILD_PAIRWISE_TEMPLATE(template class ND4J_EXPORT TrueBroadcastHelper, , PAIRWISE_TYPES_3);
        BUILD_PAIRWISE_TEMPLATE(template class ND4J_EXPORT TrueBroadcastHelper, , PAIRWISE_TYPES_4);
        BUILD_PAIRWISE_TEMPLATE(template class ND4J_EXPORT TrueBroadcastHelper, , PAIRWISE_TYPES_5);
        BUILD_PAIRWISE_TEMPLATE(template class ND4J_EXPORT TrueBroadcastHelper, , PAIRWISE_TYPES_6);
        BUILD_PAIRWISE_TEMPLATE(template class ND4J_EXPORT TrueBroadcastHelper, , PAIRWISE_TYPES_7);
        BUILD_PAIRWISE_TEMPLATE(template class ND4J_EXPORT TrueBroadcastHelper, , PAIRWISE_TYPES_8);
        BUILD_PAIRWISE_TEMPLATE(template class ND4J_EXPORT TrueBroadcastHelper, , PAIRWISE_TYPES_9);

        BUILD_DOUBLE_TEMPLATE(template class ND4J_EXPORT TrueBroadcastBoolHelper, , LIBND4J_TYPES, BOOL_TYPES);

        BUILD_SINGLE_TEMPLATE(template class ND4J_EXPORT TrueBroadcastIntHelper, , INTEGER_TYPES);
        */
    }
}
