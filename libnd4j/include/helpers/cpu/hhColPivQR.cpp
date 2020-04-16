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
// Created by Yurii Shyrma on 11.01.2018
//

#include <helpers/hhColPivQR.h>
#include <helpers/householder.h>
#include <array/NDArrayFactory.h>

namespace sd {
namespace ops {
namespace helpers {


//////////////////////////////////////////////////////////////////////////
HHcolPivQR::HHcolPivQR(const NDArray& matrix) {

    _qr = matrix;
    _diagSize = math::nd4j_min<int>(matrix.sizeAt(0), matrix.sizeAt(1));
    _coeffs = NDArrayFactory::create(matrix.ordering(), {1, _diagSize}, matrix.dataType(), matrix.getContext());

    _permut = NDArrayFactory::create(matrix.ordering(), {matrix.sizeAt(1), matrix.sizeAt(1)}, matrix.dataType(), matrix.getContext());

    evalData();
}

    void HHcolPivQR::evalData() {
        BUILD_SINGLE_SELECTOR(_qr.dataType(), _evalData, (), FLOAT_TYPES);
    }

//////////////////////////////////////////////////////////////////////////
template <typename T>
void HHcolPivQR::_evalData() {

    int rows = _qr.sizeAt(0);
    int cols = _qr.sizeAt(1);

    NDArray transp(_qr.ordering(),   {cols}/*{1, cols}*/, _qr.dataType(), _qr.getContext());
    NDArray normsUpd(_qr.ordering(), {cols}/*{1, cols}*/, _qr.dataType(), _qr.getContext());
    NDArray normsDir(_qr.ordering(), {cols}/*{1, cols}*/, _qr.dataType(), _qr.getContext());

    int transpNum = 0;

    for (int k = 0; k < cols; ++k) {

        T norm = _qr({0,0, k,k+1}).reduceNumber(reduce::Norm2).t<T>(0);
        normsDir.t<T>(k) = normsUpd.t<T>(k) = norm;
    }

    T normScaled = (normsUpd.reduceNumber(reduce::Max)).t<T>(0) * DataTypeUtils::eps<T>();
    T threshold1 = normScaled * normScaled / (T)rows;
    T threshold2 = math::nd4j_sqrt<T,T>(DataTypeUtils::eps<T>());

    T nonZeroPivots = _diagSize;
    T maxPivot = 0.;

    for(int k = 0; k < _diagSize; ++k) {

        int biggestColIndex = normsUpd({k,-1}).indexReduceNumber(indexreduce::IndexMax).e<int>(0);
        T biggestColNorm = normsUpd({k,-1}).reduceNumber(reduce::Max).t<T>(0);
        T biggestColSqNorm = biggestColNorm * biggestColNorm;
        biggestColIndex += k;

        if(nonZeroPivots == (T)_diagSize && biggestColSqNorm < threshold1 * (T)(rows-k))
            nonZeroPivots = k;

        transp.t<T>(k) = (T)biggestColIndex;

        if(k != biggestColIndex) {

            NDArray temp1(_qr({0,0, k,k+1}, true));
            NDArray temp2(_qr({0,0, biggestColIndex,biggestColIndex+1}, true));
            auto temp3 = temp1;
            temp1.assign(temp2);
            temp2.assign(temp3);

            math::nd4j_swap<T>(normsUpd.t<T>(k), normsUpd.t<T>(biggestColIndex));
            math::nd4j_swap<T>(normsDir.t<T>(k), normsDir.t<T>(biggestColIndex));

            ++transpNum;
        }

        T normX;
        NDArray qrBlock(_qr({k,rows, k,k+1}, true));
        T c;
        Householder<T>::evalHHmatrixDataI(qrBlock, c, normX);
        _coeffs.t<T>(k) = c;

        _qr.t<T>(k,k) = normX;

        T max = math::nd4j_abs<T>(normX);
        if(max > maxPivot)
            maxPivot = max;

        if(k < rows && (k+1) < cols) {
            NDArray qrBlock(_qr({k,  rows,  k+1,cols}, true));
            NDArray tail(_qr({k+1,rows,  k, k+1},   true));
            Householder<T>::mulLeft(qrBlock, tail, _coeffs.t<T>(k));
        }

        for (int j = k + 1; j < cols; ++j) {

            if (normsUpd.t<T>(j) != (T)0.f) {
                T temp = math::nd4j_abs<T>(_qr.t<T>(k, j)) / normsUpd.t<T>(j);
                temp = (1. + temp) * (1. - temp);
                temp = temp < (T)0. ? (T)0. : temp;
                T temp2 = temp * normsUpd.t<T>(j) * normsUpd.t<T>(j) / (normsDir.t<T>(j)*normsDir.t<T>(j));

                if (temp2 <= threshold2) {
                    if(k+1 < rows && j < cols)
                        normsDir.t<T>(j) = _qr({k+1,rows, j,j+1}).reduceNumber(reduce::Norm2).t<T>(0);

                    normsUpd.t<T>(j) = normsDir.t<T>(j);
                }
                else
                    normsUpd.t<T>(j) = normsUpd.t<T>(j) * math::nd4j_sqrt<T, T>(temp);
            }
        }
    }

    _permut.setIdentity();

    for(int k = 0; k < _diagSize; ++k) {

        int idx = transp.e<int>(k);
        auto temp1 = new NDArray(_permut({0,0, k, k+1},    true));
        auto temp2 = new NDArray(_permut({0,0, idx,idx+1}, true));
        auto  temp3 = *temp1;
        temp1->assign(temp2);
        temp2->assign(temp3);
        delete temp1;
        delete temp2;
    }
}

    BUILD_SINGLE_TEMPLATE(template void HHcolPivQR::_evalData, (), FLOAT_TYPES);

}
}
}

