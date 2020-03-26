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

#ifndef SD_SCATTER_H
#define SD_SCATTER_H

#include <array/NDArray.h>

namespace sd {
    namespace ops {
        namespace helpers {
            void scatter(sd::LaunchContext* context, pairwise::Ops op, const NDArray& indices, const NDArray& updates, NDArray& output, const bool lock);

            void scatterND(sd::LaunchContext* context, pairwise::Ops op, const NDArray& indices, const NDArray& updates, NDArray& output, const bool lock);

            void scatterForLoss(sd::LaunchContext* context, const NDArray& indices, NDArray& updates, NDArray& output, const bool calcGrad);

            Nd4jLong checkIndices(sd::LaunchContext *context, const NDArray& indices, const NDArray& output, const int axis = -1);
        }
    }
}

#endif //SD_SCATTER_H
