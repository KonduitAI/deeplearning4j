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

#include <system/BlasVersionHelper.h>

namespace sd {
    BlasVersionHelper::BlasVersionHelper() {
#if defined(__clang__) && defined(__CUDA__)
        _blasMajorVersion = 0;
        _blasMinorVersion = 0;
        _blasPatchVersion = 0;
#else
        _blasMajorVersion = __CUDACC_VER_MAJOR__;
        _blasMinorVersion = __CUDACC_VER_MINOR__;
        _blasPatchVersion = __CUDACC_VER_BUILD__;
#endif
    }
}