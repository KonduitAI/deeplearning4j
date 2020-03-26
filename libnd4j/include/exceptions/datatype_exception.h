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
// Created by raver on 11/26/2018.
//

#ifndef SD_DATATYPE_EXCEPTION_H
#define SD_DATATYPE_EXCEPTION_H

#include <string>
#include <stdexcept>
#include <array/DataType.h>
#include <system/dll.h>

#if defined(_MSC_VER)

// we're ignoring warning about non-exportable parent class, since std::runtime_error is a part of Standard C++ Library
#pragma warning( disable : 4275 )

#endif

namespace sd {
    class ND4J_EXPORT datatype_exception : public std::runtime_error {
    public:
        datatype_exception(std::string message);
        ~datatype_exception() = default;


        static datatype_exception build(std::string message, sd::DataType actual);
        static datatype_exception build(std::string message, sd::DataType expected, sd::DataType actual);
        static datatype_exception build(std::string message, sd::DataType expected, sd::DataType actualX, sd::DataType actualY);
    };
}


#endif //SD_DATATYPE_EXCEPTION_H
