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
// Created by raver on 9/1/2018.
//

#ifndef LIBND4J_GRAPH_EXCEPTION_H
#define LIBND4J_GRAPH_EXCEPTION_H

#include <string>
#include <stdexcept>
#include <system/pointercast.h>
#include <system/dll.h>

#if defined(_MSC_VER)

// we're ignoring warning about non-exportable parent class, since std::runtime_error is a part of Standard C++ Library
#pragma warning( disable : 4275 )

#endif

namespace sd {
    class ND4J_EXPORT graph_exception : public std::runtime_error {
    protected:
        Nd4jLong _graphId;
        std::string _message;
        std::string _description;
    public:
        graph_exception(std::string message, Nd4jLong graphId);
        graph_exception(std::string message, std::string description, Nd4jLong graphId);
        graph_exception(std::string message, const char *description, Nd4jLong graphId);
        ~graph_exception() = default;

        Nd4jLong graphId();

        const char * message();
        const char * description();
    };
}



#endif //SD_GRAPH_EXCEPTION_H
