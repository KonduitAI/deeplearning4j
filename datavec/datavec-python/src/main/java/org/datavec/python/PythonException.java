
/*******************************************************************************
 * Copyright (c) 2019 Konduit K.K.
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

package org.datavec.python;

/**
 * Thrown when an exception occurs in python land
 */
public class PythonException extends Exception {
    public PythonException(String message){
        super(message);
    }
    private static String getExceptionString(PythonObject exception){
        if (Python.isinstance(exception, Python.ExceptionType())){
            String exceptionClass = Python.type(exception).attr("__name__").toString();
            String message = exception.toString();
            return exceptionClass + ": " + message;
        }
        return exception.toString();
    }
    public PythonException(PythonObject exception){
        this(getExceptionString(exception));
    }
    public PythonException(String message, Throwable cause){
        super(message, cause);
    }
    public PythonException(Throwable cause){
        super(cause);
    }
}
