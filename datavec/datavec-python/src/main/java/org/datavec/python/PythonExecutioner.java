
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

import lombok.extern.slf4j.Slf4j;
import org.apache.commons.io.IOUtils;
import org.bytedeco.cpython.PyThreadState;
import org.bytedeco.numpy.global.numpy;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.io.ClassPathResource;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.Charset;
import java.util.Arrays;
import java.util.Map;

import static org.bytedeco.cpython.global.python.*;
import static org.bytedeco.cpython.global.python.PyThreadState_Get;
import static org.datavec.python.Python.*;

@Slf4j
public class PythonExecutioner {


    private static boolean init;
    public final static String DEFAULT_PYTHON_PATH_PROPERTY = "org.datavec.python.path";
    public final static String JAVACPP_PYTHON_APPEND_TYPE = "org.datavec.python.javacpp.path.append";
    public final static String DEFAULT_APPEND_TYPE = "before";

    static{init();}
    private static synchronized  void init() {
        if(init) {
            return;
        }
        setPythonPath();
        init = true;
        log.info("CPython: PyEval_InitThreads()");
        PyEval_InitThreads();
        log.info("CPython: Py_InitializeEx()");
        Py_InitializeEx(0);
        numpy._import_array();
    }

    private static synchronized void _exec(String code) {
        log.debug(code);
        log.info("CPython: PyRun_SimpleStringFlag()");

        int result = PyRun_SimpleStringFlags(code, null);
        if (result != 0) {
            log.info("CPython: PyErr_Print");
            PyErr_Print();
            throw new RuntimeException("exec failed");
        }
    }

    public static void setVariable(String varName, PythonObject pythonObject){
        Python.globals().set(new PythonObject(varName), pythonObject);
    }
    public static void setVariable(String varName, PythonVariables.Type varType, Object value){
        PythonObject pythonObject;
        switch(varType){
            case STR:
                pythonObject = new PythonObject((String)value);
                break;
            case INT:
                pythonObject = new PythonObject(((Number)value).longValue());
                break;
            case FLOAT:
                pythonObject = new PythonObject(((Number)value).floatValue());
                break;
            case BOOL:
                pythonObject = new PythonObject((boolean)value);
                break;
            case NDARRAY:
                if (value instanceof NumpyArray){
                    pythonObject = new PythonObject((NumpyArray) value);
                }
                else if (value instanceof INDArray){
                    pythonObject = new PythonObject((INDArray) value);
                }
                else{
                    throw new RuntimeException("Invalid value for type NDARRAY");
                }
                break;
            case LIST:
                pythonObject = new PythonObject(Arrays.asList((Object[])value));
                break;
            case DICT:
                pythonObject = new PythonObject((Map)value);
                break;
            default:
                throw new RuntimeException("Unsupported type: " + varType);

        }
        setVariable(varName, pythonObject);
    }

    public static void setVariables(PythonVariables pyVars){
        if (pyVars == null)return;
        for (String varName: pyVars.getVariables()){
            setVariable(varName, pyVars.getType(varName), pyVars.getValue(varName));
        }
    }

    public static PythonObject getVariable(String varName){
        return Python.globals().attr("get").call(varName);
    }
    public static Object getVariable(String varName, PythonVariables.Type varType){
        PythonObject pythonObject = getVariable(varName);
        if (pythonObject.isNone()){
            throw new RuntimeException("Variable not found: " + varName);
        }
        switch (varType){
            case INT:
                return pythonObject.toLong();
            case FLOAT:
                return pythonObject.toDouble();
            case NDARRAY:
                return pythonObject.toNumpy();
            case STR:
                return pythonObject.toString();
            case BOOL:
                return pythonObject.toBoolean();
            case LIST:
                return pythonObject.toList();
            case DICT:
                return pythonObject.toMap();
                default:
                    throw new RuntimeException("Unsupported type: " + varType);
        }
    }
    public static void getVariables(PythonVariables pyVars){
        for (String varName: pyVars.getVariables()){
            pyVars.setValue(varName, getVariable(varName, pyVars.getType(varName)));
        }
    }


    private static String getWrappedCode(String code) {
        try(InputStream is = new ClassPathResource("pythonexec/pythonexec.py").getInputStream()) {
            String base = IOUtils.toString(is, Charset.defaultCharset());
            StringBuffer indentedCode = new StringBuffer();
            for(String split : code.split("\n")) {
                indentedCode.append("    " + split + "\n");

            }

            String out = base.replace("    pass",indentedCode);
            return out;
        } catch (IOException e) {
            throw new IllegalStateException("Unable to read python code!",e);
        }

    }


    public static void exec(String code){
        _exec(getWrappedCode(code));
    }

    public static void exec(String code, PythonVariables outputVariables){
        _exec(getWrappedCode(code));
        getVariables(outputVariables);
    }

    public static void exec(String code, PythonVariables inputVariables, PythonVariables outputVariables){
        setVariables(inputVariables);
        _exec(getWrappedCode(code));
        getVariables(outputVariables);
    }

    public static PythonVariables execAndReturnAllVariables(String code) {
        _exec(getWrappedCode(code));
        PythonVariables out = new PythonVariables();
        PythonObject globals = Python.globals();
        PythonObject keysList = Python.list(globals.attr("keys"));
        int numKeys = Python.len(keysList).toInt();
        for (int i=0;i<numKeys;i++){
            PythonObject key = keysList.get(i);
            String keyStr = key.toString();
            if (!keyStr.startsWith("_")){
                PythonObject val = globals.get(key);
                if (Python.isinstance(val, intType())){
                    out.addInt(keyStr, val.toInt());
                }
                else if (Python.isinstance(val, floatType())){
                    out.addFloat(keyStr, val.toDouble());
                }
                else if (Python.isinstance(val, strType())){
                    out.addStr(keyStr, val.toString());
                }
                else if (Python.isinstance(val, boolType())){
                    out.addBool(keyStr, val.toBoolean());
                }
                else if (Python.isinstance(val, listType())){
                    out.addList(keyStr, val.toList().toArray(new Object[0]));
                }
                else if (Python.isinstance(val, dictType())){
                    out.addDict(keyStr, val.toMap());
                }
            }
        }
        return out;

    }

    public static PythonVariables getAllVariables(){
        PythonVariables out = new PythonVariables();
        PythonObject globals = Python.globals();
        PythonObject keysList = Python.list(globals.attr("keys").call());
        int numKeys = Python.len(keysList).toInt();
        for (int i=0;i<numKeys;i++){
            PythonObject key = keysList.get(i);
            String keyStr = key.toString();
            if (!keyStr.startsWith("_")){
                PythonObject val = globals.get(key);
                if (Python.isinstance(val, intType())){
                    out.addInt(keyStr, val.toInt());
                }
                else if (Python.isinstance(val, floatType())){
                    out.addFloat(keyStr, val.toDouble());
                }
                else if (Python.isinstance(val, strType())){
                    out.addStr(keyStr, val.toString());
                }
                else if (Python.isinstance(val, boolType())){
                    out.addBool(keyStr, val.toBoolean());
                }
                else if (Python.isinstance(val, listType())){
                    out.addList(keyStr, val.toList().toArray(new Object[0]));
                }
                else if (Python.isinstance(val, dictType())){
                    out.addDict(keyStr, val.toMap());
                }
            }
        }
        return out;
    }
    public static PythonVariables execAndReturnAllVariables(String code, PythonVariables inputs){
        setVariables(inputs);
        _exec(getWrappedCode(code));
      return getAllVariables();
    }

    /**
     * One of a few desired values
     * for how we should handle
     * using javacpp's python path.
     * BEFORE: Prepend the python path alongside a defined one
     * AFTER: Append the javacpp python path alongside the defined one
     * NONE: Don't use javacpp's python path at all
     */
    public enum JavaCppPathType {
        BEFORE,AFTER,NONE
    }

    /**
     * Set the python path.
     * Generally you can just use the PYTHONPATH environment variable,
     * but if you need to set it from code, this can work as well.
     */

    public static synchronized void setPythonPath() {
        if(!init) {
            try {
                String path = System.getProperty(DEFAULT_PYTHON_PATH_PROPERTY);
                if(path == null) {
                    log.info("Setting python default path");
                    File[] packages = numpy.cachePackages();
                    Py_SetPath(packages);
                }
                else {
                    log.info("Setting python path " + path);
                    StringBuffer sb = new StringBuffer();
                    File[] packages = numpy.cachePackages();

                    JavaCppPathType pathAppendValue = JavaCppPathType.valueOf(System.getProperty(JAVACPP_PYTHON_APPEND_TYPE,DEFAULT_APPEND_TYPE).toUpperCase());
                    switch(pathAppendValue) {
                        case BEFORE:
                            for(File cacheDir : packages) {
                                sb.append(cacheDir);
                                sb.append(java.io.File.pathSeparator);
                            }

                            sb.append(path);

                            log.info("Prepending javacpp python path " + sb.toString());
                            break;
                        case AFTER:
                            sb.append(path);

                            for(File cacheDir : packages) {
                                sb.append(cacheDir);
                                sb.append(java.io.File.pathSeparator);
                            }

                            log.info("Appending javacpp python path " + sb.toString());
                            break;
                        case NONE:
                            log.info("Not appending javacpp path");
                            sb.append(path);
                            break;
                    }

                    //prepend the javacpp packages
                    log.info("Final python path " + sb.toString());

                    Py_SetPath(sb.toString());
                }
            } catch (IOException e) {
                log.error("Failed to set python path.", e);
            }
        }
        else {
            throw new IllegalStateException("Unable to reset python path. Already initialized.");
        }
    }
}
