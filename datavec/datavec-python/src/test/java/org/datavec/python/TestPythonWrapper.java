package org.datavec.python;


import lombok.var;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

@javax.annotation.concurrent.NotThreadSafe
public class TestPythonWrapper {
    @Test
    public void testPythonWrapperBasics(){
        PythonExecutioner.init();
        var list = new PythonObject(new ArrayList());
        list.attr("append").call("item1");
        list.attr("append").call("item2");
        String str = Python.str(list).toString();
        assertEquals(str, "[\'item1\', \'item2\']");
    }


    @Test
    public void testPythonWrapperWithNumpy(){
        PythonExecutioner.init();
        var np = Python.importModule("numpy");
        var pyZeros = np.attr("zeros").call(Arrays.asList(Arrays.asList(32, 16)));
        INDArray zeros = pyZeros.toNumpy().getNd4jArray();
        assertArrayEquals(zeros.shape(), new long[]{32, 16});
    }



}
