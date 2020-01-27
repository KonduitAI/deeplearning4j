package org.nd4j.linalg.api.ops.custom;

import lombok.NoArgsConstructor;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.base.Preconditions;
import org.nd4j.linalg.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;

import java.util.Collections;
import java.util.List;

@NoArgsConstructor
public class TriangularSolve extends DynamicCustomOp {

    public TriangularSolve(INDArray matrix, INDArray rhs, boolean lower, boolean adjoint) {
        addInputArgument(matrix, rhs);
        addBArgument(lower, adjoint);
    }

    public TriangularSolve(SameDiff sameDiff, SDVariable matrix, SDVariable rhs,
                           SDVariable lower, SDVariable adjoint) {
        super(sameDiff, new SDVariable[] {matrix, rhs, lower, adjoint});
    }

    @Override
    public String opName() {
        return "triangular_solve";
    }

    @Override
    public String tensorflowName() {
        return "MatrixTriangularSolve";
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> dataTypes) {
        int n = args().length;
        Preconditions.checkState(dataTypes != null && dataTypes.size() == n, "Expected %s input data types for %s, got %s", n, getClass(), dataTypes);
        return Collections.singletonList(dataTypes.get(0));
    }
}
