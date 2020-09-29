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

package org.nd4j.linalg.aurora.ops;


import lombok.Data;
import lombok.Getter;
import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.indexer.LongIndexer;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.serde.FlatBuffersMapper;
import org.nd4j.base.Preconditions;
import org.nd4j.compression.impl.AbstractCompressor;
import org.nd4j.config.ND4JEnvironmentVars;
import org.nd4j.linalg.api.buffer.*;
import org.nd4j.linalg.api.concurrency.AffinityManager;
import org.nd4j.linalg.api.environment.Nd4jEnvironment;
import org.nd4j.linalg.api.memory.pointers.PagedPointer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ndarray.INDArrayStatistics;
import org.nd4j.linalg.api.ops.*;
import org.nd4j.linalg.api.ops.aggregates.Aggregate;
import org.nd4j.linalg.api.ops.aggregates.Batch;
import org.nd4j.linalg.api.ops.executioner.DefaultOpExecutioner;
import org.nd4j.linalg.api.ops.executioner.OpStatus;
import org.nd4j.linalg.api.ops.impl.scatter.ScatterUpdate;
import org.nd4j.linalg.api.ops.impl.summarystats.Variance;
import org.nd4j.linalg.api.ops.impl.transforms.any.IsMax;
import org.nd4j.linalg.api.ops.performance.PerformanceTracker;
import org.nd4j.linalg.api.ops.random.BaseRandomOp;
import org.nd4j.linalg.api.rng.Random;
import org.nd4j.linalg.api.shape.LongShapeDescriptor;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.api.shape.TadPack;
import org.nd4j.linalg.api.shape.options.ArrayOptionsHelper;
import org.nd4j.linalg.api.shape.options.ArrayType;
import org.nd4j.linalg.cache.ConstantHandler;
import org.nd4j.linalg.cache.TADManager;
import org.nd4j.linalg.compression.CompressionDescriptor;
import org.nd4j.linalg.compression.CompressionType;
import org.nd4j.linalg.compression.ThresholdCompression;
import org.nd4j.linalg.aurora.CpuTADManager;
import org.nd4j.linalg.aurora.buffer.BaseAuroraDataBuffer;
import org.nd4j.linalg.aurora.buffer.LongBuffer;
import org.nd4j.linalg.aurora.buffer.Utf8Buffer;
import org.nd4j.linalg.aurora.rng.CpuNativeRandom;
import org.nd4j.linalg.exception.ND4JIllegalArgumentException;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.linalg.exception.ND4JOpProfilerException;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.api.memory.MemcpyDirection;
import org.nd4j.linalg.primitives.AtomicBoolean;
import org.nd4j.linalg.primitives.Optional;
import org.nd4j.linalg.primitives.Pair;
import org.nd4j.linalg.util.ArrayUtil;
import org.nd4j.nativeblas.*;

import java.util.*;

/**
 *
 * Native operation
 * executioner in c++
 *
 * @author Adam Gibson
 */
@Slf4j
public class NativeOpExecutioner extends DefaultOpExecutioner {
    private NativeOps loop = NativeOpsHolder.getInstance().getDeviceNativeOps();
    private ConstantHandler constantHandler = Nd4j.getConstantHandler();
    @Getter
    private CpuTADManager tadManager = new CpuTADManager();

    //thread locals for custom op inputs and outputs to prevent allocations
    //every time exec(CustomOp) is called
    private ThreadLocal<Map<Integer,PointerPointer>> inputShapes = new ThreadLocal<>();
    private ThreadLocal<Map<Integer,PointerPointer>> inputBuffers = new ThreadLocal<>();
    private ThreadLocal<Map<Integer,PointerPointer>> outputShapes = new ThreadLocal<>();
    private ThreadLocal<Map<Integer,PointerPointer>> outputBuffers = new ThreadLocal<>();
    private ThreadLocal<Map<Integer,LongPointer>> iArgsPointer = new ThreadLocal<>();
    private ThreadLocal<Map<Integer,DoublePointer>> tArgsPointer = new ThreadLocal<>();
    private ThreadLocal<Map<Integer,BooleanPointer>> bArgsPointer = new ThreadLocal<>();
    private ThreadLocal<Map<Integer,ShortPointer>> halfArgsPointer = new ThreadLocal<>();

    protected Map<String, CustomOpDescriptor> customOps = null;

    protected ThreadLocal<PointerPointer> extraz = new ThreadLocal<>();

    protected AtomicBoolean experimentalMode = new AtomicBoolean(false);

    protected Map<String, Boolean> mklOverrides = new HashMap<>();

    /**
     * Instead of allocating new memory chunks for each batch invocation, we reuse them on thread/opNum basis
     * Since for NativeOpExecutioner all executions are synchronous
     */
    private ThreadLocal<Map<Integer, Pointer>> batchPointers = new ThreadLocal<>();
    private ThreadLocal<Map<Integer, AggregateMemoryBlock>> memoryBlocks = new ThreadLocal<>();

    public NativeOpExecutioner() {
        tadManager.init(loop, constantHandler);

        experimentalMode.set(loop.isExperimentalEnabled());
/*
        // filling vars for possible overrides
        val env = System.getenv(ND4JEnvironmentVars.ND4J_MKL_FALLBACK);
        if (env != null) {
            // in this case we just disable mkl-dnn globally
            if (env.equalsIgnoreCase("true")) {
                Nd4jCpu.Environment.getInstance().setUseMKLDNN(false);
            } else {
                val split = env.toLowerCase().split(",");
                for (val name:split) {
                    mklOverrides.put(name, new Boolean(true));
                }
            }
        }
*/
    }

    @Override
    public INDArray exec(Op op) {
        return exec(op, null);
    }

    @Override
    public INDArray exec(Op op, OpContext opContext) {
        checkForCompression(op);

        if (op instanceof ScalarOp) {
            ScalarOp s = (ScalarOp) op;
            exec(s, opContext);
        } else if (op instanceof TransformOp) {
            TransformOp t = (TransformOp) op;
            exec(t, opContext);
        } else if (op instanceof ReduceOp) {
            ReduceOp ac = (ReduceOp) op;
            exec(ac, opContext);
        } else if (op instanceof IndexAccumulation) {
            IndexAccumulation iac = (IndexAccumulation) op;
            exec(iac, opContext); //Currently using DefaultOpExecutioner
        } else if (op instanceof BroadcastOp) {
            BroadcastOp broadcastOp = (BroadcastOp) op;
            exec(broadcastOp, opContext);
        } else if (op instanceof RandomOp) {
            RandomOp rngOp = (RandomOp) op;
            exec(rngOp, opContext, Nd4j.getRandom());
        }

        return op.z();
    }

    @Override
    public INDArray exec(IndexAccumulation op) {
        return exec(op, null);
    }

    public INDArray exec(IndexAccumulation op, OpContext oc) {
        checkForCompression(op);

        INDArray x = getX(op, oc);
        INDArray z = getZ(op, oc);

        if (extraz.get() == null)
            extraz.set(new PointerPointer(32));

        val dimension = Shape.normalizeAxis(op.x().rank(), op.dimensions().toIntVector());

        if (op.x().isEmpty()) {
            for (val d:dimension) {
                Preconditions.checkArgument(op.x().shape()[d] != 0, "IndexReduce can't be issued along axis with 0 in shape");
            }
        }

        boolean keepDims = op.isKeepDims();
        long[] retShape = Shape.reductionShape(op.x(), dimension, true, keepDims);

        if(op.z() == null || op.x() == op.z()) {
            val ret = Nd4j.createUninitialized(DataType.LONG, retShape);

            op.setZ(ret);
        } else if(!Arrays.equals(retShape, op.z().shape())){
            throw new IllegalStateException("Z array shape does not match expected return type for op " + op
                    + ": expected shape " + Arrays.toString(retShape) + ", z.shape()=" + Arrays.toString(op.z().shape()));
        }

        op.validateDataTypes();

        Pointer dimensionAddress = constantHandler.getConstantBuffer(dimension, DataType.INT).addressPointer();

        Pair<DataBuffer, DataBuffer> tadBuffers = tadManager.getTADOnlyShapeInfo(op.x(), dimension);

        Pointer hostTadShapeInfo = tadBuffers.getFirst().addressPointer();

        DataBuffer offsets = tadBuffers.getSecond();
        Pointer hostTadOffsets = offsets == null ? null : offsets.addressPointer();

        PointerPointer dummy = extraz.get().put(hostTadShapeInfo, hostTadOffsets);

        long st = profilingConfigurableHookIn(op, tadBuffers.getFirst());

        val xb = ((BaseAuroraDataBuffer) x.data()).getOpaqueDataBuffer();
        val zb = ((BaseAuroraDataBuffer) z.data()).getOpaqueDataBuffer();

        if (z.isScalar()) {
            loop.execIndexReduceScalar(dummy, op.opNum(),
                    xb, (LongPointer) x.shapeInfoDataBuffer().addressPointer(), null,
                    getPointerForExtraArgs(op, x.dataType()),
                    zb, (LongPointer) z.shapeInfoDataBuffer().addressPointer(), null);
        } else {
            loop.execIndexReduce(dummy, op.opNum(),
                    xb, (LongPointer) x.shapeInfoDataBuffer().addressPointer(), null,
                    getPointerForExtraArgs(op, x.dataType()),
                    zb, (LongPointer) z.shapeInfoDataBuffer().addressPointer(), null,
                    ((BaseAuroraDataBuffer) op.dimensions().data()).getOpaqueDataBuffer(), (LongPointer) op.dimensions().shapeInfoDataBuffer().addressPointer(), null);
        }

        if (loop.lastErrorCode() != 0)
            throw new RuntimeException(loop.lastErrorMessage());

        profilingConfigurableHookOut(op, st);
        return getZ(op, oc);
    }

    @Override
    public INDArray exec(Variance op) {
        return exec((ReduceOp) op);
    }

    @Override
    public INDArray exec(ReduceOp op) {
        return exec(op, null);
    }

    public INDArray exec(ReduceOp op, OpContext oc) {
        INDArray x = getX(op, oc);
        INDArray y = getY(op, oc);
        INDArray z = getZ(op, oc);

        Preconditions.checkNotNull(op.x(), "Op.x() cannot be null: Was null for op %s", op);
        op.validateDataTypes(oc);

        if(op instanceof BaseReduceOp && ((BaseReduceOp)op).isEmptyReduce()){
            //Edge case for TF import compatibility: [x,y].reduce(empty) = [x,y]
            //Note that "empty" axis is NOT the same as length 0, as in INDArray.sum(new int[0]), which means "all dimensions"
            if(op.z() != null){
                Preconditions.checkState(op.x().equalShapes(op.z()), "For empty reductions, result (z) array must have same shape as x shape." +
                        " Got: x=%ndShape, z=%ndShape", op.x(), op.z());
                op.z().assign(op.x());
                return op.z();
            } else {
                op.setZ(op.x().dup());
                return op.z();
            }
        }

        val dimension = Shape.normalizeAxis(op.x().rank(), op.dimensions().toIntVector());

        //validateDataType(Nd4j.dataType(), op);

        if (extraz.get() == null)
            extraz.set(new PointerPointer(32));

        boolean keepDims = op.isKeepDims();
        long[] retShape = Shape.reductionShape(op.x(), dimension, true, keepDims);


        if (op.x().isVector() && op.x().length() == ArrayUtil.prod(retShape) && ArrayUtil.prodLong(retShape) > 1 && op.y() == null)
            return op.noOp();

        /**
         * This is the result array.
         * We create it only if we hadn't provided it before
         */
        INDArray ret;
        if (op.z() == null || op.z() == op.x()) {
            if (op.isComplexAccumulation()) {
                long xT = op.x().tensorsAlongDimension(dimension);
                long yT = op.y().tensorsAlongDimension(dimension);

                ret = Nd4j.create(op.resultType(), new long[]{xT, yT});
            } else {
                if (op.y() != null) {

                    //2 options here: either pairwise, equal sizes - OR every X TAD vs. entirety of Y
                    if(op.x().length() == op.y().length()) {
                        //Pairwise
                        if (op.x().tensorsAlongDimension(dimension) != op.y().tensorsAlongDimension(dimension)) {
                            throw new ND4JIllegalStateException("Number of TADs along dimension don't match: (x shape = " +
                                    Arrays.toString(op.x().shape()) + ", y shape = " + Arrays.toString(op.y().shape()) +
                                    ", dimension = " + Arrays.toString(dimension) + ")");
                        }
                    } else {
                        //Every X TAD vs. entirety of Y
                        val xTADSize = op.x().length() / op.x().tensorsAlongDimension(dimension);

                        if (xTADSize != op.y().length()) {
                            throw new ND4JIllegalStateException("Size of TADs along dimension don't match for pairwise execution:" +
                                    " (x TAD size = " + xTADSize + ", y size = " + op.y().length());
                        }
                    }
                }

                ret = Nd4j.create(op.resultType(), retShape);

            }
            op.setZ(ret);
        } else {
            // compare length
            long shapeProduct = (retShape.length == 0 ? 1 : ArrayUtil.prodLong(retShape));
            if (!op.isComplexAccumulation() && op.z().length() != shapeProduct) {
                if(!(op.x().isEmpty() && op.isKeepDims())){
                    //Empty reductions are special case: [1,0].sum(0,1,keep=true) -> shape [1,1]
                    throw new ND4JIllegalStateException("Shape of target array for reduction [" + Arrays.toString(op.z().shape()) + "] doesn't match expected [" + Arrays.toString(retShape) + "]");
                }
            }
            else if (op.isComplexAccumulation()) {
                long xT = op.x().tensorsAlongDimension(dimension);
                long yT = op.y().tensorsAlongDimension(dimension);

                if (op.z().length() != xT * yT)
                    throw new ND4JIllegalStateException("Shape of target array for reduction [" + Arrays.toString(op.z().shape()) + "] doesn't match expected [" + (xT * yT) + "]");
            }

            ret = op.z();
        }

        //log.info("X dtype: {}; Z dtype: {}", op.x().dataType(), op.z().dataType());

        /**
         * Returns the {@link Shape#createShapeInformation(int[], int[], int, int, char)}
         * and the associated offsets for each {@link INDArray#tensorAlongDimension(int, int...)}
         * The first item is the shape information. The second one is the offsets.
         */
        Pair<DataBuffer, DataBuffer> tadBuffers = op.x().isEmpty() ? Pair.<DataBuffer, DataBuffer>makePair(op.x().data(), null): tadManager.getTADOnlyShapeInfo(op.x(), dimension);
        Pair<DataBuffer, DataBuffer> yTadBuffers = null;
        /**
         * Note that we use addresses in libnd4j.
         * We use reinterpret cast in c to take the long
         * we pass to JNI. This manages overhead.
         */
        Pointer hostTadShapeInfo = op.x().isEmpty() ? op.x().shapeInfoDataBuffer().addressPointer() : tadBuffers.getFirst().addressPointer();

        DataBuffer offsets = op.x().isEmpty() ? null : tadBuffers.getSecond();
        Pointer hostTadOffsets = offsets == null ? null : offsets.addressPointer();

        // we're going to check, if that's TAD vs TAD comparison or TAD vs full array. if later - we're going slightly different route
        boolean tvf = false;
        if (op.y() != null) {
            if (op.x().tensorAlongDimension(0, dimension).length() == op.y().length()) {
                tvf = true;
            }
        }

        if (op.isComplexAccumulation()) {
            yTadBuffers = tadManager.getTADOnlyShapeInfo(op.y(), dimension);

            if (op.x().tensorAlongDimension(0, dimension).length() != op.y().tensorAlongDimension(0, dimension).length())
                throw new ND4JIllegalStateException("Impossible to issue AllDistances operation: TAD lengths mismatch along given dimension: " +
                        "x TAD length = " + op.x().tensorAlongDimension(0, dimension).length() + ", y TAD length " +
                        op.y().tensorAlongDimension(0, dimension).length());
        }

        /**
         * This is a pointer to a pointer in c.
         */
        //  FIXME: we need something better then 3rd element being non-null here...
        //PointerPointer dummy = extraz.get().put(hostTadShapeInfo, hostTadOffsets, tvf ? hostTadOffsets : null);

        long st = profilingConfigurableHookIn(op, tadBuffers.getFirst());

        /**
         * Note because dimension arrays don't change,
         * we use an {@link ConstantHandler} which knows how to reserve memory
         * for immutable buffers for the dimensions.
         * This gives us a pointer which is passed around in libnd4j.
         */
        Pointer dimensionAddress = constantHandler.getConstantBuffer(dimension, DataType.INT).addressPointer();
        val xb = ((BaseAuroraDataBuffer) x.data()).getOpaqueDataBuffer();
        val zb = ((BaseAuroraDataBuffer) z.data()).getOpaqueDataBuffer();

            if (op instanceof Variance) {
                if (ret.isScalar()) {
                    loop.execSummaryStatsScalar(null, op.opNum(),
                            xb, (LongPointer) x.shapeInfoDataBuffer().addressPointer(), null,
                            getPointerForExtraArgs(op, z.dataType()),
                            zb, (LongPointer) z.shapeInfoDataBuffer().addressPointer(), null,
                            ((Variance) op).isBiasCorrected());
                } else {
                    Variance var = (Variance) op;
                    try {
                        loop.execSummaryStatsTad(null, op.opNum(),
                                xb, (LongPointer) x.shapeInfoDataBuffer().addressPointer(), null,
                                getPointerForExtraArgs(op, z.dataType()),
                                zb, (LongPointer) z.shapeInfoDataBuffer().addressPointer(), null,
                                ((BaseAuroraDataBuffer) op.dimensions().data()).getOpaqueDataBuffer(), (LongPointer) op.dimensions().shapeInfoDataBuffer().addressPointer(), null,
                                var.isBiasCorrected(), null, null);
                    } catch (Throwable t){
                        String str = opInfoString(op, Optional.of(dimension));
                        throw new RuntimeException("Native AccumulationOp execution (double) failed: " + str, t);
                    }
                }

            }
            //pairwise reduction like similarity of two arrays
            else if (op.y() != null && op.getOpType() == Op.Type.REDUCE3) {
                val yb = ((BaseAuroraDataBuffer) y.data()).getOpaqueDataBuffer();
                if (op.isComplexAccumulation()) {
                    try {
                        loop.execReduce3All(null, op.opNum(),
                                xb, (LongPointer) x.shapeInfoDataBuffer().addressPointer(), null,
                                getPointerForExtraArgs(op, z.dataType()),
                                yb, (LongPointer) y.shapeInfoDataBuffer().addressPointer(), null,
                                zb, (LongPointer) z.shapeInfoDataBuffer().addressPointer(), null,
                                ((BaseAuroraDataBuffer) op.dimensions().data()).getOpaqueDataBuffer(), (LongPointer) op.dimensions().shapeInfoDataBuffer().addressPointer(), null,
                                (LongPointer) tadBuffers.getFirst().addressPointer(), new LongPointerWrapper(tadBuffers.getSecond().addressPointer()),
                                (LongPointer) yTadBuffers.getFirst().addressPointer(), new LongPointerWrapper(yTadBuffers.getSecond().addressPointer())
                        );
                    } catch (Throwable t){
                        String str = opInfoString(op, Optional.of(dimension));
                        throw new RuntimeException("Native AccumulationOp execution (double) failed: " + str, t);
                    }
                } else if (ret.isScalar()) {
                    loop.execReduce3Scalar(null, op.opNum(),
                            xb, (LongPointer) x.shapeInfoDataBuffer().addressPointer(), null,
                            getPointerForExtraArgs(op, z.dataType()),
                            yb, (LongPointer) y.shapeInfoDataBuffer().addressPointer(), null,
                            zb, (LongPointer) ret.shapeInfoDataBuffer().addressPointer(), null);
                } else {
                    try {
                        loop.execReduce3Tad(null, op.opNum(),
                                xb, (LongPointer) x.shapeInfoDataBuffer().addressPointer(), null,
                                getPointerForExtraArgs(op, z.dataType()),
                                yb, (LongPointer) y.shapeInfoDataBuffer().addressPointer(), null,
                                zb, (LongPointer) z.shapeInfoDataBuffer().addressPointer(), null,
                                ((BaseAuroraDataBuffer) op.dimensions().data()).getOpaqueDataBuffer(), (LongPointer) op.dimensions().shapeInfoDataBuffer().addressPointer(), null,
                                null, null, null, null);
                    } catch (Throwable t){
                        String str = opInfoString(op, Optional.of(dimension));
                        throw new RuntimeException("Native AccumulationOp execution (double) failed: " + str, t);
                    }
                }

            } else {
                if (ret.isScalar()) {
                    switch (op.getOpType()) {
                        case REDUCE_FLOAT:
                            loop.execReduceFloat(null, op.opNum(),
                                    xb, (LongPointer) x.shapeInfoDataBuffer().addressPointer(), null,
                                    getPointerForExtraArgs(op, z.dataType()),
                                    zb, (LongPointer) ret.shapeInfoDataBuffer().addressPointer(), null);
                            break;
                        case REDUCE_BOOL:
                            loop.execReduceBool(null, op.opNum(),
                                    xb, (LongPointer) x.shapeInfoDataBuffer().addressPointer(), null,
                                    getPointerForExtraArgs(op, x.dataType()),
                                    zb, (LongPointer) ret.shapeInfoDataBuffer().addressPointer(), null);
                            break;
                        case REDUCE_SAME:
                            loop.execReduceSame(null, op.opNum(),
                                    xb, (LongPointer) x.shapeInfoDataBuffer().addressPointer(), null,
                                    getPointerForExtraArgs(op, x.dataType()),
                                    zb, (LongPointer) ret.shapeInfoDataBuffer().addressPointer(), null);
                            break;
                        case REDUCE_LONG:
                            loop.execReduceLong(null, op.opNum(),
                                    xb, (LongPointer) x.shapeInfoDataBuffer().addressPointer(), null,
                                    getPointerForExtraArgs(op, x.dataType()),
                                    zb, (LongPointer) ret.shapeInfoDataBuffer().addressPointer(), null);
                            break;
                        default:
                            throw new UnsupportedOperationException("Unsupported op used in reduce: "+ op.getOpType());
                    }
                } else {
                    switch (op.getOpType()) {
                        case REDUCE_FLOAT:
                            loop.execReduceFloat2(null, op.opNum(),
                                    xb, (LongPointer) x.shapeInfoDataBuffer().addressPointer(), null,
                                    getPointerForExtraArgs(op, z.dataType()),
                                    zb, (LongPointer) z.shapeInfoDataBuffer().addressPointer(), null,
                                    ((BaseAuroraDataBuffer) op.dimensions().data()).getOpaqueDataBuffer(), (LongPointer) op.dimensions().shapeInfoDataBuffer().addressPointer(), null);
                            break;
                        case REDUCE_LONG:
                            loop.execReduceLong2(null, op.opNum(),
                                    xb, (LongPointer) x.shapeInfoDataBuffer().addressPointer(), null,
                                    getPointerForExtraArgs(op, x.dataType()),
                                    zb, (LongPointer) z.shapeInfoDataBuffer().addressPointer(), null,
                                    ((BaseAuroraDataBuffer) op.dimensions().data()).getOpaqueDataBuffer(),
                                    (LongPointer) op.dimensions().shapeInfoDataBuffer().addressPointer(), null);
                            break;
                        case REDUCE_SAME:
                            loop.execReduceSame2(null, op.opNum(),
                                    xb, (LongPointer) x.shapeInfoDataBuffer().addressPointer(), null,
                                    getPointerForExtraArgs(op, z.dataType()),
                                    zb, (LongPointer) z.shapeInfoDataBuffer().addressPointer(), null,
                                    ((BaseAuroraDataBuffer) op.dimensions().data()).getOpaqueDataBuffer(),
                                    (LongPointer) op.dimensions().shapeInfoDataBuffer().addressPointer(), null);
                            break;
                        case REDUCE_BOOL:
                            loop.execReduceBool2(null, op.opNum(),
                                    xb, (LongPointer) x.shapeInfoDataBuffer().addressPointer(), null,
                                    getPointerForExtraArgs(op, x.dataType()),
                                    zb, (LongPointer) z.shapeInfoDataBuffer().addressPointer(), null,
                                    ((BaseAuroraDataBuffer) op.dimensions().data()).getOpaqueDataBuffer(),
                                    (LongPointer) op.dimensions().shapeInfoDataBuffer().addressPointer(), null);
                            break;
                        default:
                            throw new UnsupportedOperationException("Unsupported op used in reduce: "+ op.getOpType());
                    }
                }
            }

        if (loop.lastErrorCode() != 0)
            throw new RuntimeException(loop.lastErrorMessage());

        return ret;
    }

    /**
     * ScalarOp execution
     * @param op Op to execute
     */
    private void invokeScalarAlongDimension(ScalarOp op) {
        invokeScalarAlongDimension(op, null);
    }

    private void invokeScalarAlongDimension(ScalarOp op, OpContext oc) {
        INDArray x = getX(op, oc);
        INDArray y = getY(op, oc);
        INDArray z = getZ(op, oc);

        val dimension = op.dimensions().toIntVector();
        //dimension = Shape.normalizeAxis(op.x().rank(), dimension);
        // do tad magic
        /**
         * Returns the {@link Shape#createShapeInformation(int[], int[], int, int, char)}
         * and the associated offsets for each {@link INDArray#tensorAlongDimension(int, int...)}
         * The first item is the shape information. The second one is the offsets.
         */
        Pair<DataBuffer, DataBuffer> tadBuffers = tadManager.getTADOnlyShapeInfo(op.x(), dimension);

        Pointer hostTadShapeInfo = tadBuffers.getFirst().addressPointer();
        Pointer hostTadOffsets = tadBuffers.getSecond().addressPointer();

        Pointer devTadShapeInfoZ = null;
        Pointer devTadOffsetsZ = null;
        /**
         * Returns the {@link Shape#createShapeInformation(int[], int[], int, int, char)}
         * and the associated offsets for each {@link INDArray#tensorAlongDimension(int, int...)}
         * The first item is the shape information. The second one is the offsets.
         *
         * Note that this is the *result* TAD information. An op is always input (x) and output (z)
         * for result.
         * This is for assigning the result to of the operation along
         * the proper dimension.
         */
        Pair<DataBuffer, DataBuffer> tadBuffersZ = tadManager.getTADOnlyShapeInfo(op.z(), dimension);

        devTadShapeInfoZ = tadBuffersZ.getFirst().addressPointer();
        devTadOffsetsZ = tadBuffersZ.getSecond().addressPointer();

        if (extraz.get() == null)
            extraz.set(new PointerPointer(32));

        //PointerPointer dummy = extraz.get().put(hostTadShapeInfo, hostTadOffsets, devTadShapeInfoZ, devTadOffsetsZ);
        val xb = ((BaseAuroraDataBuffer) x.data()).getOpaqueDataBuffer();
        val yb = ((BaseAuroraDataBuffer) y.data()).getOpaqueDataBuffer();
        val zb = ((BaseAuroraDataBuffer) z.data()).getOpaqueDataBuffer();

        switch (op.getOpType()) {
            case SCALAR:
                loop.execScalarTad(null, op.opNum(),
                        xb, (LongPointer) op.x().shapeInfoDataBuffer().addressPointer(), null,
                        zb, (LongPointer) op.z().shapeInfoDataBuffer().addressPointer(), null,
                        yb, (LongPointer) y.shapeInfoDataBuffer().addressPointer(), null,
                        getPointerForExtraArgs(op, op.z().dataType()),
                        ((BaseAuroraDataBuffer) op.dimensions().data()).getOpaqueDataBuffer(), (LongPointer) op.dimensions().shapeInfoDataBuffer().addressPointer(),null,
                        (LongPointer) hostTadShapeInfo, (LongPointer) hostTadOffsets,
                        (LongPointer) devTadShapeInfoZ, (LongPointer) devTadOffsetsZ);
                break;
            case SCALAR_BOOL:
                loop.execScalarBoolTad(null, op.opNum(),
                        xb, (LongPointer) op.x().shapeInfoDataBuffer().addressPointer(), null,
                        zb, (LongPointer) op.z().shapeInfoDataBuffer().addressPointer(), null,
                        yb, (LongPointer) op.y().shapeInfoDataBuffer().addressPointer(), null,
                        getPointerForExtraArgs(op, op.z().dataType()),
                        ((BaseAuroraDataBuffer) op.dimensions().data()).getOpaqueDataBuffer(), (LongPointer) op.dimensions().shapeInfoDataBuffer().addressPointer(), null,
                        (LongPointer) hostTadShapeInfo, (LongPointer) hostTadOffsets,
                        (LongPointer) devTadShapeInfoZ, (LongPointer) devTadOffsetsZ);
                break;
            default:
                throw new UnsupportedOperationException();
        }

        if (loop.lastErrorCode() != 0)
            throw new RuntimeException(loop.lastErrorMessage());
    }

    public INDArray exec(ScalarOp op){
        return exec(op, null);
    }

    public INDArray exec(ScalarOp op, OpContext oc) {
        long st = profilingConfigurableHookIn(op);

        //validateDataType(Nd4j.dataType(), op);

        if (op.x().length() != op.z().length())
            throw new ND4JIllegalStateException("op.X length should be equal to op.Z length: " +
                    "x.length()=" + op.x().length() + ", z.length()=" + op.z().length() + " - x shape info = ["
                    + Arrays.toString(op.x().shapeInfoDataBuffer().asInt()) + "], z shape info = ["
                    + Arrays.toString(op.z().shapeInfoDataBuffer().asInt()) + "]");

        if (op.dimensions() != null) {
            invokeScalarAlongDimension(op);
            return op.z();
        }

        val x = ((BaseAuroraDataBuffer) getX(op, oc).data()).getOpaqueDataBuffer();
        val scalar = ((BaseAuroraDataBuffer) op.scalar().data()).getOpaqueDataBuffer();
        val z = ((BaseAuroraDataBuffer) getZ(op, oc).data()).getOpaqueDataBuffer();


        switch (op.getOpType()) {
            case SCALAR:
                loop.execScalar(null,
                        op.opNum(),
                        x, (LongPointer) getX(op, oc).shapeInfoDataBuffer().addressPointer(), null,
                        z, (LongPointer) getZ(op, oc).shapeInfoDataBuffer().addressPointer(), null,
                        scalar, (LongPointer) op.scalar().shapeInfoDataBuffer().addressPointer(), null,
                        getPointerForExtraArgs(op, getZ(op, oc).dataType()));
                break;
            case SCALAR_BOOL:
                loop.execScalarBool(null,
                        op.opNum(),
                        x, (LongPointer) getX(op, oc).shapeInfoDataBuffer().addressPointer(), null,
                        z, (LongPointer) getZ(op, oc).shapeInfoDataBuffer().addressPointer(), null,
                        scalar, (LongPointer) op.scalar().shapeInfoDataBuffer().addressPointer(), null,
                        getPointerForExtraArgs(op, getX(op, oc).dataType()));
                break;
            default:
                throw new ND4JIllegalStateException("Unknown op type: [" + op.getOpType() +"]");
        }

        if (loop.lastErrorCode() != 0)
            throw new RuntimeException(loop.lastErrorMessage());

        profilingConfigurableHookOut(op, st);

        return op.z();
    }

    private Pointer getPointerForExtraArgs(Op op, DataType type) {
        if (op.extraArgs() != null){
            val eadb = op.extraArgsDataBuff(type);
            if (eadb != null)
                return eadb.addressPointer();
            else
                return null;
        }

        return null;
    }

    private void exec(TransformOp op) {
        exec(op, null);
    }

    private void exec(TransformOp op, OpContext oc) {
        INDArray x = getX(op, oc);
        INDArray y = getY(op, oc);
        INDArray z = getZ(op, oc);

        long st = 0;

//        validateDataType(Nd4j.dataType(), op);

        if (extraz.get() == null)
            extraz.set(new PointerPointer(32));

        PointerPointer dummy = extraz.get();

        // Pow operations might be special
        if (op.opNum() == 31) {
            if (op.y() != null && op.y().isScalar()) {
                op.setY(Nd4j.valueArrayOf(op.x().shape(), op.y().getDouble(0)));
            }
        }

        /**
         * This is the {@link IsMax}
         * operation.
         *
         * @see {@link Op#extraArgs()}
         * for what an extra argument is in an op.
         *
         * The extra argument in the op here is the {@link IsMax#IsMax(INDArray, int...)}
         * dimension to do the ismax along
         */
        if (op.opName().equalsIgnoreCase("ismax") && op.extraArgs() != null && op.extraArgs().length > 0) {
            int[] dimension = new int[(int) op.extraArgs()[0]];

            for (int i = 0; i < dimension.length; i++) {
                dimension[i] = (int) op.extraArgs()[i + 1];
            }


            /**
             * Returns the {@link Shape#createShapeInformation(int[], int[], int, int, char)}
             * and the associated offsets for each {@link INDArray#tensorAlongDimension(int, int...)}
             * The first item is the shape information. The second one is the offsets.
             */
            Pair<DataBuffer, DataBuffer> tadBuffers = tadManager.getTADOnlyShapeInfo(op.z(), dimension);


            Pointer tad = tadBuffers.getFirst().addressPointer();

            DataBuffer offsets = tadBuffers.getSecond();
            Pointer off = offsets == null ? null : offsets.addressPointer();
            dummy.put(tad, off);

            st = profilingConfigurableHookIn(op, tadBuffers.getFirst());
        } else
            st = profilingConfigurableHookIn(op);

            if (op.y() != null) {

                if (op.z() == null)
                    op.setZ(Nd4j.create(op.resultType(), op.x().shape()));


                op.validateDataTypes(oc, experimentalMode.get());

                //log.info("X type: {}; Y type: {}; Z type: {}; OpNum: {}", op.x().dataType(), op.y().dataType(), op.z().dataType(), op.opNum());

                int xEWS = op.x().elementWiseStride();
                int yEWS = op.y().elementWiseStride();
                int zEWS = op.z().elementWiseStride();

                boolean xRow = op.x().isRowVector();
                boolean yRow = op.y().isRowVector();
                boolean zRow = op.z().isRowVector();

                if (op.x().length() != op.y().length() || op.x().length() != op.z().length())
                    throw new ND4JIllegalStateException("X, Y and Z arguments should have the same length for PairwiseTransform " +
                            op.opName() + ". x: length " + op.x().length() + ", shape " + Arrays.toString(op.x().shape()) +
                            "; y: " + op.y().length() + ", shape " + Arrays.toString(op.y().shape()) +
                            "; z: " + op.z().length() + ", shape " + Arrays.toString(op.z().shape()));

                val xb = ((BaseAuroraDataBuffer) x.data()).getOpaqueDataBuffer();
                val yb = ((BaseAuroraDataBuffer) y.data()).getOpaqueDataBuffer();
                val zb = ((BaseAuroraDataBuffer) z.data()).getOpaqueDataBuffer();

                switch (op.getOpType()) {
                    case TRANSFORM_ANY:
                    case TRANSFORM_FLOAT:
                    case TRANSFORM_STRICT:
                    case TRANSFORM_SAME:
                        if (!experimentalMode.get())
                            Preconditions.checkArgument(x.dataType() == y.dataType() || y.dataType() == DataType.BOOL,
                                    "Op.X and Op.Y must have the same data type, but got %s vs. %s", x.dataType(), y.dataType());

                        loop.execPairwiseTransform(dummy, op.opNum(),
                                xb, (LongPointer) x.shapeInfoDataBuffer().addressPointer(), null,
                                yb, (LongPointer) y.shapeInfoDataBuffer().addressPointer(), null,
                                zb, (LongPointer) z.shapeInfoDataBuffer().addressPointer(), null,
                                getPointerForExtraArgs(op, z.dataType()));
                        break;
                    case TRANSFORM_BOOL:
                    case PAIRWISE_BOOL:
                        loop.execPairwiseTransformBool(dummy, op.opNum(),
                                xb, (LongPointer) x.shapeInfoDataBuffer().addressPointer(), null,
                                yb, (LongPointer) y.shapeInfoDataBuffer().addressPointer(), null,
                                zb, (LongPointer) z.shapeInfoDataBuffer().addressPointer(), null,
                                getPointerForExtraArgs(op, x.dataType()));
                        break;
                }
            } else {

                if (op.z() == null)
                    op.setZ(Nd4j.create(op.resultType(), op.x().shape()));

                op.validateDataTypes(oc, experimentalMode.get());

                val xb = ((BaseAuroraDataBuffer) x.data()).getOpaqueDataBuffer();
                val zb = ((BaseAuroraDataBuffer) z.data()).getOpaqueDataBuffer();

                switch (op.getOpType()) {
                    case TRANSFORM_FLOAT: {
                        val xtraz = getPointerForExtraArgs(op, z.dataType());

                        loop.execTransformFloat(dummy, op.opNum(),
                                xb, (LongPointer) x.shapeInfoDataBuffer().addressPointer(), null,
                                zb, (LongPointer) z.shapeInfoDataBuffer().addressPointer(),
                                null, xtraz);
                        break;
                    }
                    case TRANSFORM_STRICT: {
                        val xtraz = getPointerForExtraArgs(op, z.dataType());

                        loop.execTransformStrict(dummy, op.opNum(),
                                xb, (LongPointer) x.shapeInfoDataBuffer().addressPointer(), null,
                                zb, (LongPointer) z.shapeInfoDataBuffer().addressPointer(), null,
                                xtraz);
                        break;
                    }
                    case TRANSFORM_SAME: {
                        val xtraz = getPointerForExtraArgs(op, z.dataType());

                        loop.execTransformSame(dummy, op.opNum(),
                                xb, (LongPointer) x.shapeInfoDataBuffer().addressPointer(), null,
                                zb, (LongPointer) z.shapeInfoDataBuffer().addressPointer(), null,
                                xtraz);
                        break;
                    }
                    case TRANSFORM_ANY: {
                        val xtraz = getPointerForExtraArgs(op, x.dataType());
                        val opNum = op.opNum();

                        loop.execTransformAny(dummy, opNum,
                                xb, (LongPointer) x.shapeInfoDataBuffer().addressPointer(), null,
                                zb, (LongPointer) z.shapeInfoDataBuffer().addressPointer(), null,
                                xtraz);
                        break;
                    }
                    case TRANSFORM_BOOL: {
                        val xtraz = getPointerForExtraArgs(op, x.dataType());
                        val opNum = op.opNum();

                        loop.execTransformBool(dummy, opNum,
                                xb, (LongPointer) x.shapeInfoDataBuffer().addressPointer(), null,
                                zb, (LongPointer) z.shapeInfoDataBuffer().addressPointer(), null,
                                xtraz);
                        break;
                    }
                    default:
                        throw new UnsupportedOperationException("Unknown transform type: [" + op.getOpType() + "]");
                }

            }

        if (loop.lastErrorCode() != 0)
            throw new RuntimeException(loop.lastErrorMessage());

        profilingConfigurableHookOut(op, st);
    }

    public INDArray exec(BroadcastOp op) {
        return exec(op, null);
    }

    public INDArray exec(BroadcastOp op, OpContext oc) {
        INDArray x = getX(op, oc);
        INDArray y = getY(op, oc);
        INDArray z = getZ(op, oc);

        long st = profilingConfigurableHookIn(op);

        op.validateDataTypes(experimentalMode.get());

        val dimension = op.dimensions().toIntVector();

        /**
         * Returns the {@link Shape#createShapeInformation(int[], int[], int, int, char)}
         * and the associated offsets for each {@link INDArray#tensorAlongDimension(int, int...)}
         * The first item is the shape information. The second one is the offsets.
         */
        Pair<DataBuffer, DataBuffer> tadBuffers = tadManager.getTADOnlyShapeInfo(op.x(), dimension);

        Pointer hostTadShapeInfo = tadBuffers.getFirst().addressPointer();
        Pointer hostTadOffsets = tadBuffers.getSecond().addressPointer();

        Pointer devTadShapeInfoZ = null;
        Pointer devTadOffsetsZ = null;

        //        if (!Arrays.equals(op.x().shape(),op.z().shape()) || !Arrays.equals(op.x().stride(),op.z().stride()) || op.x().ordering() != op.z().ordering()) {
        // that's the place where we're going to have second TAD in place
        Pair<DataBuffer, DataBuffer> tadBuffersZ = tadManager.getTADOnlyShapeInfo(op.z(), dimension);

        devTadShapeInfoZ = tadBuffersZ.getFirst().addressPointer();
        devTadOffsetsZ = tadBuffersZ.getSecond().addressPointer();
        /*
        log.info("Broascast dimension: {}", Arrays.toString(dimension));
        log.info("x shape: {}; x TAD: {}; comp TAD: {}", Arrays.toString(op.x().shapeInfoDataBuffer().asInt()), Arrays.toString(tadBuffers.getFirst().asInt()), Arrays.toString(op.x().tensorAlongDimension(0, dimension).shapeInfoDataBuffer().asInt()));
        log.info("z shape: {}; z TAD: {}", Arrays.toString(op.z().shapeInfoDataBuffer().asInt()), Arrays.toString(tadBuffersZ.getFirst().asInt()));
        log.info("y shape: {}", Arrays.toString(op.y().shapeInfoDataBuffer().asInt()));
        log.info("-------------");
        */

        if (extraz.get() == null)
            extraz.set(new PointerPointer(32));

        PointerPointer dummy = extraz.get().put(hostTadShapeInfo, hostTadOffsets, devTadShapeInfoZ, devTadOffsetsZ);

        Pointer dimensionAddress = constantHandler.getConstantBuffer(dimension, DataType.INT).addressPointer();

        val xb = ((BaseAuroraDataBuffer) x.data()).getOpaqueDataBuffer();
        val yb = ((BaseAuroraDataBuffer) y.data()).getOpaqueDataBuffer();
        val zb = ((BaseAuroraDataBuffer) z.data()).getOpaqueDataBuffer();

        switch (op.getOpType()) {
            case BROADCAST:
                loop.execBroadcast(dummy, op.opNum(),
                        xb, (LongPointer) x.shapeInfoDataBuffer().addressPointer(), null,
                        yb, (LongPointer) y.shapeInfoDataBuffer().addressPointer(), null,
                        zb, (LongPointer) z.shapeInfoDataBuffer().addressPointer(), null,
                        ((BaseAuroraDataBuffer) op.dimensions().data()).getOpaqueDataBuffer(), (LongPointer) op.dimensions().shapeInfoDataBuffer().addressPointer(), null);
                break;
            case BROADCAST_BOOL:
                loop.execBroadcastBool(dummy, op.opNum(),
                        xb, (LongPointer) x.shapeInfoDataBuffer().addressPointer(), null,
                        yb, (LongPointer) y.shapeInfoDataBuffer().addressPointer(), null,
                        zb, (LongPointer) z.shapeInfoDataBuffer().addressPointer(), null,
                        null,
                        ((BaseAuroraDataBuffer) op.dimensions().data()).getOpaqueDataBuffer(), (LongPointer) op.dimensions().shapeInfoDataBuffer().addressPointer(), null);
                break;
            default:
                throw new UnsupportedOperationException("Unknown operation type: [" + op.getOpType() + "]");
        }

        if (loop.lastErrorCode() != 0)
            throw new RuntimeException(loop.lastErrorMessage());

        return op.z();
    }


    protected <T extends Aggregate> Pointer getPointer(Batch<T> batch) {
        if (batchPointers.get() == null)
            batchPointers.set(new HashMap<Integer, Pointer>());

        if (!batchPointers.get().containsKey(batch.opNum())) {
            val pointer = new IntPointer(batch.getSample().getRequiredBatchMemorySize() / 4 );
            batchPointers.get().put(batch.opNum(), pointer);
            return pointer;
        }

        return batchPointers.get().get(batch.opNum());
    }


    /**
     * This method executes previously built batch
     *
     * @param batch
     */
    @Override
    public <T extends Aggregate> void exec(Batch<T> batch) {
        //profilingHookIn(batch);

        IntPointer pointer = (IntPointer) getPointer(batch);

        int maxTypes = 5;

        int maxIntArrays = batch.getSample().maxIntArrays();

        int maxArraySize = batch.getSample().maxIntArraySize();


        int indexPos = maxTypes * Batch.getBatchLimit();
        int intArraysPos = indexPos + (batch.getSample().maxIndexArguments() * Batch.getBatchLimit());
        int realPos = (intArraysPos + (maxIntArrays * maxArraySize * Batch.getBatchLimit()))
                / (Nd4j.dataType() == DataType.DOUBLE ? 2 : 1);
        int argsPos = (realPos + ((batch.getSample().maxRealArguments() * Batch.getBatchLimit())))
                / (Nd4j.dataType() == DataType.DOUBLE ? 1 : 2);
        int shapesPos = argsPos + (batch.getSample().maxArguments() * Batch.getBatchLimit());
        DataType dataType = null;
        List<Pointer> buffers = new ArrayList<Pointer>();
        List<Pointer> shapes = new ArrayList<Pointer>();
        List<Pointer> deviceBuffers = new ArrayList<Pointer>();
        List<Pointer> deviceShapes = new ArrayList<Pointer>();
        for (int i = 0; i < batch.getNumAggregates(); i++) {
            T op = batch.getAggregates().get(i);

            if (i == 0)
                dataType = op.getArguments().get(0).dataType();

            // put num arguments
            int idx = i * maxTypes;
            pointer.put(idx, op.getArguments().size());
            pointer.put(idx + 1, op.getShapes().size());
            pointer.put(idx + 2, op.getIndexingArguments().size());
            pointer.put(idx + 3, op.getRealArguments().size());
            pointer.put(idx + 4, op.getIntArrayArguments().size());


            // putting indexing arguments
            for (int e = 0; e < op.getIndexingArguments().size(); e++) {
                idx = indexPos + i * batch.getSample().maxIndexArguments();
                pointer.put(idx + e, op.getIndexingArguments().get(e));
            }

            // putting intArray values
            int bsize = maxIntArrays * maxArraySize;
            for (int e = 0; e < op.getIntArrayArguments().size(); e++) {
                int step = (i * bsize) + (e * maxArraySize);
                if (op.getIntArrayArguments().get(e) != null)
                    for (int x = 0; x < op.getIntArrayArguments().get(e).length; x++) {
                        idx = intArraysPos + step + x;
                        pointer.put(idx, op.getIntArrayArguments().get(e)[x]);
                    }
            }

            // TODO: variable datatype should be handled here
            // putting real arguments

            switch (dataType){
                case FLOAT:
                    FloatPointer fPtr = new FloatPointer(pointer);
                    for (int e = 0; e < op.getRealArguments().size(); e++) {
                        idx = realPos + i * op.maxRealArguments();
                        fPtr.put(idx + e, op.getRealArguments().get(e).floatValue());
                    }
                    break;
                case DOUBLE:
                    DoublePointer dPtr = new DoublePointer(pointer);
                    for (int e = 0; e < op.getRealArguments().size(); e++) {
                        idx = realPos + (i * op.maxRealArguments());
                        dPtr.put(idx + e, op.getRealArguments().get(e).doubleValue());
                    }
                    break;
                default:
                    throw new ND4JIllegalArgumentException("Only FLOAT and DOUBLE datatypes are supported");
            }

            if (extraz.get() == null)
                extraz.set(new PointerPointer(32));

            // putting arguments pointers

            PointerPointer ptrPtr = new PointerPointer(pointer);//extraz.get().put(pointer);

            for (int e = 0; e < op.getArguments().size(); e++) {
                idx = argsPos + i * batch.getSample().maxArguments();

                if (op.getArguments().get(e) != null) {
                    Pointer p = op.getArguments().get(e).data().addressPointer();
                    long size = (p.limit() - p.position()) * p.sizeof();
                    Pointer p2 = loop.mallocDevice(size, -1, 0);
                    loop.memcpySync(p2, p, size, 0, null);
                    ptrPtr.put(idx + e, p2);

                    buffers.add(p);
                    deviceBuffers.add(p2);
                }
            }


            // putting shape pointers
            for (int e = 0; e < op.getShapes().size(); e++) {
                idx = shapesPos + i * batch.getSample().maxShapes();

                if (op.getShapes().get(e) != null) {
                    Pointer p = op.getShapes().get(e).addressPointer();
                    long size = (p.limit() - p.position()) * p.sizeof();
                    Pointer p2 = loop.mallocDevice(size, -1, 0);
                    loop.memcpySync(p2, p, size, 0, null);
                    ptrPtr.put(idx + e, p2);

                    shapes.add(p);
                    deviceShapes.add(p2);
                }
            }
        }

        loop.execAggregateBatch(null, batch.getNumAggregates(), batch.opNum(),
                    batch.getSample().maxArguments(), batch.getSample().maxShapes(),
                    batch.getSample().maxIntArrays(), batch.getSample().maxIntArraySize(),
                    batch.getSample().maxIndexArguments(), batch.getSample().maxRealArguments(), pointer, FlatBuffersMapper.getDataTypeAsByte(dataType));

        if (loop.lastErrorCode() != 0)
            throw new RuntimeException(loop.lastErrorMessage());

        for (int i = 0; i < buffers.size(); i++) {
            Pointer p = buffers.get(i);
            Pointer p2 = deviceBuffers.get(i);
            long size = (p.limit() - p.position()) * p.sizeof();
            loop.memcpySync(p, p2, size, 0, null);
            loop.freeDevice(p2, -1);
        }
        for (int i = 0; i < shapes.size(); i++) {
            Pointer p2 = deviceShapes.get(i);
            loop.freeDevice(p2, -1);
        }
    }

    /**
     * This method takes arbitrary
     * sized list of {@link Aggregate},
     * and packs them into batches
     * Note here that this is mainly used for random number generation
     * for {@link RandomOp} and things like {@link org.nd4j.linalg.api.rng.distribution.Distribution}
     * @param batch the list of {@link Aggregate} to
     *              execute upon
     */
    @Override
    public void exec(List<Aggregate> batch) {
        if (batch.size() == 0)
            return;

        List<Batch<Aggregate>> batches = Batch.getBatches(batch);
        for (Batch<Aggregate> single : batches) {
            this.exec(single);
        }
    }

    /**
     * This method takes arbitrary
     * sized list of {@link Aggregate},
     * and packs them into batches
     * Note here that this is mainly used for random number generation
     * for {@link RandomOp} and things like {@link org.nd4j.linalg.api.rng.distribution.Distribution}
     * @param op the list of {@link Aggregate} to
     *              execute upon
     */
    @Override
    public void exec(Aggregate op) {
        // long st = profilingHookIn(op);

        if (memoryBlocks.get() == null)
            memoryBlocks.set(new HashMap<Integer, AggregateMemoryBlock>());

        if (memoryBlocks.get().get(op.opNum()) == null)
            memoryBlocks.get().put(op.opNum(), new AggregateMemoryBlock(op));

        AggregateMemoryBlock block = memoryBlocks.get().get(op.opNum());

        int numArguments = op.getArguments().size();
        int numIndexArguments = op.getIndexingArguments().size();
        int numRealArguments = op.getRealArguments().size();
        int numShapes = op.getShapes().size();
        int numIntArrays = op.getIntArrayArguments().size();

        Pointer[] arguments = new Pointer[numArguments];
        Pointer[] intArrays = new Pointer[numIntArrays];
        val dataType = op.getArguments().get(0).dataType();

        for (int x = 0; x < numArguments; x++) {
            arguments[x] = op.getArguments().get(x) == null ? null
                    : op.getArguments().get(x).data().addressPointer();
        }

        Pointer[] shapes = new Pointer[numShapes];

        for (int x = 0; x < numShapes; x++) {
            if (op.getShapes().get(x).dataType() != DataType.LONG)
                throw new RuntimeException("ShapeBuffers should have LONG data opType");

            shapes[x] = op.getShapes().get(x) == null ? null : op.getShapes().get(x).addressPointer();
        }

        //int[] indexes = new int[numIndexArguments];
        IntPointer pointer = block.getIndexingPointer();
        for (int x = 0; x < numIndexArguments; x++) {
            pointer.put(x, op.getIndexingArguments().get(x));
        }

        //IntPointer pointer = new IntPointer(indexes);

        double[] reals = new double[numRealArguments];
        for (int x = 0; x < numRealArguments; x++) {
            //reals[x] = op.getRealArguments().get(x).doubleValue();
            switch (dataType) {
                case FLOAT:
                    ((FloatPointer) block.getRealArgumentsPointer()).put(x, op.getRealArguments().get(x).floatValue());
                    break;
                case DOUBLE:
                    ((DoublePointer) block.getRealArgumentsPointer()).put(x, op.getRealArguments().get(x).doubleValue());
                    break;
                default:
                    throw new ND4JIllegalArgumentException("Only FLOAT and DOUBLE datatypes are supported");
            }
        }

        for (int x = 0; x < numIntArrays; x++) {
            IntPointer intPtr = block.getIntArrays().get(x); //new IntPointer(op.getIntArrayArguments().get(x));
            intPtr.put(op.getIntArrayArguments().get(x), 0, op.getIntArrayArguments().get(x).length);
            intArrays[x] = intPtr;
        }

        //INDArray realsBuffer = Nd4j.create(reals);



        loop.execAggregate(null, op.opNum(), block.getArgumentsPointer().put(arguments), numArguments, block.getShapesPointer().put(shapes), numShapes, pointer,
                    numIndexArguments, block.getArraysPointer().put(intArrays), numIntArrays, block.getRealArgumentsPointer(),
                    numRealArguments, FlatBuffersMapper.getDataTypeAsByte(dataType));

        if (loop.lastErrorCode() != 0)
            throw new RuntimeException(loop.lastErrorMessage());
    }

    /**
     * This method return set of key/value and
     * key/key/value objects,
     * describing current environment
     *
     * @return
     */
    @Override
    public Properties getEnvironmentInformation() {
        Properties properties = super.getEnvironmentInformation();
        properties.put(Nd4jEnvironment.BACKEND_KEY, "CPU");
        properties.put(Nd4jEnvironment.OMP_THREADS_KEY, loop.ompGetMaxThreads());
        properties.put(Nd4jEnvironment.BLAS_THREADS_KEY, Nd4j.factory().blas().getMaxThreads());
        properties.put(Nd4jEnvironment.BLAS_VENDOR_KEY, (Nd4j.factory().blas()).getBlasVendor().toString());
        properties.put(Nd4jEnvironment.HOST_FREE_MEMORY_KEY, Pointer.maxBytes() - Pointer.totalBytes());

        // fill bandwidth information
        /*
        Note: Environment information is logged as part of ND4J initialization... but PerformanceTracker required
        ND4J init to be completed before it can be initialized. Hence we can get a null PerformanceTracker when
        OpExecutioner.printEnvironmentInformation() is called as part of ND4J class initialization - even
        though PerformanceTracker.getInstance() refers to a static final field (as it may not yet be initialized)
         */
        if(PerformanceTracker.getInstance() != null) {
            properties.put(Nd4jEnvironment.MEMORY_BANDWIDTH_KEY, PerformanceTracker.getInstance().getCurrentBandwidth());
        }

        return properties;
    }

    /**
     * This method executes specified RandomOp using default RNG available via Nd4j.getRandom()
     *
     * @param op
     */
    @Override
    public INDArray exec(RandomOp op) {
        return exec(op, Nd4j.getRandom());
    }

    /**
     * This method executes specific
     * RandomOp against specified RNG
     *
     * @param op
     * @param rng
     */
    @Override
    public INDArray exec(RandomOp op, Random rng) {
        return exec(op, null, rng);
    }


    public INDArray exec(RandomOp op, OpContext oc, Random rng) {
        INDArray x = getX(op, oc);
        INDArray y = getY(op, oc);
        INDArray z = getZ(op, oc);

        if(op instanceof BaseRandomOp && ((BaseRandomOp)op).isTripleArgRngOp() && z != null && x == null && y == null){
            //Ugly hack to ensure the triple arg call occurs
            //See GaussianDistribution.setZ etc
            x = z;
            y = z;
        }

        if (!(rng instanceof CpuNativeRandom))
            throw new IllegalStateException(
                    "You should use one of NativeRandom classes for NativeOperations execution. Op class: " + op.getClass().getName());

        long st = profilingConfigurableHookIn(op);

        //validateDataType(Nd4j.dataType(), op);

        Preconditions.checkArgument(op.z().isR(), "Op.Z must have one of floating point types");

        val xb = x == null ? null : ((BaseAuroraDataBuffer) x.data()).getOpaqueDataBuffer();
        val yb = y == null ? null : ((BaseAuroraDataBuffer) y.data()).getOpaqueDataBuffer();
        val zb = z == null ? null : ((BaseAuroraDataBuffer) z.data()).getOpaqueDataBuffer();

        if (x != null && y != null && z != null) {
            // triple arg call
            loop.execRandom3(null, op.opNum(), rng.getStatePointer(), // rng state ptr
                    xb, (LongPointer) x.shapeInfoDataBuffer().addressPointer(), null,
                    yb, (LongPointer) y.shapeInfoDataBuffer().addressPointer(), null,
                    zb, (LongPointer) z.shapeInfoDataBuffer().addressPointer(), null,
                    op.extraArgsDataBuff(z.dataType()).addressPointer());
        } else if (x != null && z != null) {
            //double arg call
            loop.execRandom2(null, op.opNum(), rng.getStatePointer(), // rng state ptr
                    xb, (LongPointer) x.shapeInfoDataBuffer().addressPointer(), null,
                    zb, (LongPointer) z.shapeInfoDataBuffer().addressPointer(), null,
                    op.extraArgsDataBuff(z.dataType()).addressPointer());
        } else {
            // single arg call
            loop.execRandom(null, op.opNum(), rng.getStatePointer(), // rng state ptr
                    zb, (LongPointer) z.shapeInfoDataBuffer().addressPointer(), null,
                    op.extraArgsDataBuff(z.dataType()).addressPointer());
        }

        if (loop.lastErrorCode() != 0)
            throw new RuntimeException(loop.lastErrorMessage());

        profilingConfigurableHookOut(op, st);

        return op.z();
    }

    @Override
    public TADManager getTADManager() {
        return tadManager;
    }

    /**
     * This class holds memory chunks required for single specific Aggregate op.
     * Can be used together with ThreadLocal variables
     */
    @Data
    private static class AggregateMemoryBlock {
        private List<IntPointer> intArrays = new ArrayList<>();
        private IntPointer indexingPointer;
        private Pointer realArgumentsPointer;
        private PointerPointer shapesPointer;
        private PointerPointer argumentsPointer;
        private PointerPointer arraysPointer;

        private final int opNum;

        private AggregateMemoryBlock(@NonNull Aggregate op) {

            opNum = op.opNum();

            // creating IntArrays
            for (int i = 0; i < op.maxIntArrays(); i++) {
                intArrays.add(new IntPointer(op.maxIntArraySize()));
            }

            // allocating chunk for IndexingArguments
            indexingPointer = new IntPointer(op.maxIndexArguments());

            // allocating chunk for RealArguments
            realArgumentsPointer = Nd4j.dataType() == DataType.DOUBLE ? new DoublePointer(op.maxRealArguments())
                    : new FloatPointer(op.maxRealArguments());

            // allocating chunk for shapesPointer
            shapesPointer = new PointerPointer(op.maxShapes());

            // allocating chunk for argumentsPointer
            argumentsPointer = new PointerPointer(op.maxArguments());

            // chunk for intArrays
            arraysPointer = new PointerPointer(op.maxIntArrays());
        }

        @Override
        public boolean equals(Object o) {
            if (this == o)
                return true;
            if (o == null || getClass() != o.getClass())
                return false;

            AggregateMemoryBlock that = (AggregateMemoryBlock) o;

            return opNum == that.opNum;
        }

        @Override
        public int hashCode() {
            return opNum;
        }
    }

    @Override
    public INDArray thresholdEncode(INDArray input, double threshold) {
        return thresholdEncode(input, threshold, null);
    }

    @Override
    public INDArray thresholdEncode(INDArray input, double threshold, Integer boundary) {

        //val condition = new MatchCondition(input, Conditions.absGreaterThanOrEqual(threshold));
        //long t1 = System.currentTimeMillis();
        int cntAbs = loop.estimateThreshold(null,
                input.data().addressPointer(),
                (LongPointer) input.shapeInfoDataBuffer().addressPointer(),
                (int) input.length(),
                (float) threshold);
        //long t2 = System.currentTimeMillis();

        if (loop.lastErrorCode() != 0)
            throw new RuntimeException(loop.lastErrorMessage());

        if (cntAbs < 2)
            return null;

        if (boundary != null)
            cntAbs = Math.min(cntAbs, boundary);

        //log.info("S: {}; T: {}", cntAbs, t2 - t1);

        DataBuffer buffer = input.data();

        long originalLength = buffer.length() * Nd4j.sizeOfDataType(buffer.dataType());
        int compressedLength = cntAbs + 4;
        // first 3 elements contain header

        DataBuffer encodedBuffer = Nd4j.getMemoryManager().getCurrentWorkspace() == null ? Nd4j.getDataBufferFactory().createInt(4+cntAbs, false) : Nd4j.getDataBufferFactory().createInt(4+cntAbs, false, Nd4j.getMemoryManager().getCurrentWorkspace());

        encodedBuffer.put(0, cntAbs);
        encodedBuffer.put(1, (int) buffer.length());
        encodedBuffer.put(2, Float.floatToIntBits((float) threshold));

        // format id
        encodedBuffer.put(3, ThresholdCompression.FLEXIBLE_ENCODING);

        CompressionDescriptor descriptor = new CompressionDescriptor();
        descriptor.setCompressedLength(compressedLength * 4); // sizeOf(INT)
        descriptor.setOriginalLength(originalLength);
        descriptor.setOriginalElementSize(Nd4j.sizeOfDataType(buffer.dataType()));
        descriptor.setNumberOfElements(buffer.length());

        descriptor.setCompressionAlgorithm("THRESHOLD");
        descriptor.setCompressionType(CompressionType.LOSSLESS);

        //CompressedDataBuffer cbuff = new CompressedDataBuffer(pointer, descriptor);

        Nd4j.getNDArrayFactory().convertDataEx(AbstractCompressor.getBufferTypeEx(buffer), buffer.addressPointer(), DataTypeEx.THRESHOLD, encodedBuffer.addressPointer(), buffer.length());

        Nd4j.getAffinityManager().tagLocation(buffer, AffinityManager.Location.HOST);

        return Nd4j.createArrayFromShapeBuffer(encodedBuffer, input.shapeInfoDataBuffer());
    }

    @Override
    public INDArray thresholdDecode(INDArray encoded, INDArray target) {
        DataBuffer buffer = encoded.data();

        if (buffer.dataType() != DataType.INT)
            throw new ND4JIllegalStateException("thresholdEncoded array should have dataType of INT");

        long compressedLength = buffer.getInt(0);
        long originalLength = buffer.getInt(1);
        float threshold = buffer.getInt(2);

        if (target.length() != originalLength)
            throw new ND4JIllegalStateException("originalLength ["+ originalLength+"] stored in encoded array doesn't match target length ["+ target.length()+"]");

        DataTypeEx typeDst = AbstractCompressor.getBufferTypeEx(target.data());

        loop.convertTypes(null, DataTypeEx.THRESHOLD.ordinal(), buffer.addressPointer(), target.length(), typeDst.ordinal(), target.data().addressPointer());

        if (loop.lastErrorCode() != 0)
            throw new RuntimeException(loop.lastErrorMessage());

        return target;
    }


    @Override
    public long bitmapEncode(INDArray indArray, INDArray target, double threshold) {
        long length = indArray.length();
        long tLen = target.data().length();

        if (tLen != (length / 16 + 5))
            throw new ND4JIllegalStateException("Length of target array should be " + (length / 16 + 5));

        if (target.data().dataType() != DataType.INT)
            throw new ND4JIllegalStateException("Target array should have INT dataType");

        DataBuffer buffer = target.data();

        buffer.put(0, (int) length);
        buffer.put(1, (int) length);
        buffer.put(2, Float.floatToIntBits((float) threshold));

        // format id
        buffer.put(3, ThresholdCompression.BITMAP_ENCODING);

        long affected = loop.encodeBitmap(null,
                indArray.data().addressPointer(),
                (LongPointer) indArray.shapeInfoDataBuffer().addressPointer(),
                length,
                (IntPointer) buffer.addressPointer(),
                (float) threshold);

        if (loop.lastErrorCode() != 0)
            throw new RuntimeException(loop.lastErrorMessage());

        return affected;
    }

    @Override
    public INDArray bitmapDecode(INDArray encoded, INDArray target) {

        loop.decodeBitmap(null,
                encoded.data().addressPointer(),
                target.length(),
                target.data().addressPointer(),
                (LongPointer) target.shapeInfoDataBuffer().addressPointer()
        );

        if (loop.lastErrorCode() != 0)
            throw new RuntimeException(loop.lastErrorMessage());

        return target;
    }


    @Override
    public synchronized Map<String, CustomOpDescriptor> getCustomOperations() {
        if (customOps == null) {
            String list = loop.getAllCustomOps();

            if (list == null || list.isEmpty()) {
                log.warn("No customs ops available!");
                customOps = Collections.emptyMap();
                return customOps;
            }

            val map = new HashMap<String, CustomOpDescriptor>();

            String[] split = list.split(";");
            for (String op : split) {
                if (op == null || op.isEmpty())
                    continue;

                String[] another = op.split(":");

                CustomOpDescriptor descriptor = CustomOpDescriptor.builder()
                        .hash(Long.valueOf(another[1]))
                        .numInputs(Integer.valueOf(another[2]))
                        .numOutputs(Integer.valueOf(another[3]))
                        .allowsInplace(Integer.valueOf(another[4]) == 1)
                        .numTArgs(Integer.valueOf(another[5]))
                        .numIArgs(Integer.valueOf(another[6]))
                        .build();

                map.put(another[0], descriptor);
            }

            customOps = Collections.unmodifiableMap(map);
        }

        return customOps;
    }


    private PointerPointer getPointerPointerFrom(ThreadLocal<Map<Integer,PointerPointer>> map,int numArguments) {
        if(map.get() == null) {
            Map<Integer,PointerPointer> store = new HashMap<>();
            store.put(numArguments,new PointerPointer(numArguments));
            map.set(store);
            return map.get().get(numArguments);
        }
        else if (map.get().get(numArguments) == null) {
            PointerPointer pointerPointer = new PointerPointer(numArguments);
            map.get().put(numArguments,pointerPointer);
            return pointerPointer;
        }

        return map.get().get(numArguments);
    }




    private ShortPointer getShortPointerFrom(ThreadLocal<Map<Integer,ShortPointer>> map,int numArguments) {
        if(map.get() == null) {
            Map<Integer,ShortPointer> store = new HashMap<>();
            store.put(numArguments,new ShortPointer(numArguments));
            map.set(store);
            return map.get().get(numArguments);
        }
        else if (map.get().get(numArguments) == null) {
            ShortPointer pointerPointer = new ShortPointer(numArguments);
            map.get().put(numArguments,pointerPointer);
            return pointerPointer;
        }

        return map.get().get(numArguments);
    }


    private LongPointer getLongPointerFrom(ThreadLocal<Map<Integer,LongPointer>> map,int numArguments) {
        if(map.get() == null) {
            Map<Integer,LongPointer> store = new HashMap<>();
            store.put(numArguments,new LongPointer(numArguments));
            map.set(store);
            return map.get().get(numArguments);
        }
        else if (map.get().get(numArguments) == null) {
            val pointerPointer = new LongPointer(numArguments);
            map.get().put(numArguments,pointerPointer);
            return pointerPointer;
        }

        return map.get().get(numArguments);
    }

    private DoublePointer getDoublePointerFrom(ThreadLocal<Map<Integer,DoublePointer>> map,int numArguments) {
        if(map.get() == null) {
            Map<Integer,DoublePointer> store = new HashMap<>();
            store.put(numArguments,new DoublePointer(numArguments));
            map.set(store);
            return map.get().get(numArguments);
        }
        else if (map.get().get(numArguments) == null) {
            DoublePointer pointerPointer = new DoublePointer(numArguments);
            map.get().put(numArguments,pointerPointer);
            return pointerPointer;
        }

        return map.get().get(numArguments);
    }


    private BooleanPointer getBooleanPointerFrom(ThreadLocal<Map<Integer,BooleanPointer>> map,int numArguments) {
        if(map.get() == null) {
            Map<Integer,BooleanPointer> store = new HashMap<>();
            store.put(numArguments,new BooleanPointer(numArguments));
            map.set(store);
            return map.get().get(numArguments);
        }
        else if (map.get().get(numArguments) == null) {
            val pointerPointer = new BooleanPointer(numArguments);
            map.get().put(numArguments,pointerPointer);
            return pointerPointer;
        }

        return map.get().get(numArguments);
    }


    private PointerPointer getInputShapes(int numArguments) {
       return getPointerPointerFrom(inputShapes,numArguments);
    }

    private PointerPointer getInputBuffers(int numArguments) {
        return getPointerPointerFrom(inputBuffers,numArguments);

    }

    private PointerPointer getOutputShapes(int numArguments) {
        return getPointerPointerFrom(outputShapes,numArguments);

    }

    private PointerPointer getOutputBuffers(int numArguments) {
        return getPointerPointerFrom(outputBuffers,numArguments);

    }

    /**
     * This method executes given CustomOp
     *
     * PLEASE NOTE: You're responsible for input/output validation
     * @param op
     */
    @Override
    public INDArray[] exec(@NonNull CustomOp op) {

        if (op.numOutputArguments() == 0 && !op.isInplaceCall()) {
            try {
                val list = this.calculateOutputShape(op);
                if (list.isEmpty())
                    throw new ND4JIllegalStateException("Op name " + op.opName() + " failed to execute. You can't execute non-inplace CustomOp without outputs being specified");

                for (LongShapeDescriptor shape : list)
                    op.addOutputArgument(Nd4j.create(shape, false));

            } catch (Exception e) {
                throw new ND4JIllegalStateException("Op name " + op.opName() + " failed to execute. You can't execute non-inplace CustomOp without outputs being specified");
            }
        }

        val name = op.opName();
        try (val context = buildContext()) {

            context.markInplace(op.isInplaceCall());

            // transferring rng state
            context.setRngStates(Nd4j.getRandom().rootState(), Nd4j.getRandom().nodeState());

            //transferring input/output arrays
            context.setInputArrays(op.inputArguments());
            context.setOutputArrays(op.outputArguments());

            // transferring static args
            context.setBArguments(op.bArgs());
            context.setIArguments(op.iArgs());
            context.setTArguments(op.tArgs());

            val result = exec(op, context);
            val states = context.getRngStates();

            // pulling states back
            Nd4j.getRandom().setStates(states.getFirst(), states.getSecond());

            return result;
        } catch (ND4JOpProfilerException e){
            throw e;
        } catch (Exception e) {
            throw new RuntimeException("Op [" + name + "] execution failed", e);
        }
    }

    protected LongShapeDescriptor getShapeFromPointer(LongPointer ptr) {
        val rank = (int) ptr.get(0);

        val shape = new long[rank * 2 + 4];
        for (int i = 0; i < shape.length; i++) {
            shape[i] = ptr.get(i);
        }

        //val extras = ptr.get(Shape.shapeInfoLength(rank) - 3);
        val t = ArrayOptionsHelper.arrayType(shape);
        return LongShapeDescriptor.fromShape(Shape.shape(shape), Shape.stride(shape), Shape.elementWiseStride(shape), Shape.order(shape), ArrayOptionsHelper.dataType(shape), t == ArrayType.EMPTY);
    }

    @Override
    public List<LongShapeDescriptor> calculateOutputShape(@NonNull CustomOp op) {
        return calculateOutputShape(op, null);
    }

    @Override
    public List<LongShapeDescriptor> calculateOutputShape(@NonNull CustomOp op, OpContext opContext) {
        val lc = op.opName().toLowerCase();
        val hash = op.opHash();

        val result = new ArrayList<LongShapeDescriptor>();
        if(op.numInputArguments() < 1 && op.getDescriptor().getNumInputs() != -2) {
            if(log.isTraceEnabled()){
                log.trace("Could not calculate output shape for op {}: number of input args was 0",
                        op.getClass().getName());
            }
            return Collections.emptyList();
        }


        int nIn = opContext != null ? opContext.numInputArguments() : op.numInputArguments();
        val inputBuffers = new PointerPointer<>(nIn);
        val inputShapes = new PointerPointer<>(nIn);
        val inputArgs = opContext != null ? opContext.getInputArrays() : op.inputArguments();
        int cnt= 0;
        for (val in: inputArgs) {
            if (!in.isEmpty())
                inputBuffers.put(cnt, in.data().addressPointer());

            inputShapes.put(cnt++, in.shapeInfoDataBuffer().addressPointer());
        }


        int nIArgs = opContext != null ? opContext.numIArguments() : op.numIArguments();
        val iArgs = nIArgs > 0 ? new LongPointer(nIArgs) : null;
        cnt = 0;
        if(opContext != null){
            for (val i: opContext.getIArguments())
                iArgs.put(cnt++, i);
        } else {
            for (val i: op.iArgs())
                iArgs.put(cnt++, i);
        }


        int nTArgs = opContext != null ? opContext.numTArguments() : op.numTArguments();
        val tArgs = nTArgs > 0 ? new DoublePointer(nTArgs) : null;

        int nBArgs = opContext != null ? opContext.numBArguments() : op.numBArguments();
        val bArgs = nBArgs > 0 ? new BooleanPointer(nBArgs) : null;

        int nDArgs = opContext != null ? opContext.numDArguments() : op.numDArguments();
        val dArgs = nDArgs > 0 ? new IntPointer(nDArgs) : null;

        cnt = 0;
        if(opContext != null){
            for (val b: opContext.getBArguments())
                bArgs.put(cnt++, b);
        } else {
            for (val b: op.bArgs())
                bArgs.put(cnt++, b);
        }


        cnt = 0;
        if(opContext != null){
            for (val b: opContext.getTArguments())
                tArgs.put(cnt++, b);
        } else {
            for (val b: op.tArgs())
                tArgs.put(cnt++, b);
        }

        cnt = 0;
        if(opContext != null){
            for (val b: opContext.getDArguments())
                dArgs.put(cnt++, b.toInt());
        } else {
            for (val b: op.dArgs())
                dArgs.put(cnt++, b.toInt());
        }

        OpaqueShapeList ptrptr;
        try {
            ptrptr = loop.calculateOutputShapes2(null, hash, inputBuffers, inputShapes, nIn, tArgs, nTArgs, iArgs, nIArgs, bArgs, nBArgs, dArgs, nDArgs);

            if (loop.lastErrorCode() != 0)
                throw new RuntimeException(loop.lastErrorMessage());
        } catch (Throwable t){
            StringBuilder sb = new StringBuilder();
            sb.append("Inputs: [(");
            for( int i=0; i<inputArgs.size(); i++ ){
                if(i > 0)
                    sb.append("), (");
                sb.append(Shape.shapeToStringShort(inputArgs.get(i)));
            }
            sb.append(")]");
            if(op instanceof DifferentialFunction && ((DifferentialFunction)op).getSameDiff() != null){
                appendSameDiffInfo(sb, (DifferentialFunction) op);
            }

            log.error("Failed to calculate output shapes for op " + op.opName() + ". Attempted to execute with " +
                    String.valueOf(op.numInputArguments()) + " inputs, " +
                    String.valueOf(op.numOutputArguments()) + " outputs, "+
                    String.valueOf(op.numTArguments()) + " targs and " +
                    String.valueOf(op.numIArguments()) + " iargs. " +
                    sb.toString() +
                    " - Please see above message (printed out from c++) for a possible cause of error.");
            throw t;
        }

        if (loop.lastErrorCode() != 0)
            throw new RuntimeException(loop.lastErrorMessage());

            if (ptrptr == null)
                throw new RuntimeException();

            for (int e = 0; e < loop.getShapeListSize(ptrptr); e++ ) {
                Pointer deviceBuffer = loop.getShape(ptrptr, e);

                IntPointer rank = new IntPointer(1);
                loop.memcpySync(rank, deviceBuffer, 4, 0, null);

                LongPointer hostPointer = new LongPointer(rank.get(0) * 2 + 4);
                loop.memcpySync(hostPointer, deviceBuffer, hostPointer.capacity() * 8, 0, null);
                result.add(getShapeFromPointer(hostPointer));
            }


            loop.deleteShapeList(ptrptr);

        if(log.isTraceEnabled()){
            String[] arr = new String[result.size()];
            for( int i=0; i<result.size(); i++ ){
                arr[i] = result.get(i).toString();
            }
            log.trace("Calculated output shapes for op {} - {}", op.getClass().getName(), Arrays.toString(arr));
        }
        return result;
    }


    @Override
    public void enableDebugMode(boolean reallyEnable) {
        debug.set(reallyEnable);
        loop.enableDebugMode(reallyEnable);
    }

    @Override
    public void enableVerboseMode(boolean reallyEnable) {
        verbose.set(reallyEnable);
        loop.enableVerboseMode(reallyEnable);
    }


    @Override
    public void registerGraph(long id, Pointer graph) {
         loop.registerGraph(null, id, graph);

        if (loop.lastErrorCode() != 0)
            throw new RuntimeException(loop.lastErrorMessage());
    }

    @Override
    public Map<String, INDArray> executeGraph(long id, @NonNull Map<String, INDArray> map, @NonNull Map<String, Integer> reverseMap) {

        val ptrBuffers = new Pointer[map.size()];
        val ptrShapes = new Pointer[map.size()];
        val ptrIndices = new IntPointer(map.size());

        int cnt = 0;
        val keySet = new ArrayList<String>(map.keySet());
        for (val key: keySet) {
            val array = map.get(key);

            ptrBuffers[cnt] = array.data().addressPointer();
            ptrShapes[cnt] = array.shapeInfoDataBuffer().addressPointer();
            ptrIndices.put(cnt, reverseMap.get(key));

            cnt++;
        }

        val newMap = new LinkedHashMap<String, INDArray>();

        OpaqueVariablesSet result = loop.executeStoredGraph(null, id, new PointerPointer(ptrBuffers), new PointerPointer(ptrShapes), ptrIndices, map.size());

        if (loop.lastErrorCode() != 0)
            throw new RuntimeException(loop.lastErrorMessage());

            OpStatus status = OpStatus.byNumber(loop.getVariablesSetStatus(result));

            if (status != OpStatus.ND4J_STATUS_OK)
                throw new ND4JIllegalStateException("Op execution failed: " + status);

            for (int e = 0; e < loop.getVariablesSetSize(result); e++) {
                OpaqueVariable var = loop.getVariable(result, e);
                int nodeId = loop.getVariableId(var);
                int index = loop.getVariableIndex(var);
                LongPointer deviceShapeInfo = loop.getVariableShape(var);
                Pointer deviceBuffer = loop.getVariableBuffer(var);

                IntPointer rank = new IntPointer(1);
                loop.memcpySync(rank, deviceShapeInfo, 4, 0, null);

                LongPointer hostShapeInfo = new LongPointer(rank.get(0) * 2 + 4);
                loop.memcpySync(hostShapeInfo, deviceShapeInfo, hostShapeInfo.capacity() * 8, 0, null);

                val jshape = new long[rank.get(0) * 2 + 4];
                for (int i = 0; i < jshape.length; i++) {
                    jshape[i] = hostShapeInfo.get(i);
                }

                val shapeOf = Shape.shapeOf(jshape);
                val stridesOf = Shape.stridesOf(jshape);
                val order = Shape.order(jshape);
                val array = Nd4j.create(shapeOf, stridesOf, 0, order);

                val perfX = PerformanceTracker.getInstance().helperStartTransaction();

                loop.memcpySync(array.data().addressPointer(), deviceBuffer, Shape.lengthOf(shapeOf) * Nd4j.sizeOfDataType(array.dataType()), 0, null);

                PerformanceTracker.getInstance().helperRegisterTransaction(0, perfX, Shape.lengthOf(shapeOf) * Nd4j.sizeOfDataType(array.dataType()), MemcpyDirection.DEVICE_TO_HOST);

                //newMap.put(keySet.get(nodeId), array);
                String nodeName = loop.getVariableName(var);
                newMap.put(nodeName, array);
            }

        loop.deleteVariablesSet(result);

        return newMap;
    }

    @Override
    public void forgetGraph(long id) {
        loop.unregisterGraph(null, id);
        if (loop.lastErrorCode() != 0)
            throw new RuntimeException(loop.lastErrorMessage());
    }

    /**
     * This method allows to set desired number of elements per thread, for performance optimization purposes.
     * I.e. if array contains 2048 elements, and threshold is set to 1024, 2 threads will be used for given op execution.
     * <p>
     * Default value: 1024
     *
     * @param threshold
     */
    @Override
    public void setElementsThreshold(int threshold) {
        loop.setElementThreshold(threshold);
    }

    /**
     * This method allows to set desired number of sub-arrays per thread, for performance optimization purposes.
     * I.e. if matrix has shape of 64 x 128, and threshold is set to 8, each thread will be processing 8 sub-arrays (sure, if you have 8 core cpu).
     * If your cpu has, say, 4, cores, only 4 threads will be spawned, and each will process 16 sub-arrays
     * <p>
     * Default value: 8
     *
     * @param threshold
     */
    @Override
    public void setTadThreshold(int threshold) {
        loop.setTADThreshold(threshold);
    }

    @Override
    public String getString(DataBuffer buffer, long index) {
        Preconditions.checkArgument(buffer instanceof Utf8Buffer, "Expected Utf8Buffer");
/*
        val addr = ((LongIndexer) buffer.indexer()).get(index);
        val ptr = new PagedPointer(addr);
        val str = new Nd4jAurora.utf8string(ptr);
        return str._buffer().capacity(str._length()).getString();
        */
        return null;
    }

    @Override
    public ExecutionerType type() {
        return ExecutionerType.NATIVE_CPU;
    }

    @Override
    public boolean isExperimentalMode() {
        return experimentalMode.get();
    }

    @Override
    public void scatterUpdate(ScatterUpdate.UpdateOp op, @NonNull INDArray array, @NonNull INDArray indices, @NonNull INDArray updates, @NonNull int[] axis) {
        val tadX = tadManager.getTADOnlyShapeInfo(array, axis);
        val tadY = tadManager.getTADOnlyShapeInfo(updates, axis);

        if (tadY.getSecond().length() != indices.length())
            throw new IllegalStateException("Number of updates doesn't match number of indices. Bad dimensions used?");

        loop.scatterUpdate(null, op.ordinal(), (int) indices.length(),
                array.data().addressPointer(), (LongPointer) tadX.getFirst().addressPointer(), (LongPointer) tadX.getSecond().addressPointer(), null, null, null,
                updates.data().addressPointer(), (LongPointer) tadY.getFirst().addressPointer(), (LongPointer) tadY.getSecond().addressPointer(), null, null, null,
                indices.data().addressPointer(), (LongPointer) indices.shapeInfoDataBuffer().addressPointer(), null, null);

        if (loop.lastErrorCode() != 0)
            throw new RuntimeException(loop.lastErrorMessage());
    }

    @Override
    public OpContext buildContext() {
        return new CpuOpContext();
    }

    @Override
    public INDArray[] exec(CustomOp op, @NonNull OpContext context) {
        long st = profilingConfigurableHookIn(op);
        boolean mklOverride = false;
        try {
/*
            if (Nd4jCpu.Environment.getInstance().isUseMKLDNN()) {
                val opName = op.opName();
                val state = mklOverrides.get(op);
                if (state != null && state == true) {
                    mklOverride = true;
                    Nd4jCpu.Environment.getInstance().setUseMKLDNN(true);
                }
            }
*/
            val status = loop.execCustomOp2(null, op.opHash(), context.contextPointer());

            if (loop.lastErrorCode() != 0)
                throw new RuntimeException(loop.lastErrorMessage());

            if (status != 0)
                throw new RuntimeException("Op [" + op.opName() + "] execution failed");

            if (context.getOutputArrays().isEmpty())
                return new INDArray[0];
            else
                return context.getOutputArrays().toArray(new INDArray[context.getOutputArrays().size()]);
        } catch (Exception e) {
            val sb = new StringBuilder();
            sb.append("Inputs: [(");
            int nIn = (context.getInputArrays() == null ? 0 : context.getInputArrays().size());
            for (int i = 0; i < nIn; i++) {
                if (i > 0)
                    sb.append("), (");
                sb.append(Shape.shapeToStringShort(context.getInputArrays().get(i)));
            }
            sb.append(")]. Outputs: [(");
            int nOut = (context.getOutputArrays() == null ? 0 : context.getOutputArrays().size());
            for (int i = 0; i < nOut; i++) {
                if (i > 0)
                    sb.append("), (");
                sb.append(Shape.shapeToStringShort(context.getOutputArrays().get(i)));
            }
            sb.append(")]. tArgs: ");
            int nT = (context.getTArguments() == null ? 0 : context.getTArguments().size());
            if (nT > 0) {
                sb.append(context.getTArguments());
            } else {
                sb.append("-");
            }
            sb.append(". iArgs: ");
            int nI = (context.getIArguments() == null ? 0 : context.getIArguments().size());
            if (nI > 0) {
                sb.append(context.getIArguments());
            } else {
                sb.append("-");
            }
            sb.append(". bArgs: ");
            int nB = (context.getBArguments() == null ? 0 : context.getBArguments().size());
            if (nB > 0) {
                sb.append(context.getBArguments());
            } else {
                sb.append("-");
            }
            if (op instanceof DifferentialFunction) {
                String n = ((DifferentialFunction) op).getOwnName();
                if (n != null && !n.equals(op.opName())) {
                    sb.append(". Op own name: \"").append(n).append("\"");
                }
            }

            if(op instanceof DifferentialFunction && ((DifferentialFunction)op).getSameDiff() != null){
                appendSameDiffInfo(sb, (DifferentialFunction) op);
            }

            log.error("Failed to execute op " + op.opName() + ". Attempted to execute with " +
                    nIn + " inputs, " +
                    nOut + " outputs, " +
                    nT + " targs," +
                    nB + " bargs and " +
                    nI + " iargs. " +
                    sb.toString() +
                    " - Please see above message (printed out from c++) for a possible cause of error.");
            throw e;
        } finally {
/*
            if (mklOverride)
                Nd4jCpu.Environment.getInstance().setUseMKLDNN(true);
*/
            profilingConfigurableHookOut(op, st);
        }
    }

    @Override
    public INDArrayStatistics inspectArray(INDArray array) {
        throw new UnsupportedOperationException("Not supported yet.");
/*
        val debugInfo = new Nd4jCpu.DebugInfo();

        loop.inspectArray(null, array.data().addressPointer(), (LongPointer) array.shapeInfoDataBuffer().addressPointer(), null, null, debugInfo);

        if (loop.lastErrorCode() != 0)
            throw new RuntimeException(loop.lastErrorMessage());

        return INDArrayStatistics.builder()
                .minValue(debugInfo._minValue())
                .maxValue(debugInfo._maxValue())
                .meanValue(debugInfo._meanValue())
                .stdDevValue(debugInfo._stdDevValue())
                .countInf(debugInfo._infCount())
                .countNaN(debugInfo._nanCount())
                .countNegative(debugInfo._negativeCount())
                .countPositive(debugInfo._positiveCount())
                .countZero(debugInfo._zeroCount())
                .build();
*/
    }

    @Override
    public DataBuffer createShapeInfo(long[] shape, long[] stride, long elementWiseStride, char order, DataType dtype, boolean empty) {
        OpaqueConstantDataBuffer dbf = loop.shapeBuffer(shape.length, new LongPointer(shape), new LongPointer(stride), dtype.toInt(), order, elementWiseStride, empty);
        if (loop.lastErrorCode() != 0)
            throw new RuntimeException(loop.lastErrorMessage());

        Pointer hostPointer = new LongPointer(Shape.shapeInfoLength(shape.length));
        Pointer deviceBuffer = loop.getConstantDataBufferPrimary(dbf);
        loop.memcpySync(hostPointer, deviceBuffer, hostPointer.capacity() * 8, 0, null);
        val result = new LongBuffer(hostPointer, hostPointer.capacity());

        loop.deleteShapeBuffer(dbf);

        return result;
    }

    @Override
    public TadPack tadShapeInfoAndOffsets(INDArray array, int[] dimension) {
        OpaqueTadPack pack = loop.tadOnlyShapeInfo((LongPointer) array.shapeInfoDataBuffer().addressPointer(), new IntPointer(dimension), dimension.length);

        if (loop.lastErrorCode() != 0)
            throw new RuntimeException(loop.lastErrorMessage());

        Pointer shapeHostPointer = new LongPointer(loop.getShapeInfoLength(pack));
        Pointer shapeDeviceBuffer = loop.getPrimaryShapeInfo(pack);
        loop.memcpySync(shapeHostPointer, shapeDeviceBuffer, shapeHostPointer.capacity() * 8, 0, null);
        val tadShape = new LongBuffer(shapeHostPointer, shapeHostPointer.capacity());

        Pointer offsetsHostPointer = new LongPointer(loop.getNumberOfTads(pack));
        Pointer offsetsDeviceBuffer = loop.getPrimaryOffsets(pack);
        loop.memcpySync(offsetsHostPointer, offsetsDeviceBuffer, offsetsHostPointer.capacity() * 8, 0, null);
        val tadOffsets = new LongBuffer(offsetsHostPointer, offsetsHostPointer.capacity());

        loop.deleteTadPack(pack);

        return new TadPack(tadShape, tadOffsets);
    }

    protected void appendSameDiffInfo(StringBuilder sb, DifferentialFunction df){
        String[] inNames = df.argNames();
        String[] outNames = df.outputVariablesNames();
        if(inNames != null){
            sb.append(". Input var names: ").append(Arrays.toString(inNames));
        }
        if(outNames != null){
            sb.append(". Output var names: ").append(Arrays.toString(outNames));
        }
    }


    @Override
    public String runLightBenchmarkSuit(boolean printOut) {
        val s = loop.runLightBenchmarkSuit(printOut);
        if (loop.lastErrorCode() != 0)
            throw new RuntimeException(loop.lastErrorMessage());

        return s;
    }

    @Override
    public String runFullBenchmarkSuit(boolean printOut) {
        val s = loop.runFullBenchmarkSuit(printOut);
        if (loop.lastErrorCode() != 0)
            throw new RuntimeException(loop.lastErrorMessage());

        return s;
    }
}
