// <auto-generated>
//  automatically generated by the FlatBuffers compiler, do not modify
// </auto-generated>

namespace sd.graph
{

using global::System;
using global::FlatBuffers;

public struct FlatArray : IFlatbufferObject
{
  private Table __p;
  public ByteBuffer ByteBuffer { get { return __p.bb; } }
  public static FlatArray GetRootAsFlatArray(ByteBuffer _bb) { return GetRootAsFlatArray(_bb, new FlatArray()); }
  public static FlatArray GetRootAsFlatArray(ByteBuffer _bb, FlatArray obj) { return (obj.__assign(_bb.GetInt(_bb.Position) + _bb.Position, _bb)); }
  public void __init(int _i, ByteBuffer _bb) { __p.bb_pos = _i; __p.bb = _bb; }
  public FlatArray __assign(int _i, ByteBuffer _bb) { __init(_i, _bb); return this; }

  public long Shape(int j) { int o = __p.__offset(4); return o != 0 ? __p.bb.GetLong(__p.__vector(o) + j * 8) : (long)0; }
  public int ShapeLength { get { int o = __p.__offset(4); return o != 0 ? __p.__vector_len(o) : 0; } }
#if ENABLE_SPAN_T
  public Span<byte> GetShapeBytes() { return __p.__vector_as_span(4); }
#else
  public ArraySegment<byte>? GetShapeBytes() { return __p.__vector_as_arraysegment(4); }
#endif
  public long[] GetShapeArray() { return __p.__vector_as_array<long>(4); }
  public sbyte Buffer(int j) { int o = __p.__offset(6); return o != 0 ? __p.bb.GetSbyte(__p.__vector(o) + j * 1) : (sbyte)0; }
  public int BufferLength { get { int o = __p.__offset(6); return o != 0 ? __p.__vector_len(o) : 0; } }
#if ENABLE_SPAN_T
  public Span<byte> GetBufferBytes() { return __p.__vector_as_span(6); }
#else
  public ArraySegment<byte>? GetBufferBytes() { return __p.__vector_as_arraysegment(6); }
#endif
  public sbyte[] GetBufferArray() { return __p.__vector_as_array<sbyte>(6); }
  public DType Dtype { get { int o = __p.__offset(8); return o != 0 ? (DType)__p.bb.GetSbyte(o + __p.bb_pos) : DType.INHERIT; } }
  public ByteOrder ByteOrder { get { int o = __p.__offset(10); return o != 0 ? (ByteOrder)__p.bb.GetSbyte(o + __p.bb_pos) : ByteOrder.LE; } }

  public static Offset<FlatArray> CreateFlatArray(FlatBufferBuilder builder,
      VectorOffset shapeOffset = default(VectorOffset),
      VectorOffset bufferOffset = default(VectorOffset),
      DType dtype = DType.INHERIT,
      ByteOrder byteOrder = ByteOrder.LE) {
    builder.StartObject(4);
    FlatArray.AddBuffer(builder, bufferOffset);
    FlatArray.AddShape(builder, shapeOffset);
    FlatArray.AddByteOrder(builder, byteOrder);
    FlatArray.AddDtype(builder, dtype);
    return FlatArray.EndFlatArray(builder);
  }

  public static void StartFlatArray(FlatBufferBuilder builder) { builder.StartObject(4); }
  public static void AddShape(FlatBufferBuilder builder, VectorOffset shapeOffset) { builder.AddOffset(0, shapeOffset.Value, 0); }
  public static VectorOffset CreateShapeVector(FlatBufferBuilder builder, long[] data) { builder.StartVector(8, data.Length, 8); for (int i = data.Length - 1; i >= 0; i--) builder.AddLong(data[i]); return builder.EndVector(); }
  public static VectorOffset CreateShapeVectorBlock(FlatBufferBuilder builder, long[] data) { builder.StartVector(8, data.Length, 8); builder.Add(data); return builder.EndVector(); }
  public static void StartShapeVector(FlatBufferBuilder builder, int numElems) { builder.StartVector(8, numElems, 8); }
  public static void AddBuffer(FlatBufferBuilder builder, VectorOffset bufferOffset) { builder.AddOffset(1, bufferOffset.Value, 0); }
  public static VectorOffset CreateBufferVector(FlatBufferBuilder builder, sbyte[] data) { builder.StartVector(1, data.Length, 1); for (int i = data.Length - 1; i >= 0; i--) builder.AddSbyte(data[i]); return builder.EndVector(); }
  public static VectorOffset CreateBufferVectorBlock(FlatBufferBuilder builder, sbyte[] data) { builder.StartVector(1, data.Length, 1); builder.Add(data); return builder.EndVector(); }
  public static void StartBufferVector(FlatBufferBuilder builder, int numElems) { builder.StartVector(1, numElems, 1); }
  public static void AddDtype(FlatBufferBuilder builder, DType dtype) { builder.AddSbyte(2, (sbyte)dtype, 0); }
  public static void AddByteOrder(FlatBufferBuilder builder, ByteOrder byteOrder) { builder.AddSbyte(3, (sbyte)byteOrder, 0); }
  public static Offset<FlatArray> EndFlatArray(FlatBufferBuilder builder) {
    int o = builder.EndObject();
    return new Offset<FlatArray>(o);
  }
  public static void FinishFlatArrayBuffer(FlatBufferBuilder builder, Offset<FlatArray> offset) { builder.Finish(offset.Value); }
  public static void FinishSizePrefixedFlatArrayBuffer(FlatBufferBuilder builder, Offset<FlatArray> offset) { builder.FinishSizePrefixed(offset.Value); }
};


}
