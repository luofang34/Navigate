//! Minimal pure-Rust ONNX protobuf message definitions for graph preparation.
//!
//! Field tags match the upstream ONNX `onnx.proto3` wire specification so
//! `navigate-model` compiles without external git dependencies or `protoc`.

/// Top-level ONNX model container (`onnx.ModelProto`).
#[derive(Clone, PartialEq, prost::Message)]
pub struct ModelProto {
    /// ONNX intermediate representation version.
    #[prost(int64, tag = "1")]
    pub ir_version: i64,
    /// Producer framework name.
    #[prost(string, tag = "2")]
    pub producer_name: String,
    /// Producer framework version.
    #[prost(string, tag = "3")]
    pub producer_version: String,
    /// Model namespace domain.
    #[prost(string, tag = "4")]
    pub domain: String,
    /// Model version number.
    #[prost(int64, tag = "5")]
    pub model_version: i64,
    /// Human-readable model documentation string.
    #[prost(string, tag = "6")]
    pub doc_string: String,
    /// Primary computation graph.
    #[prost(message, optional, tag = "7")]
    pub graph: Option<GraphProto>,
    /// Imported operator set versions.
    #[prost(message, repeated, tag = "8")]
    pub opset_import: Vec<OperatorSetIdProto>,
    /// Key-value model metadata entries.
    #[prost(message, repeated, tag = "14")]
    pub metadata_props: Vec<StringStringEntryProto>,
}

/// Imported operator set version (`onnx.OperatorSetIdProto`).
#[derive(Clone, PartialEq, prost::Message)]
pub struct OperatorSetIdProto {
    /// Operator domain (`""` for default ONNX opset).
    #[prost(string, tag = "1")]
    pub domain: String,
    /// Operator set version integer.
    #[prost(int64, tag = "2")]
    pub version: i64,
}

/// Key-value string metadata entry (`onnx.StringStringEntryProto`).
#[derive(Clone, PartialEq, prost::Message)]
pub struct StringStringEntryProto {
    /// Metadata key.
    #[prost(string, tag = "1")]
    pub key: String,
    /// Metadata value.
    #[prost(string, tag = "2")]
    pub value: String,
}

/// Computation graph (`onnx.GraphProto`).
#[derive(Clone, PartialEq, prost::Message)]
pub struct GraphProto {
    /// Topologically sorted operator nodes.
    #[prost(message, repeated, tag = "1")]
    pub node: Vec<NodeProto>,
    /// Graph identifier.
    #[prost(string, tag = "2")]
    pub name: String,
    /// Constant weight initializers.
    #[prost(message, repeated, tag = "5")]
    pub initializer: Vec<TensorProto>,
    /// Human-readable graph documentation string.
    #[prost(string, tag = "10")]
    pub doc_string: String,
    /// Declared graph inputs.
    #[prost(message, repeated, tag = "11")]
    pub input: Vec<ValueInfoProto>,
    /// Declared graph outputs.
    #[prost(message, repeated, tag = "12")]
    pub output: Vec<ValueInfoProto>,
    /// Intermediate tensor shape annotations.
    #[prost(message, repeated, tag = "13")]
    pub value_info: Vec<ValueInfoProto>,
    /// Key-value graph metadata entries.
    #[prost(message, repeated, tag = "16")]
    pub metadata_props: Vec<StringStringEntryProto>,
}

/// Operator node (`onnx.NodeProto`).
#[derive(Clone, PartialEq, prost::Message)]
pub struct NodeProto {
    /// Input value names.
    #[prost(string, repeated, tag = "1")]
    pub input: Vec<String>,
    /// Output value names.
    #[prost(string, repeated, tag = "2")]
    pub output: Vec<String>,
    /// Node identifier.
    #[prost(string, tag = "3")]
    pub name: String,
    /// ONNX operator name.
    #[prost(string, tag = "4")]
    pub op_type: String,
    /// Operator attributes.
    #[prost(message, repeated, tag = "5")]
    pub attribute: Vec<AttributeProto>,
    /// Human-readable node documentation string.
    #[prost(string, tag = "6")]
    pub doc_string: String,
    /// Operator domain.
    #[prost(string, tag = "7")]
    pub domain: String,
    /// Function overload identifier.
    #[prost(string, tag = "8")]
    pub overload: String,
    /// Key-value node metadata entries.
    #[prost(message, repeated, tag = "9")]
    pub metadata_props: Vec<StringStringEntryProto>,
}

/// Operator attribute (`onnx.AttributeProto`).
#[derive(Clone, PartialEq, prost::Message)]
pub struct AttributeProto {
    /// Attribute name.
    #[prost(string, tag = "1")]
    pub name: String,
    /// Scalar `f32` attribute value.
    #[prost(float, tag = "2")]
    pub f: f32,
    /// Scalar `i64` attribute value.
    #[prost(int64, tag = "3")]
    pub i: i64,
    /// Byte/string attribute value.
    #[prost(bytes = "vec", tag = "4")]
    pub s: Vec<u8>,
    /// Tensor attribute value.
    #[prost(message, optional, tag = "5")]
    pub t: Option<TensorProto>,
    /// Subgraph attribute value.
    #[prost(message, optional, boxed, tag = "6")]
    pub g: Option<Box<GraphProto>>,
    /// Repeated `f32` attribute values.
    #[prost(float, repeated, tag = "7")]
    pub floats: Vec<f32>,
    /// Repeated `i64` attribute values.
    #[prost(int64, repeated, tag = "8")]
    pub ints: Vec<i64>,
    /// Repeated byte/string attribute values.
    #[prost(bytes = "vec", repeated, tag = "9")]
    pub strings: Vec<Vec<u8>>,
    /// Repeated tensor attribute values.
    #[prost(message, repeated, tag = "10")]
    pub tensors: Vec<TensorProto>,
    /// Repeated subgraph attribute values.
    #[prost(message, repeated, tag = "11")]
    pub graphs: Vec<GraphProto>,
    /// Human-readable attribute documentation string.
    #[prost(string, tag = "13")]
    pub doc_string: String,
    /// Type attribute value.
    #[prost(message, optional, tag = "14")]
    pub tp: Option<TypeProto>,
    /// Repeated type attribute values.
    #[prost(message, repeated, tag = "15")]
    pub type_protos: Vec<TypeProto>,
    /// Discriminator tag (`attribute_proto::AttributeType`).
    #[prost(int32, tag = "20")]
    pub r#type: i32,
    /// Referenced parent-function attribute name.
    #[prost(string, tag = "21")]
    pub ref_attr_name: String,
}

/// Nested types for [`AttributeProto`].
pub mod attribute_proto {
    /// Attribute discriminator values (`onnx.AttributeProto.AttributeType`).
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    #[repr(i32)]
    pub enum AttributeType {
        /// Scalar `i64` attribute.
        Int = 2,
        /// Repeated `i64` list attribute.
        Ints = 7,
    }
}

/// Constant initializer or attribute tensor (`onnx.TensorProto`).
#[derive(Clone, PartialEq, prost::Message)]
pub struct TensorProto {
    /// Tensor dimensions.
    #[prost(int64, repeated, tag = "1")]
    pub dims: Vec<i64>,
    /// Element data type (`1` = `FLOAT`, `7` = `INT64`).
    #[prost(int32, tag = "2")]
    pub data_type: i32,
    /// Packed `f32` elements.
    #[prost(float, repeated, tag = "4")]
    pub float_data: Vec<f32>,
    /// Packed `i32` elements.
    #[prost(int32, repeated, tag = "5")]
    pub int32_data: Vec<i32>,
    /// String elements.
    #[prost(bytes = "vec", repeated, tag = "6")]
    pub string_data: Vec<Vec<u8>>,
    /// Packed `i64` elements.
    #[prost(int64, repeated, tag = "7")]
    pub int64_data: Vec<i64>,
    /// Tensor name.
    #[prost(string, tag = "8")]
    pub name: String,
    /// Little-endian raw tensor byte buffer.
    #[prost(bytes = "vec", tag = "9")]
    pub raw_data: Vec<u8>,
    /// Packed `f64` elements.
    #[prost(double, repeated, tag = "10")]
    pub double_data: Vec<f64>,
    /// Packed `u64` elements.
    #[prost(uint64, repeated, tag = "11")]
    pub uint64_data: Vec<u64>,
    /// Human-readable tensor documentation string.
    #[prost(string, tag = "12")]
    pub doc_string: String,
    /// External tensor storage descriptors.
    #[prost(message, repeated, tag = "13")]
    pub external_data: Vec<StringStringEntryProto>,
    /// Storage location enum (`0` = `DEFAULT`, `1` = `EXTERNAL`).
    #[prost(int32, tag = "14")]
    pub data_location: i32,
    /// Key-value tensor metadata entries.
    #[prost(message, repeated, tag = "16")]
    pub metadata_props: Vec<StringStringEntryProto>,
}

/// Typed graph value descriptor (`onnx.ValueInfoProto`).
#[derive(Clone, PartialEq, prost::Message)]
pub struct ValueInfoProto {
    /// Value identifier.
    #[prost(string, tag = "1")]
    pub name: String,
    /// Value type and shape.
    #[prost(message, optional, tag = "2")]
    pub r#type: Option<TypeProto>,
    /// Human-readable value documentation string.
    #[prost(string, tag = "3")]
    pub doc_string: String,
    /// Key-value metadata entries.
    #[prost(message, repeated, tag = "4")]
    pub metadata_props: Vec<StringStringEntryProto>,
}

/// Value type descriptor (`onnx.TypeProto`).
#[derive(Clone, PartialEq, prost::Message)]
pub struct TypeProto {
    /// Semantic type denotation.
    #[prost(string, tag = "6")]
    pub denotation: String,
    /// Concrete type variant.
    #[prost(oneof = "type_proto::Value", tags = "1")]
    pub value: Option<type_proto::Value>,
}

/// Nested types for [`TypeProto`].
pub mod type_proto {
    /// Dense tensor type (`onnx.TypeProto.Tensor`).
    #[derive(Clone, PartialEq, prost::Message)]
    pub struct Tensor {
        /// Element data type (`1` = `FLOAT`, `7` = `INT64`).
        #[prost(int32, tag = "1")]
        pub elem_type: i32,
        /// Optional tensor shape.
        #[prost(message, optional, tag = "2")]
        pub shape: Option<super::TensorShapeProto>,
    }

    /// Type union (`onnx.TypeProto.value`).
    #[derive(Clone, PartialEq, prost::Oneof)]
    pub enum Value {
        /// Dense tensor value.
        #[prost(message, tag = "1")]
        TensorType(Tensor),
    }
}

/// Tensor shape (`onnx.TensorShapeProto`).
#[derive(Clone, PartialEq, prost::Message)]
pub struct TensorShapeProto {
    /// Ordered tensor dimensions.
    #[prost(message, repeated, tag = "1")]
    pub dim: Vec<tensor_shape_proto::Dimension>,
}

/// Nested types for [`TensorShapeProto`].
pub mod tensor_shape_proto {
    /// Single tensor dimension (`onnx.TensorShapeProto.Dimension`).
    #[derive(Clone, PartialEq, prost::Message)]
    pub struct Dimension {
        /// Semantic dimension denotation.
        #[prost(string, tag = "3")]
        pub denotation: String,
        /// Concrete dimension bound.
        #[prost(oneof = "dimension::Value", tags = "1, 2")]
        pub value: Option<dimension::Value>,
    }

    /// Nested types for [`Dimension`].
    pub mod dimension {
        /// Concrete or symbolic dimension bound (`onnx.TensorShapeProto.Dimension.value`).
        #[derive(Clone, PartialEq, prost::Oneof)]
        pub enum Value {
            /// Static integer dimension.
            #[prost(int64, tag = "1")]
            DimValue(i64),
            /// Symbolic dimension parameter name.
            #[prost(string, tag = "2")]
            DimParam(String),
        }
    }
}
