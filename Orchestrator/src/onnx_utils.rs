use tch::Tensor;
use crate::onnx::{ModelProto, GraphProto, NodeProto, TensorProto, ValueInfoProto, TypeProto, tensor_proto, type_proto, tensor_shape_proto, OperatorSetIdProto, attribute_proto, AttributeProto, TensorShapeProto};
use crate::onnx::tensor_shape_proto::Dimension;
use prost::Message;
use std::fs::File;
use std::io::Write;

pub fn export_ppo_onnx(
    common_weight: &Tensor, common_bias: &Tensor,
    actor_weight: &Tensor, actor_bias: &Tensor,
    obs_dim: i64, act_dim: i64, _hidden_dim: i64,
    output_path: &str,
    sensor_sizes: &Option<Vec<i64>>
) -> Result<(), Box<dyn std::error::Error>> {
    
    // Helper to create tensor proto from tch tensor
    let create_tensor_proto = |name: &str, t: &Tensor| -> TensorProto {
        let size = t.size();
        let data: Vec<f32> = t.flatten(0, -1).try_into().unwrap();
        
        TensorProto {
            dims: size,
            data_type: Some(tensor_proto::DataType::Float as i32),
            name: Some(name.to_string()),
            float_data: data,
            ..Default::default()
        }
    };

    // Helper to create value info
    let create_value_info = |name: &str, shape: &[i64], dtype: tensor_proto::DataType| -> ValueInfoProto {
        let dims: Vec<tensor_shape_proto::Dimension> = shape.iter().map(|&d| {
            if d == -1 {
                Dimension { 
                    value: Some(tensor_shape_proto::dimension::Value::DimParam("batch_size".to_string())),
                    ..Default::default()
                }
            } else {
                Dimension { 
                    value: Some(tensor_shape_proto::dimension::Value::DimValue(d)),
                    ..Default::default()
                }
            }
        }).collect();

        ValueInfoProto {
            name: Some(name.to_string()),
            r#type: Some(TypeProto {
                value: Some(type_proto::Value::TensorType(type_proto::Tensor {
                    elem_type: Some(dtype as i32),
                    shape: Some(TensorShapeProto { dim: dims, ..Default::default() }),
                    ..Default::default()
                })),
                ..Default::default()
            }),
            ..Default::default()
        }
    };
    
    // Logic to handle split inputs based on provided configuration
    let mut inputs = Vec::new();
    let mut nodes = Vec::new();
    let input_name_for_matmul: String;
    
    if let Some(sizes) = sensor_sizes {
        // Validate that sizes sum up to obs_dim
        let total_size: i64 = sizes.iter().sum();
        if total_size != obs_dim {
            eprintln!("⚠️ Warning: Provided sensor sizes sum to {}, but obs_dim is {}. Model might fail.", total_size, obs_dim);
        }
        
        let mut concat_inputs = Vec::new();
        for (i, &size) in sizes.iter().enumerate() {
            let name = format!("obs_{}", i);
            
            let inp = create_value_info(&name, &[-1, size], tensor_proto::DataType::Float);
            inputs.push(inp);
            concat_inputs.push(name);
        }
        
        // Add Concat Node
        if inputs.len() > 1 {
            let node_concat = NodeProto {
                op_type: Some("Concat".to_string()),
                input: concat_inputs,
                output: vec!["concat_obs".to_string()],
                attribute: vec![AttributeProto {
                    name: Some("axis".to_string()),
                    r#type: Some(attribute_proto::AttributeType::Int as i32),
                    i: Some(1), // Axis 1 (features)
                    ..Default::default()
                }],
                ..Default::default()
            };
            nodes.push(node_concat);
            input_name_for_matmul = "concat_obs".to_string();
        } else {
             input_name_for_matmul = concat_inputs[0].clone();
        }
        
    } else {
        // Fallback: Single input named 'vector_observation'
        let in_obs = create_value_info("vector_observation", &[-1, obs_dim], tensor_proto::DataType::Float);
        inputs.push(in_obs);
        input_name_for_matmul = "vector_observation".to_string();
    }

    // Initializers
    let w_common_t = common_weight.transpose(0, 1);
    let w_actor_t = actor_weight.transpose(0, 1);
    
    let init_w_common = create_tensor_proto("common.weight", &w_common_t);
    let init_b_common = create_tensor_proto("common.bias", common_bias);
    let init_w_actor = create_tensor_proto("actor.weight", &w_actor_t);
    let init_b_actor = create_tensor_proto("actor.bias", actor_bias);
    
    // Nodes
    // 1. MatMul(input, common.weight) -> x1
    let node1 = NodeProto {
        op_type: Some("MatMul".to_string()),
        input: vec![input_name_for_matmul, "common.weight".to_string()],
        output: vec!["x1".to_string()],
        ..Default::default()
    };
    nodes.push(node1);
    
    // 2. Add
    let node2 = NodeProto {
        op_type: Some("Add".to_string()),
        input: vec!["x1".to_string(), "common.bias".to_string()],
        output: vec!["x2".to_string()],
        ..Default::default()
    };
    nodes.push(node2);
    
    // 3. Relu
    let node3 = NodeProto {
        op_type: Some("Relu".to_string()),
        input: vec!["x2".to_string()],
        output: vec!["x3".to_string()],
        ..Default::default()
    };
    nodes.push(node3);
    
    // 4. MatMul
    let node4 = NodeProto {
        op_type: Some("MatMul".to_string()),
        input: vec!["x3".to_string(), "actor.weight".to_string()],
        output: vec!["x4".to_string()],
        ..Default::default()
    };
    nodes.push(node4);
    
    // 5. Add
    let node5 = NodeProto {
        op_type: Some("Add".to_string()),
        input: vec!["x4".to_string(), "actor.bias".to_string()],
        output: vec!["x5".to_string()],
        ..Default::default()
    };
    nodes.push(node5);

    // 6. Tanh -> continuous_actions
    let node6 = NodeProto {
        op_type: Some("Tanh".to_string()),
        input: vec!["x5".to_string()],
        output: vec!["continuous_actions".to_string()],
        ..Default::default()
    };
    nodes.push(node6);
    
    // 7. Identity -> deterministic_continuous_actions
    let node_det = NodeProto {
        op_type: Some("Identity".to_string()),
        input: vec!["continuous_actions".to_string()],
        output: vec!["deterministic_continuous_actions".to_string()],
        ..Default::default()
    };
    nodes.push(node_det);
    
    // Metadata Constants (FLOAT)
    let node_ver_const = NodeProto {
        op_type: Some("Constant".to_string()),
        output: vec!["const_version".to_string()],
        attribute: vec![AttributeProto {
            name: Some("value".to_string()),
            r#type: Some(attribute_proto::AttributeType::Tensor as i32),
            t: Some(TensorProto {
                data_type: Some(tensor_proto::DataType::Float as i32),
                float_data: vec![3.0],
                dims: vec![1],
                ..Default::default()
            }),
            ..Default::default()
        }],
        ..Default::default()
    };
    let node_ver_id = NodeProto {
        op_type: Some("Identity".to_string()),
        input: vec!["const_version".to_string()],
        output: vec!["version_number".to_string()],
        ..Default::default()
    };
    nodes.push(node_ver_const);
    nodes.push(node_ver_id);

    // Memory Size
    let node_mem_const = NodeProto {
        op_type: Some("Constant".to_string()),
        output: vec!["const_memory".to_string()],
        attribute: vec![AttributeProto {
            name: Some("value".to_string()),
            r#type: Some(attribute_proto::AttributeType::Tensor as i32),
            t: Some(TensorProto {
                data_type: Some(tensor_proto::DataType::Float as i32),
                float_data: vec![0.0],
                dims: vec![1],
                ..Default::default()
            }),
            ..Default::default()
        }],
        ..Default::default()
    };
    let node_mem_id = NodeProto {
        op_type: Some("Identity".to_string()),
        input: vec!["const_memory".to_string()],
        output: vec!["memory_size".to_string()],
        ..Default::default()
    };
    nodes.push(node_mem_const);
    nodes.push(node_mem_id);
    
    // Output Shape
    let node_shape_const = NodeProto {
        op_type: Some("Constant".to_string()),
        output: vec!["const_shape".to_string()],
        attribute: vec![AttributeProto {
            name: Some("value".to_string()),
            r#type: Some(attribute_proto::AttributeType::Tensor as i32),
            t: Some(TensorProto {
                data_type: Some(tensor_proto::DataType::Float as i32),
                float_data: vec![act_dim as f32],
                dims: vec![1],
                ..Default::default()
            }),
            ..Default::default()
        }],
        ..Default::default()
    };
    let node_shape_id = NodeProto {
        op_type: Some("Identity".to_string()),
        input: vec!["const_shape".to_string()],
        output: vec!["continuous_action_output_shape".to_string()],
        ..Default::default()
    };
    nodes.push(node_shape_const);
    nodes.push(node_shape_id);

    // Outputs
    let out_actions = create_value_info("continuous_actions", &[-1, act_dim], tensor_proto::DataType::Float);
    let out_det_actions = create_value_info("deterministic_continuous_actions", &[-1, act_dim], tensor_proto::DataType::Float);
    let out_version = create_value_info("version_number", &[1], tensor_proto::DataType::Float);
    let out_memory = create_value_info("memory_size", &[1], tensor_proto::DataType::Float);
    let out_shape = create_value_info("continuous_action_output_shape", &[1], tensor_proto::DataType::Float);

    let graph = GraphProto {
        node: nodes,
        name: Some("PPO_Export".to_string()),
        initializer: vec![init_w_common, init_b_common, init_w_actor, init_b_actor],
        input: inputs,
        output: vec![out_actions, out_det_actions, out_version, out_memory, out_shape],
        ..Default::default()
    };

    let model = ModelProto {
        ir_version: Some(4),
        producer_name: Some("Rust_Orchestrator".to_string()),
        producer_version: Some("0.1".to_string()),
        graph: Some(graph),
        opset_import: vec![OperatorSetIdProto { domain: Some("".to_string()), version: Some(9), ..Default::default() }],
        ..Default::default()
    };

    let mut file = File::create(output_path)?;
    let mut buf = Vec::new();
    model.encode(&mut buf)?;
    file.write_all(&buf)?;
    
    Ok(())
}

pub fn export_sac_onnx(
    l1_w: &Tensor, l1_b: &Tensor,
    l2_w: &Tensor, l2_b: &Tensor,
    mean_w: &Tensor, mean_b: &Tensor,
    obs_dim: i64, act_dim: i64,
    output_path: &str,
    sensor_sizes: &Option<Vec<i64>>
) -> Result<(), Box<dyn std::error::Error>> {
    
    // Helper to create tensor proto from tch tensor
    let create_tensor_proto = |name: &str, t: &Tensor| -> TensorProto {
        let size = t.size();
        let data: Vec<f32> = t.flatten(0, -1).try_into().unwrap();
        
        TensorProto {
            dims: size,
            data_type: Some(tensor_proto::DataType::Float as i32),
            name: Some(name.to_string()),
            float_data: data,
            ..Default::default()
        }
    };

    // Helper to create value info
    let create_value_info = |name: &str, shape: &[i64], dtype: tensor_proto::DataType| -> ValueInfoProto {
        let dims: Vec<tensor_shape_proto::Dimension> = shape.iter().map(|&d| {
            if d == -1 {
                Dimension { 
                    value: Some(tensor_shape_proto::dimension::Value::DimParam("batch_size".to_string())),
                    ..Default::default()
                }
            } else {
                Dimension { 
                    value: Some(tensor_shape_proto::dimension::Value::DimValue(d)),
                    ..Default::default()
                }
            }
        }).collect();

        ValueInfoProto {
            name: Some(name.to_string()),
            r#type: Some(TypeProto {
                value: Some(type_proto::Value::TensorType(type_proto::Tensor {
                    elem_type: Some(dtype as i32),
                    shape: Some(TensorShapeProto { dim: dims, ..Default::default() }),
                    ..Default::default()
                })),
                ..Default::default()
            }),
            ..Default::default()
        }
    };
    
    let mut inputs = Vec::new();
    let mut nodes = Vec::new();
    let input_name_for_net: String;
    
    if let Some(sizes) = sensor_sizes {
        let mut concat_inputs = Vec::new();
        for (i, &size) in sizes.iter().enumerate() {
            let name = format!("obs_{}", i);
            let inp = create_value_info(&name, &[-1, size], tensor_proto::DataType::Float);
            inputs.push(inp);
            concat_inputs.push(name);
        }
        
        if inputs.len() > 1 {
            let node_concat = NodeProto {
                op_type: Some("Concat".to_string()),
                input: concat_inputs,
                output: vec!["concat_obs".to_string()],
                attribute: vec![AttributeProto {
                    name: Some("axis".to_string()),
                    r#type: Some(attribute_proto::AttributeType::Int as i32),
                    i: Some(1),
                    ..Default::default()
                }],
                ..Default::default()
            };
            nodes.push(node_concat);
            input_name_for_net = "concat_obs".to_string();
        } else {
             input_name_for_net = concat_inputs[0].clone();
        }
    } else {
        let in_obs = create_value_info("vector_observation", &[-1, obs_dim], tensor_proto::DataType::Float);
        inputs.push(in_obs);
        input_name_for_net = "vector_observation".to_string();
    }

    // Initializers (Transposed weights for ONNX MatMul B)
    let w_l1_t = l1_w.transpose(0, 1);
    let w_l2_t = l2_w.transpose(0, 1);
    let w_mean_t = mean_w.transpose(0, 1);
    
    let init_w_l1 = create_tensor_proto("l1.weight", &w_l1_t);
    let init_b_l1 = create_tensor_proto("l1.bias", l1_b);
    let init_w_l2 = create_tensor_proto("l2.weight", &w_l2_t);
    let init_b_l2 = create_tensor_proto("l2.bias", l2_b);
    let init_w_mean = create_tensor_proto("mean.weight", &w_mean_t);
    let init_b_mean = create_tensor_proto("mean.bias", mean_b);
    
    // Layer 1: MatMul + Add + Relu
    nodes.push(NodeProto {
        op_type: Some("MatMul".to_string()),
        input: vec![input_name_for_net, "l1.weight".to_string()],
        output: vec!["l1_mm".to_string()],
        ..Default::default()
    });
    nodes.push(NodeProto {
        op_type: Some("Add".to_string()),
        input: vec!["l1_mm".to_string(), "l1.bias".to_string()],
        output: vec!["l1_add".to_string()],
        ..Default::default()
    });
    nodes.push(NodeProto {
        op_type: Some("Relu".to_string()),
        input: vec!["l1_add".to_string()],
        output: vec!["l1_out".to_string()],
        ..Default::default()
    });
    
    // Layer 2: MatMul + Add + Relu
    nodes.push(NodeProto {
        op_type: Some("MatMul".to_string()),
        input: vec!["l1_out".to_string(), "l2.weight".to_string()],
        output: vec!["l2_mm".to_string()],
        ..Default::default()
    });
    nodes.push(NodeProto {
        op_type: Some("Add".to_string()),
        input: vec!["l2_mm".to_string(), "l2.bias".to_string()],
        output: vec!["l2_add".to_string()],
        ..Default::default()
    });
    nodes.push(NodeProto {
        op_type: Some("Relu".to_string()),
        input: vec!["l2_add".to_string()],
        output: vec!["l2_out".to_string()],
        ..Default::default()
    });

    // Mean Layer: MatMul + Add
    nodes.push(NodeProto {
        op_type: Some("MatMul".to_string()),
        input: vec!["l2_out".to_string(), "mean.weight".to_string()],
        output: vec!["mean_mm".to_string()],
        ..Default::default()
    });
    nodes.push(NodeProto {
        op_type: Some("Add".to_string()),
        input: vec!["mean_mm".to_string(), "mean.bias".to_string()],
        output: vec!["mean_pre_tanh".to_string()],
        ..Default::default()
    });
    
    // Tanh Output
    nodes.push(NodeProto {
        op_type: Some("Tanh".to_string()),
        input: vec!["mean_pre_tanh".to_string()],
        output: vec!["continuous_actions".to_string()],
        ..Default::default()
    });

    // Identity for deterministic actions (same as continuous_actions)
    nodes.push(NodeProto {
        op_type: Some("Identity".to_string()),
        input: vec!["continuous_actions".to_string()],
        output: vec!["deterministic_continuous_actions".to_string()],
        ..Default::default()
    });
    
    // Metadata
    let node_ver_const = NodeProto {
        op_type: Some("Constant".to_string()),
        output: vec!["const_version".to_string()],
        attribute: vec![AttributeProto {
            name: Some("value".to_string()),
            r#type: Some(attribute_proto::AttributeType::Tensor as i32),
            t: Some(TensorProto {
                data_type: Some(tensor_proto::DataType::Float as i32),
                float_data: vec![3.0],
                dims: vec![1],
                ..Default::default()
            }),
            ..Default::default()
        }],
        ..Default::default()
    };
    let node_ver_id = NodeProto {
        op_type: Some("Identity".to_string()),
        input: vec!["const_version".to_string()],
        output: vec!["version_number".to_string()],
        ..Default::default()
    };
    nodes.push(node_ver_const);
    nodes.push(node_ver_id);

    let node_mem_const = NodeProto {
        op_type: Some("Constant".to_string()),
        output: vec!["const_memory".to_string()],
        attribute: vec![AttributeProto {
            name: Some("value".to_string()),
            r#type: Some(attribute_proto::AttributeType::Tensor as i32),
            t: Some(TensorProto {
                data_type: Some(tensor_proto::DataType::Float as i32),
                float_data: vec![0.0],
                dims: vec![1],
                ..Default::default()
            }),
            ..Default::default()
        }],
        ..Default::default()
    };
    let node_mem_id = NodeProto {
        op_type: Some("Identity".to_string()),
        input: vec!["const_memory".to_string()],
        output: vec!["memory_size".to_string()],
        ..Default::default()
    };
    nodes.push(node_mem_const);
    nodes.push(node_mem_id);
    
    let node_shape_const = NodeProto {
        op_type: Some("Constant".to_string()),
        output: vec!["const_shape".to_string()],
        attribute: vec![AttributeProto {
            name: Some("value".to_string()),
            r#type: Some(attribute_proto::AttributeType::Tensor as i32),
            t: Some(TensorProto {
                data_type: Some(tensor_proto::DataType::Float as i32),
                float_data: vec![act_dim as f32],
                dims: vec![1],
                ..Default::default()
            }),
            ..Default::default()
        }],
        ..Default::default()
    };
    let node_shape_id = NodeProto {
        op_type: Some("Identity".to_string()),
        input: vec!["const_shape".to_string()],
        output: vec!["continuous_action_output_shape".to_string()],
        ..Default::default()
    };
    nodes.push(node_shape_const);
    nodes.push(node_shape_id);

    // Final definition
    let out_actions = create_value_info("continuous_actions", &[-1, act_dim], tensor_proto::DataType::Float);
    let out_det_actions = create_value_info("deterministic_continuous_actions", &[-1, act_dim], tensor_proto::DataType::Float);
    let out_version = create_value_info("version_number", &[1], tensor_proto::DataType::Float);
    let out_memory = create_value_info("memory_size", &[1], tensor_proto::DataType::Float);
    let out_shape = create_value_info("continuous_action_output_shape", &[1], tensor_proto::DataType::Float);

    let graph = GraphProto {
        node: nodes,
        name: Some("SAC_Export".to_string()),
        initializer: vec![init_w_l1, init_b_l1, init_w_l2, init_b_l2, init_w_mean, init_b_mean],
        input: inputs,
        output: vec![out_actions, out_det_actions, out_version, out_memory, out_shape],
        ..Default::default()
    };

    let model = ModelProto {
        ir_version: Some(4),
        producer_name: Some("Rust_Orchestrator".to_string()),
        producer_version: Some("0.1".to_string()),
        graph: Some(graph),
        opset_import: vec![OperatorSetIdProto { domain: Some("".to_string()), version: Some(9), ..Default::default() }],
        ..Default::default()
    };

    let mut file = File::create(output_path)?;
    let mut buf = Vec::new();
    model.encode(&mut buf)?;
    file.write_all(&buf)?;
    
    Ok(())
}


pub fn export_td3_onnx(
    l1_w: &Tensor, l1_b: &Tensor,
    l2_w: &Tensor, l2_b: &Tensor,
    l3_w: &Tensor, l3_b: &Tensor,
    obs_dim: i64, act_dim: i64,
    output_path: &str,
    sensor_sizes: &Option<Vec<i64>>
) -> Result<(), Box<dyn std::error::Error>> {
    
    // Helper to create tensor proto from tch tensor
    let create_tensor_proto = |name: &str, t: &Tensor| -> TensorProto {
        let size = t.size();
        let data: Vec<f32> = t.flatten(0, -1).try_into().unwrap();
        
        TensorProto {
            dims: size,
            data_type: Some(tensor_proto::DataType::Float as i32),
            name: Some(name.to_string()),
            float_data: data,
            ..Default::default()
        }
    };

    // Helper to create value info
    let create_value_info = |name: &str, shape: &[i64], dtype: tensor_proto::DataType| -> ValueInfoProto {
        let dims: Vec<tensor_shape_proto::Dimension> = shape.iter().map(|&d| {
            if d == -1 {
                Dimension { 
                    value: Some(tensor_shape_proto::dimension::Value::DimParam("batch_size".to_string())),
                    ..Default::default()
                }
            } else {
                Dimension { 
                    value: Some(tensor_shape_proto::dimension::Value::DimValue(d)),
                    ..Default::default()
                }
            }
        }).collect();

        ValueInfoProto {
            name: Some(name.to_string()),
            r#type: Some(TypeProto {
                value: Some(type_proto::Value::TensorType(type_proto::Tensor {
                    elem_type: Some(dtype as i32),
                    shape: Some(TensorShapeProto { dim: dims, ..Default::default() }),
                    ..Default::default()
                })),
                ..Default::default()
            }),
            ..Default::default()
        }
    };

    let mut inputs = Vec::new();
    let mut nodes = Vec::new();
    let input_name_for_net: String;
    
    // Input Handling
    if let Some(sizes) = sensor_sizes {
        let mut concat_inputs = Vec::new();
        for (i, &size) in sizes.iter().enumerate() {
            let name = format!("obs_{}", i);
            let inp = create_value_info(&name, &[-1, size], tensor_proto::DataType::Float);
            inputs.push(inp);
            concat_inputs.push(name);
        }
        
        if inputs.len() > 1 {
            nodes.push(NodeProto {
                op_type: Some("Concat".to_string()),
                input: concat_inputs,
                output: vec!["concat_obs".to_string()],
                attribute: vec![AttributeProto {
                    name: Some("axis".to_string()),
                    r#type: Some(attribute_proto::AttributeType::Int as i32),
                    i: Some(1),
                    ..Default::default()
                }],
                ..Default::default()
            });
            input_name_for_net = "concat_obs".to_string();
        } else {
             input_name_for_net = concat_inputs[0].clone();
        }
    } else {
        let in_obs = create_value_info("vector_observation", &[-1, obs_dim], tensor_proto::DataType::Float);
        inputs.push(in_obs);
        input_name_for_net = "vector_observation".to_string();
    }

    // Initializers (Weights)
    let w_l1_t = l1_w.transpose(0, 1);
    let w_l2_t = l2_w.transpose(0, 1);
    let w_l3_t = l3_w.transpose(0, 1);
    
    let init_ws = vec![
        create_tensor_proto("l1.weight", &w_l1_t),
        create_tensor_proto("l1.bias", l1_b),
        create_tensor_proto("l2.weight", &w_l2_t),
        create_tensor_proto("l2.bias", l2_b),
        create_tensor_proto("l3.weight", &w_l3_t),
        create_tensor_proto("l3.bias", l3_b),
    ];
    
    // Layer 1: MatMul + Add + Relu
    nodes.push(NodeProto {
        op_type: Some("MatMul".to_string()),
        input: vec![input_name_for_net, "l1.weight".to_string()],
        output: vec!["l1_mm".to_string()],
        ..Default::default()
    });
    nodes.push(NodeProto {
        op_type: Some("Add".to_string()),
        input: vec!["l1_mm".to_string(), "l1.bias".to_string()],
        output: vec!["l1_add".to_string()],
        ..Default::default()
    });
    nodes.push(NodeProto {
        op_type: Some("Relu".to_string()),
        input: vec!["l1_add".to_string()],
        output: vec!["l1_relu".to_string()],
        ..Default::default()
    });
    
    // Layer 2: MatMul + Add + Relu
    nodes.push(NodeProto {
        op_type: Some("MatMul".to_string()),
        input: vec!["l1_relu".to_string(), "l2.weight".to_string()],
        output: vec!["l2_mm".to_string()],
        ..Default::default()
    });
    nodes.push(NodeProto {
        op_type: Some("Add".to_string()),
        input: vec!["l2_mm".to_string(), "l2.bias".to_string()],
        output: vec!["l2_add".to_string()],
        ..Default::default()
    });
    nodes.push(NodeProto {
        op_type: Some("Relu".to_string()),
        input: vec!["l2_add".to_string()],
        output: vec!["l2_relu".to_string()],
        ..Default::default()
    });

    // Layer 3: MatMul + Add + Tanh -> action
    nodes.push(NodeProto {
        op_type: Some("MatMul".to_string()),
        input: vec!["l2_relu".to_string(), "l3.weight".to_string()],
        output: vec!["l3_mm".to_string()],
        ..Default::default()
    });
    nodes.push(NodeProto {
        op_type: Some("Add".to_string()),
        input: vec!["l3_mm".to_string(), "l3.bias".to_string()],
        output: vec!["l3_add".to_string()],
        ..Default::default()
    });
    nodes.push(NodeProto {
        op_type: Some("Tanh".to_string()),
        input: vec!["l3_add".to_string()],
        output: vec!["continuous_actions".to_string()],
        ..Default::default()
    });

    // Identity for deterministic actions
    nodes.push(NodeProto {
        op_type: Some("Identity".to_string()),
        input: vec!["continuous_actions".to_string()],
        output: vec!["deterministic_continuous_actions".to_string()],
        ..Default::default()
    });

    // Alias 'action' for compatibility with various Unity versions
    nodes.push(NodeProto {
        op_type: Some("Identity".to_string()),
        input: vec!["continuous_actions".to_string()],
        output: vec!["action".to_string()],
        ..Default::default()
    });
    // Alias 'deterministic_action'
     nodes.push(NodeProto {
        op_type: Some("Identity".to_string()),
        input: vec!["continuous_actions".to_string()],
        output: vec!["deterministic_action".to_string()],
        ..Default::default()
    });

    // Version Number Node
     nodes.push(NodeProto {
        op_type: Some("Constant".to_string()),
        output: vec!["const_version".to_string()],
        attribute: vec![AttributeProto {
            name: Some("value".to_string()),
            r#type: Some(attribute_proto::AttributeType::Tensor as i32),
            t: Some(TensorProto {
                data_type: Some(tensor_proto::DataType::Float as i32),
                float_data: vec![3.0], // Increased version
                dims: vec![1],
                ..Default::default()
            }),
            ..Default::default()
        }],
        ..Default::default()
    });
    nodes.push(NodeProto {
        op_type: Some("Identity".to_string()),
        input: vec!["const_version".to_string()],
        output: vec!["version_number".to_string()],
        ..Default::default()
    });

    // Memory Size Node
     nodes.push(NodeProto {
        op_type: Some("Constant".to_string()),
        output: vec!["const_memory".to_string()],
        attribute: vec![AttributeProto {
            name: Some("value".to_string()),
            r#type: Some(attribute_proto::AttributeType::Tensor as i32),
            t: Some(TensorProto {
                data_type: Some(tensor_proto::DataType::Float as i32),
                float_data: vec![0.0],
                dims: vec![1],
                ..Default::default()
            }),
            ..Default::default()
        }],
        ..Default::default()
    });
    nodes.push(NodeProto {
        op_type: Some("Identity".to_string()),
        input: vec!["const_memory".to_string()],
        output: vec!["memory_size".to_string()],
        ..Default::default()
    });

    // Output Shape Node (Critical for Unity validation)
    nodes.push(NodeProto {
        op_type: Some("Constant".to_string()),
        output: vec!["const_shape".to_string()],
        attribute: vec![AttributeProto {
            name: Some("value".to_string()),
            r#type: Some(attribute_proto::AttributeType::Tensor as i32),
            t: Some(TensorProto {
                data_type: Some(tensor_proto::DataType::Float as i32),
                float_data: vec![act_dim as f32],
                dims: vec![1],
                ..Default::default()
            }),
            ..Default::default()
        }],
        ..Default::default()
    });
    nodes.push(NodeProto {
        op_type: Some("Identity".to_string()),
        input: vec!["const_shape".to_string()],
        output: vec!["continuous_action_output_shape".to_string()],
        ..Default::default()
    });

    // Outputs
    let out_actions = create_value_info("continuous_actions", &[-1, act_dim], tensor_proto::DataType::Float);
    let out_det_actions = create_value_info("deterministic_continuous_actions", &[-1, act_dim], tensor_proto::DataType::Float);
    // Legacy/Alternate aliases
    let out_action_alias = create_value_info("action", &[-1, act_dim], tensor_proto::DataType::Float);
    let out_det_action_alias = create_value_info("deterministic_action", &[-1, act_dim], tensor_proto::DataType::Float);

    let out_version = create_value_info("version_number", &[1], tensor_proto::DataType::Float);
    let out_memory = create_value_info("memory_size", &[1], tensor_proto::DataType::Float);
    let out_shape = create_value_info("continuous_action_output_shape", &[1], tensor_proto::DataType::Float);

    let graph = GraphProto {
        node: nodes,
        name: Some("TD3_Actor".to_string()),
        initializer: init_ws,
        input: inputs,
        // Expose ALL variations to ensure Unity finds what it wants
        output: vec![out_actions, out_det_actions, out_action_alias, out_det_action_alias, out_version, out_memory, out_shape], 
        value_info: vec![],
        quantization_annotation: vec![],
        doc_string: Some("".to_string()),
        sparse_initializer: vec![],
        ..Default::default()
    };

    let model = ModelProto {
        ir_version: Some(7),
        opset_import: vec![OperatorSetIdProto { domain: Some("".to_string()), version: Some(9) }],
        producer_name: Some("CognitionLearn_Rust".to_string()),
        producer_version: Some("1.0".to_string()),
        domain: Some("".to_string()),
        model_version: Some(1),
        doc_string: Some("".to_string()),
        graph: Some(graph),
        metadata_props: vec![],
        training_info: vec![],
        functions: vec![],
        ..Default::default()
    };

    let mut file = std::fs::File::create(output_path)?;
    let mut buf = Vec::new();
    model.encode(&mut buf)?;
    use std::io::Write;
    file.write_all(&buf)?;

    Ok(())
}
 