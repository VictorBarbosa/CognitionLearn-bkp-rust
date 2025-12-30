use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct AgentAction {
    pub continuous_actions: Vec<f32>,
    pub discrete_actions: Vec<i32>,
}

#[derive(Debug, Clone)]
pub struct AgentInfo {
    pub id: i32,
    pub observations: Vec<f32>,
    pub reward: f32,
    pub done: bool,
    pub max_step_reached: bool,
    pub sensor_shapes: Vec<i64>,
    pub action_shapes: Vec<i64>,
    pub stored_discrete_actions: Vec<i32>, // For imitation learning / checking
}

#[derive(Debug, Clone)]
pub struct TransitionInfo {
    pub id: i32,
    pub observations: Vec<f32>,
    pub actions: Vec<f32>, // Continuous
    pub discrete_actions: Vec<i32>, // Discrete
    pub reward: f32,
    pub next_observations: Vec<f32>,
    pub done: bool,
    pub sensor_shapes: Vec<i64>,
    pub action_shapes: Vec<i64>,
}

pub fn parse_step_data(data: &str) -> Result<HashMap<String, Vec<AgentInfo>>, String> {
    let mut all_agents = HashMap::new();
    let mut lines = data.lines();
    
    let first = lines.next().map(|s| s.trim());
    if first != Some("STEP") {
        return Err(format!("Invalid step data format: expected STEP, got {:?}", first));
    }
    
    let mut current_behavior = String::new();
    let mut current_agent: Option<AgentInfo> = None;

    for line in lines {
        let line = line.trim();
        if line.is_empty() { continue; }

        if line.starts_with("BEHAVIOR:") {
            if let Some(agent) = current_agent.take() {
                all_agents.entry(current_behavior.clone()).or_insert_with(Vec::new).push(agent);
            }
            current_behavior = line["BEHAVIOR:".len()..].to_string();
            all_agents.entry(current_behavior.clone()).or_insert_with(Vec::new);
        } else if line == "AGENT" {
            if let Some(agent) = current_agent.take() {
                all_agents.entry(current_behavior.clone()).or_insert_with(Vec::new).push(agent);
            }
            current_agent = Some(AgentInfo {
                id: 0, reward: 0.0, done: false, max_step_reached: false, 
                observations: vec![], sensor_shapes: vec![], action_shapes: vec![],
                stored_discrete_actions: vec![],
            });
        } else if let Some(agent) = current_agent.as_mut() {
            if line.len() < 3 { continue; }
            let first_char = line.chars().next().unwrap().to_ascii_lowercase();
            match first_char {
                'i' => if line.get(..3).map(|s| s.eq_ignore_ascii_case("id:")).unwrap_or(false) {
                    agent.id = line[3..].trim().parse().unwrap_or(0);
                },
                'r' => if line.get(..7).map(|s| s.eq_ignore_ascii_case("reward:")).unwrap_or(false) {
                    agent.reward = line[7..].trim().parse().unwrap_or(0.0);
                },
                'd' => {
                    if line.get(..5).map(|s| s.eq_ignore_ascii_case("done:")).unwrap_or(false) {
                        agent.done = line[5..].trim().parse().unwrap_or(false);
                    } else if line.starts_with("discrete:") {
                        agent.stored_discrete_actions = line[9..].split(|c| c == ';' || c == ',' || c == ' ')
                            .filter(|part| !part.is_empty())
                            .map(|part| part.parse::<i32>().unwrap_or(0))
                            .collect();
                    }
                },
                'm' => if line.get(..15).map(|s| s.eq_ignore_ascii_case("maxstepreached:")).unwrap_or(false) {
                    agent.max_step_reached = line[15..].trim().parse().unwrap_or(false);
                },
                'o' => {
                    let s = if line.starts_with("obs:") { &line[4..] } 
                            else if line.starts_with("observations:") { &line[13..] }
                            else { "" };
                    if !s.is_empty() {
                        agent.observations = s.split(|c| c == ';' || c == ',' || c == ' ')
                            .filter(|part| !part.is_empty())
                            .map(|part| part.parse::<f32>().unwrap_or(0.0))
                            .collect();
                    }
                },
                's' => if line.starts_with("sensor_shapes:") {
                    agent.sensor_shapes = line[14..].split(|c| c == ';' || c == ',' || c == ' ')
                        .filter(|part| !part.is_empty())
                        .map(|part| part.parse::<i64>().unwrap_or(0))
                        .collect();
                },
                'a' => if line.starts_with("action_shapes:") {
                    agent.action_shapes = line[14..].split(|c| c == ';' || c == ',' || c == ' ')
                        .filter(|part| !part.is_empty())
                        .map(|part| part.parse::<i64>().unwrap_or(0))
                        .collect();
                },
                _ => {}
            }
        }
    }
    if let Some(agent) = current_agent {
        all_agents.entry(current_behavior).or_insert_with(Vec::new).push(agent);
    }
    Ok(all_agents)
}

pub fn parse_transition_data(data: &str) -> Result<HashMap<String, Vec<TransitionInfo>>, String> {
    let mut all_transitions = HashMap::new();
    let mut lines = data.lines();
    
    let first = lines.next().map(|s| s.trim());
    if first != Some("TRANSITION") {
        return Err(format!("Invalid transition format: expected TRANSITION, got {:?}", first));
    }
    
    let mut current_behavior = String::new();
    let mut current_trans: Option<TransitionInfo> = None;

    for line in lines {
        let line = line.trim();
        if line.is_empty() { continue; }

        if line.starts_with("BEHAVIOR:") {
            if let Some(trans) = current_trans.take() {
                all_transitions.entry(current_behavior.clone()).or_insert_with(Vec::new).push(trans);
            }
            current_behavior = line["BEHAVIOR:".len()..].to_string();
            all_transitions.entry(current_behavior.clone()).or_insert_with(Vec::new);
        } else if line == "TRANS" {
            if let Some(trans) = current_trans.take() {
                all_transitions.entry(current_behavior.clone()).or_insert_with(Vec::new).push(trans);
            }
            current_trans = Some(TransitionInfo {
                id: 0, reward: 0.0, done: false, observations: vec![], actions: vec![], discrete_actions: vec![],
                next_observations: vec![], sensor_shapes: vec![], action_shapes: vec![],
            });
        } else if let Some(trans) = current_trans.as_mut() {
            if line.len() < 3 { continue; }
            let first_char = line.chars().next().unwrap().to_ascii_lowercase();
            match first_char {
                'i' => if line.get(..3).map(|s| s.eq_ignore_ascii_case("id:")).unwrap_or(false) {
                    trans.id = line[3..].trim().parse().unwrap_or(0);
                },
                'r' => if line.get(..7).map(|s| s.eq_ignore_ascii_case("reward:")).unwrap_or(false) {
                    trans.reward = line[7..].trim().parse().unwrap_or(0.0);
                },
                'd' => {
                    if line.get(..5).map(|s| s.eq_ignore_ascii_case("done:")).unwrap_or(false) {
                        trans.done = line[5..].trim().parse().unwrap_or(false);
                    } else if line.starts_with("discrete:") {
                        trans.discrete_actions = line[9..].split(|c| c == ';' || c == ',' || c == ' ')
                            .filter(|part| !part.is_empty())
                            .map(|part| part.parse::<i32>().unwrap_or(0))
                            .collect();
                    }
                },
                'o' => {
                    let s = if line.starts_with("obs:") { &line[4..] } 
                            else if line.starts_with("observations:") { &line[13..] }
                            else { "" };
                    if !s.is_empty() {
                        trans.observations = s.split(|c| c == ';' || c == ',' || c == ' ')
                            .filter(|part| !part.is_empty())
                            .map(|part| part.parse::<f32>().unwrap_or(0.0))
                            .collect();
                    }
                },
                'a' => {
                    if line.starts_with("act:") || line.starts_with("actions:") {
                        let s = if line.starts_with("act:") { &line[4..] } else { &line[8..] };
                        trans.actions = s.split(|c| c == ';' || c == ',' || c == ' ')
                            .filter(|part| !part.is_empty())
                            .map(|part| part.parse::<f32>().unwrap_or(0.0))
                            .collect();
                    } else if line.starts_with("action_shapes:") {
                        trans.action_shapes = line[14..].split(|c| c == ';' || c == ',' || c == ' ')
                            .filter(|part| !part.is_empty())
                            .map(|part| part.parse::<i64>().unwrap_or(0))
                            .collect();
                    }
                },
                'n' => if line.starts_with("next_obs:") || line.starts_with("next_observations:") {
                    let s = if line.starts_with("next_obs:") { &line[9..] } else { &line[18..] };
                    trans.next_observations = s.split(|c| c == ';' || c == ',' || c == ' ')
                        .filter(|part| !part.is_empty())
                        .map(|part| part.parse::<f32>().unwrap_or(0.0))
                        .collect();
                },
                's' => if line.starts_with("sensor_shapes:") {
                    trans.sensor_shapes = line[14..].split(|c| c == ';' || c == ',' || c == ' ')
                        .filter(|part| !part.is_empty())
                        .map(|part| part.parse::<i64>().unwrap_or(0))
                        .collect();
                },
                _ => {}
            }
        }
    }
    if let Some(trans) = current_trans {
        all_transitions.entry(current_behavior).or_insert_with(Vec::new).push(trans);
    }
    Ok(all_transitions)
}

pub fn format_action_data(actions: &HashMap<String, Vec<AgentAction>>) -> String {
    let mut output = String::new();
    output.push_str("ACTIONS\n");
    for (behavior, agent_actions) in actions {
        output.push_str(&format!("BEHAVIOR:{}\n", behavior));
        for action in agent_actions {
            output.push_str("AGENT\n");
            
            // Continuous
            let cont_str: Vec<String> = action.continuous_actions.iter().map(|a| a.to_string()).collect();
            output.push_str(&format!("continuous:{}\n", cont_str.join(";")));
            
            // Discrete (Only if present)
            if !action.discrete_actions.is_empty() {
                let disc_str: Vec<String> = action.discrete_actions.iter().map(|a| a.to_string()).collect();
                output.push_str(&format!("discrete:{}\n", disc_str.join(";")));
            }
        }
    }
    output
}
