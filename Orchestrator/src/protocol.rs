use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct AgentAction {
    pub continuous_actions: Vec<f32>,
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

}



#[derive(Debug, Clone)]

pub struct TransitionInfo {

    pub id: i32,

    pub observations: Vec<f32>,

    pub actions: Vec<f32>,

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

                id: 0, reward: 0.0, done: false, max_step_reached: false, observations: vec![], sensor_shapes: vec![], action_shapes: vec![],

            });

        } else if let Some(agent) = current_agent.as_mut() {
            let lower_line = line.to_lowercase();
            if let Some(val) = lower_line.strip_prefix("id:") {
                agent.id = val.trim().parse().unwrap_or(0);
            } else if let Some(val) = lower_line.strip_prefix("reward:") {
                agent.reward = val.trim().parse().unwrap_or(0.0);
            } else if let Some(val) = lower_line.strip_prefix("done:") {
                agent.done = val.trim().parse().unwrap_or(false);
            } else if let Some(val) = lower_line.strip_prefix("maxstepreached:") {
                agent.max_step_reached = val.trim().parse().unwrap_or(false);
            } else if lower_line.starts_with("obs:") || lower_line.starts_with("observations:") {
                let s = if lower_line.starts_with("obs:") { &line["obs:".len()..] } else { &line["observations:".len()..] };
                agent.observations = s.split(|c| c == ';' || c == ',' || c == ' ')
                    .map(|s| s.trim()).filter(|s| !s.is_empty())
                    .filter_map(|s| s.parse().ok()).collect();
            } else if lower_line.starts_with("sensor_shapes:") {
                let s = &line["sensor_shapes:".len()..];
                agent.sensor_shapes = s.split(|c| c == ';' || c == ',' || c == ' ')
                    .map(|s| s.trim()).filter(|s| !s.is_empty())
                    .filter_map(|s| s.parse().ok()).collect();
            } else if lower_line.starts_with("action_shapes:") {
                let s = &line["action_shapes:".len()..];
                agent.action_shapes = s.split(|c| c == ';' || c == ',' || c == ' ')
                    .map(|s| s.trim()).filter(|s| !s.is_empty())
                    .filter_map(|s| s.parse().ok()).collect();
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

                id: 0, reward: 0.0, done: false, observations: vec![], actions: vec![], next_observations: vec![], sensor_shapes: vec![], action_shapes: vec![],

            });

        } else if let Some(trans) = current_trans.as_mut() {
            let lower_line = line.to_lowercase();
            if let Some(val) = lower_line.strip_prefix("id:") {
                trans.id = val.trim().parse().unwrap_or(0);
            } else if let Some(val) = lower_line.strip_prefix("reward:") {
                trans.reward = val.trim().parse().unwrap_or(0.0);
            } else if let Some(val) = lower_line.strip_prefix("done:") {
                trans.done = val.trim().parse().unwrap_or(false);
            } else if lower_line.starts_with("obs:") || lower_line.starts_with("observations:") {
                let s = if lower_line.starts_with("obs:") { &line["obs:".len()..] } else { &line["observations:".len()..] };
                trans.observations = s.split(|c| c == ';' || c == ',' || c == ' ')
                    .map(|s| s.trim()).filter(|s| !s.is_empty())
                    .filter_map(|s| s.parse().ok()).collect();
            } else if lower_line.starts_with("act:") || lower_line.starts_with("actions:") {
                let s = if lower_line.starts_with("act:") { &line["act:".len()..] } else { &line["actions:".len()..] };
                trans.actions = s.split(|c| c == ';' || c == ',' || c == ' ')
                    .map(|s| s.trim()).filter(|s| !s.is_empty())
                    .filter_map(|s| s.parse().ok()).collect();
            } else if lower_line.starts_with("next_obs:") || lower_line.starts_with("next_observations:") {
                let s = if lower_line.starts_with("next_obs:") { &line["next_obs:".len()..] } else { &line["next_observations:".len()..] };
                trans.next_observations = s.split(|c| c == ';' || c == ',' || c == ' ')
                    .map(|s| s.trim()).filter(|s| !s.is_empty())
                    .filter_map(|s| s.parse().ok()).collect();
            } else if lower_line.starts_with("sensor_shapes:") {
                let s = &line["sensor_shapes:".len()..];
                trans.sensor_shapes = s.split(|c| c == ';' || c == ',' || c == ' ')
                    .map(|s| s.trim()).filter(|s| !s.is_empty())
                    .filter_map(|s| s.parse().ok()).collect();
            } else if lower_line.starts_with("action_shapes:") {
                let s = &line["action_shapes:".len()..];
                trans.action_shapes = s.split(|c| c == ';' || c == ',' || c == ' ')
                    .map(|s| s.trim()).filter(|s| !s.is_empty())
                    .filter_map(|s| s.parse().ok()).collect();
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
            let action_str: Vec<String> = action.continuous_actions.iter().map(|a| a.to_string()).collect();
            output.push_str(&format!("continuous:{}\n", action_str.join(";")));
        }
    }
    output
}
