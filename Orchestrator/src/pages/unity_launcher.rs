use std::collections::HashMap;
use std::process;
use std::sync::{Arc, atomic::{AtomicBool, Ordering}};
use std::thread;
use std::process::Stdio;
use std::io::{BufRead, BufReader};
use crate::pages::config_types::AlgoConfig;

// Unity launcher logic
pub struct UnityLauncher {
    pub unity_processes: Vec<process::Child>,
    pub launched_ports: Vec<u16>,
    pub port_algorithm_map: std::collections::HashMap<u16, String>,
    pub tcp_server_handles: Vec<thread::JoinHandle<()>>,
    pub is_training_started: bool,
    pub shutdown_requested: Arc<AtomicBool>,
    pub dummy_launch_info: Option<(String, Vec<String>, HashMap<String, String>)>, // Path, Args, Envs
}

impl UnityLauncher {
    pub fn new() -> Self {
        Self {
            unity_processes: Vec::new(),
            launched_ports: Vec::new(),
            port_algorithm_map: std::collections::HashMap::new(),
            tcp_server_handles: Vec::new(),
            is_training_started: false,
            shutdown_requested: Arc::new(AtomicBool::new(false)),
            dummy_launch_info: None,
        }
    }

    pub fn launch_unity_environment(
        &mut self,
        env_path: &str,
        total_env: u16,
        base_port: u16,
        selected_algorithms_for_config: &[String],
        num_areas: u16,
        timeout_wait: u16,
        seed: i32,
        max_lifetime_restarts: u16,
        restarts_rate_limit_n: u16,
        restarts_rate_limit_period_s: u16,
        headless: bool,
        visual_progress: bool,
        engine_settings: &crate::pages::config_types::EngineSettings,
        run_id: &str,
        device: &str,
        algorithm_configs: &HashMap<String, AlgoConfig>,
        _checkpoint_interval: u32,
        _keep_checkpoints: u32,
        results_path: &str,
        shared_memory_path: &str, 
    ) -> Result<(), String> {
        let had_servers = !self.tcp_server_handles.is_empty();
        for handle in self.tcp_server_handles.drain(..) {
            drop(handle);
        }

        if had_servers {
            thread::sleep(std::time::Duration::from_millis(500));
        }

        self.launched_ports.clear();
        self.dummy_launch_info = None; // Reset dummy info

        for mut process in self.unity_processes.drain(..) {
            let _ = process.kill();
            let _ = process.wait();
        }

        let log_dir = std::path::Path::new(results_path).join("logs");
        let _ = std::fs::create_dir_all(&log_dir);

        if !env_path.is_empty() {
            use std::os::unix::fs::PermissionsExt;

            if !std::path::Path::new(env_path).exists() {
                return Err(format!("Unity path not found: {}", &env_path));
            }

            match std::fs::metadata(env_path) {
                Ok(metadata) => {
                    let permissions = metadata.permissions();
                    let mode = permissions.mode();

                    if metadata.is_file() {
                        if mode & 0o111 == 0 {
                            let mut new_permissions = permissions;
                            new_permissions.set_mode(mode | 0o111);
                            let _ = std::fs::set_permissions(env_path, new_permissions);
                        }
                    } else if metadata.is_dir() && env_path.ends_with(".app") {
                        let macos_dir = format!("{}/Contents/MacOS", env_path);
                        let mut executable_path: Option<String> = None;

                        if let Ok(entries) = std::fs::read_dir(&macos_dir) {
                            for entry in entries.flatten() {
                                if let Ok(file_type) = entry.file_type() {
                                    if file_type.is_file() && !entry.file_name().to_string_lossy().starts_with('.') {
                                        executable_path = Some(entry.path().to_str().unwrap().to_string());
                                        break;
                                    }
                                }
                            }
                        }

                        if let Some(exec_path) = executable_path {
                            return self.execute_unity_executable(
                                &exec_path, total_env, base_port, selected_algorithms_for_config,
                                num_areas, timeout_wait, seed, max_lifetime_restarts,
                                restarts_rate_limit_n, restarts_rate_limit_period_s,
                                headless, visual_progress, engine_settings, run_id, device,
                                algorithm_configs, _checkpoint_interval, _keep_checkpoints,
                                results_path, shared_memory_path,
                            );
                        }
                    }
                }
                Err(e) => return Err(e.to_string()),
            }

            return self.execute_unity_executable(
                env_path, total_env, base_port, selected_algorithms_for_config,
                num_areas, timeout_wait, seed, max_lifetime_restarts,
                restarts_rate_limit_n, restarts_rate_limit_period_s,
                headless, visual_progress, engine_settings, run_id, device,
                algorithm_configs, _checkpoint_interval, _keep_checkpoints,
                results_path, shared_memory_path,
            );
        }
        Ok(())
    }

    fn execute_unity_executable(
        &mut self,
        executable_path: &str,
        total_env: u16,
        base_port: u16,
        selected_algorithms_for_config: &[String],
        num_areas: u16,
        timeout_wait: u16,
        seed: i32,
        max_lifetime_restarts: u16,
        restarts_rate_limit_n: u16,
        restarts_rate_limit_period_s: u16,
        headless: bool,
        visual_progress: bool,
        engine_settings: &crate::pages::config_types::EngineSettings,
        run_id: &str,
        device: &str,
        algorithm_configs: &HashMap<String, AlgoConfig>,
        _checkpoint_interval: u32,
        _keep_checkpoints: u32,
        results_path: &str,
        shared_memory_path: &str,
    ) -> Result<(), String> {
        use std::os::unix::fs::PermissionsExt;

        let resolved_path = if executable_path.ends_with(".app") {
            let path = std::path::Path::new(executable_path);
            let file_stem = path.file_stem().unwrap().to_str().unwrap();
            let binary_path = path.join("Contents/MacOS").join(file_stem);
            if binary_path.exists() { binary_path.to_string_lossy().to_string() } else { executable_path.to_string() }
        } else {
            executable_path.to_string()
        };

        let exec_cmd_path = &resolved_path; 
        
        if let Ok(metadata) = std::fs::metadata(exec_cmd_path) {
            let mut permissions = metadata.permissions();
            if permissions.mode() & 0o111 == 0 {
                permissions.set_mode(0o755);
                let _ = std::fs::set_permissions(exec_cmd_path, permissions);
            }
        }

        let num_to_launch = if visual_progress { total_env + 1 } else { total_env };

        for i in 0..num_to_launch {
            let is_dummy = visual_progress && i == num_to_launch - 1;
            let current_port = base_port + (i as u16);
            let algorithm_name = if is_dummy {
                String::from("DUMMY")
            } else if !selected_algorithms_for_config.is_empty() {
                let algo_index = (i as usize) % selected_algorithms_for_config.len();
                selected_algorithms_for_config[algo_index].clone()
            } else {
                String::from("PPO")
            };

            let mut cmd = std::process::Command::new(exec_cmd_path);
            let mut envs = HashMap::new();
            
            // Collect Envs
            envs.insert("UNITY_BASE_PORT".to_string(), current_port.to_string());
            envs.insert("UNITY_ALGORITHM".to_string(), algorithm_name.clone());
            envs.insert("UNITY_SHARED_MEMORY_PATH".to_string(), shared_memory_path.to_string());

            let algo_config = if is_dummy {
                AlgoConfig::ppo()
            } else {
                algorithm_configs.get(&algorithm_name).cloned().unwrap_or_else(|| {
                    match algorithm_name.as_str() {
                        "PPO" => AlgoConfig::ppo(),
                        "SAC" => AlgoConfig::sac(),
                        "TD3" => AlgoConfig::td3(),
                        "TDSAC" => AlgoConfig::tdsac(),
                        "TQC" => AlgoConfig::tqc(),
                        "CrossQ" => AlgoConfig::crossq(),
                        "DrQV2" => AlgoConfig::drqv2(),
                        "PPO_ET" => AlgoConfig::ppo_et(),
                        _ => AlgoConfig::ppo(),
                    }
                })
            };
            
            envs.insert("UNITY_ALGORITHM_CONFIG".to_string(), format!("{:?}", algo_config));

            // Apply Envs
            for (k, v) in &envs {
                cmd.env(k, v);
            }

            // Collect Args
            let mut args = Vec::new();
            args.push(format!("--base-port={}", current_port));
            args.push(format!("--algorithm={}", algorithm_name));
            args.push(format!("--num-areas={}", num_areas));
            args.push(format!("--timeout-wait={}", timeout_wait));
            args.push(format!("--seed={}", seed));
            args.push(format!("--max-lifetime-restarts={}", max_lifetime_restarts));
            args.push(format!("--restarts-rate-limit-n={}", restarts_rate_limit_n));
            args.push(format!("--restarts-rate-limit-period-s={}", restarts_rate_limit_period_s));

            if is_dummy {
                args.push("-screen-width".to_string()); args.push("800".to_string());
                args.push("-screen-height".to_string()); args.push("600".to_string());
                args.push("-screen-fullscreen".to_string()); args.push("0".to_string());
            } else {
                if headless { args.push("--headless".to_string()); }
                if engine_settings.no_graphics { args.push("--no-graphics".to_string()); }
                args.push(format!("--width={}", engine_settings.width));
                args.push(format!("--height={}", engine_settings.height));
            }

            args.push(format!("--quality-level={}", engine_settings.quality_level));
            args.push(format!("-speed={}", if is_dummy { 1.0 } else { engine_settings.time_scale }));
            args.push(format!("--time-scale={}", if is_dummy { 1.0 } else { engine_settings.time_scale }));
            args.push(format!("--target-framerate={}", engine_settings.target_frame_rate));
            args.push(format!("--capture-frame-rate={}", engine_settings.capture_frame_rate));

            if !run_id.is_empty() { args.push(format!("--run-id={}", run_id)); }

            let log_file_name = if is_dummy { "player-dummy.log".to_string() } else { format!("player-{}.log", i + 1) };
            let log_file_path = std::path::Path::new(results_path).join("logs").join(log_file_name);
            args.push("-logFile".to_string()); 
            args.push(log_file_path.to_string_lossy().to_string());
            args.push(format!("--device={}", device));

            // Apply Args
            for arg in &args {
                cmd.arg(arg);
            }

            // Save dummy info
            if is_dummy {
                self.dummy_launch_info = Some((exec_cmd_path.to_string(), args.clone(), envs.clone()));
            }

            cmd.stdout(Stdio::piped()).stderr(Stdio::piped());

            match cmd.spawn() {
                Ok(mut child) => {
                    if let Some(stdout) = child.stdout.take() {
                        thread::spawn(move || { let reader = BufReader::new(stdout); for _ in reader.lines() {} });
                    }
                    if let Some(stderr) = child.stderr.take() {
                        thread::spawn(move || { let reader = BufReader::new(stderr); for _ in reader.lines() {} });
                    }
                    self.unity_processes.push(child);
                    self.launched_ports.push(current_port);
                    self.port_algorithm_map.insert(current_port, algorithm_name.clone());
                }
                Err(_) => {}
            }
        }
        Ok(())
    }

    pub fn relaunch_dummy(&mut self) -> Result<(), String> {
        if let Some((path, args, envs)) = &self.dummy_launch_info {
            println!("🔄 Relaunching Dummy Instance...");
            let mut cmd = std::process::Command::new(path);
            
            for (k, v) in envs {
                cmd.env(k, v);
            }
            for arg in args {
                cmd.arg(arg);
            }
            
            cmd.stdout(Stdio::piped()).stderr(Stdio::piped());

            match cmd.spawn() {
                Ok(mut child) => {
                    if let Some(stdout) = child.stdout.take() {
                        thread::spawn(move || { let reader = BufReader::new(stdout); for _ in reader.lines() {} });
                    }
                    if let Some(stderr) = child.stderr.take() {
                        thread::spawn(move || { let reader = BufReader::new(stderr); for _ in reader.lines() {} });
                    }
                    self.unity_processes.push(child);
                    // Note: We don't add to launched_ports or map again as it should be the same port/config
                    Ok(())
                }
                Err(e) => Err(format!("Failed to relaunch dummy: {}", e))
            }
        } else {
            Err("No dummy launch info available.".to_string())
        }
    }

    pub fn stop_training(&mut self) {
        self.shutdown_requested.store(true, Ordering::SeqCst);
        self.tcp_server_handles.clear();
        for mut process in self.unity_processes.drain(..) {
            let _ = process.kill();
            let _ = process.wait();
        }
        self.launched_ports.clear();
        self.port_algorithm_map.clear();
        self.is_training_started = false;
        self.dummy_launch_info = None;
    }
}