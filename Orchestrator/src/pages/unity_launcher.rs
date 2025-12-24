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
    ) -> Result<(), String> {
        // First, terminate previous TCP servers if they exist
        let had_servers = !self.tcp_server_handles.is_empty();
        for handle in self.tcp_server_handles.drain(..) {
            println!("Terminating previous TCP server...");
            drop(handle);
        }

        // Wait a little for the OS to release ports
        if had_servers {
            thread::sleep(std::time::Duration::from_millis(500));
        }

        // Clear launched ports list
        self.launched_ports.clear();

        // Check if Unity processes are already running
        for mut process in self.unity_processes.drain(..) {
            // Try to terminate each existing process
            let _ = process.kill();
            let _ = process.wait();
        }

        // Create logs directory
        let log_dir = std::path::Path::new(results_path).join("logs");
        if let Err(e) = std::fs::create_dir_all(&log_dir) {
            eprintln!("Failed to create log directory: {}", e);
        }

        // Launch Unity environment
        if !env_path.is_empty() {

            use std::os::unix::fs::PermissionsExt;

            println!("Attempting to launch Unity: {}", &env_path);

            // Check if path exists
            if !std::path::Path::new(env_path).exists() {
                let msg = format!("Unity path not found: {}", &env_path);
                eprintln!("{}", msg);
                return Err(msg);
            }

            // Check and adjust file or directory permissions
            match std::fs::metadata(env_path) {
                Ok(metadata) => {
                    let permissions = metadata.permissions();
                    let mode = permissions.mode();
                    println!("Current Unity path permissions: {:o}", mode);

                    // Check file/directory type
                    if metadata.is_file() {
                        println!("Path is a regular file");

                        // For regular files, check and adjust permissions
                        if mode & 0o111 == 0 {
                            // Add execution permission (equivalent to chmod +x)
                            let mut new_permissions = permissions;
                            new_permissions.set_mode(mode | 0o111);
                            match std::fs::set_permissions(env_path, new_permissions) {
                                Ok(_) => {
                                    println!("Execution permissions set for: {}", &env_path);
                                }
                                Err(e) => {
                                    eprintln!("Error setting Unity file permissions: {}", e);
                                    return Err(format!("Error setting Unity file permissions: {}", e)); // Exit if unable to set permissions
                                }
                            }
                        } else {
                            println!("Unity file already has execution permissions");
                        }
                    } else if metadata.is_dir() {
                        // Check if it's a macOS application
                        if env_path.ends_with(".app") {
                            println!("Path is a macOS application bundle");

                            let macos_dir = format!("{}/Contents/MacOS", env_path);
                            let mut executable_path: Option<String> = None;

                            if let Ok(entries) = std::fs::read_dir(&macos_dir) {
                                for entry in entries.flatten() {
                                    if let Ok(file_type) = entry.file_type() {
                                        if file_type.is_file() {
                                            if let Some(path) = entry.path().to_str() {
                                                // Make sure we're not picking up .DS_Store or other hidden files
                                                if !entry.file_name().to_string_lossy().starts_with('.') {
                                                    executable_path = Some(path.to_string());
                                                    break; // Found it
                                                }
                                            }
                                        }
                                    }
                                }
                            }

                            if let Some(exec_path) = executable_path {
                                println!("Using found executable: {}", exec_path);
                                self.execute_unity_executable(
                                    &exec_path,
                                    total_env,
                                    base_port,
                                    selected_algorithms_for_config,
                                    num_areas,
                                    timeout_wait,
                                    seed,
                                    max_lifetime_restarts,
                                    restarts_rate_limit_n,
                                    restarts_rate_limit_period_s,
                                    headless,
                                    visual_progress,
                                    engine_settings,
                                    run_id,
                                    device,
                                    algorithm_configs,
                                    _checkpoint_interval,
                                    _keep_checkpoints,
                                    results_path
                                )?; // Propagate error
                            } else {
                                let msg = format!("Executable not found inside {}", macos_dir);
                                eprintln!("{}", msg);
                                return Err(msg);
                            }
                        } else {
                            let msg = format!("Error: The specified path is a directory, not an executable file: {}", &env_path);
                            eprintln!("{}", msg);
                            return Err(msg);
                        }

                        return Ok(()); // Return after handling .app
                    } else {
                        println!("Path is another type of file");
                        return Err(format!("Path is not a recognized executable or .app bundle: {}", env_path));
                    }
                }
                Err(e) => {
                    eprintln!("Error getting Unity path metadata: {}", e);
                    return Err(format!("Error getting Unity path metadata: {}", e));
                }
            }

            println!("Launching {} instances of Unity environment", total_env);

            for i in 0..total_env {
                let current_port = base_port + (i as u16);

                // Distribute algorithms: each instance uses a different algorithm (round-robin)
                let algorithm_name = if !selected_algorithms_for_config.is_empty() {
                    let algo_index = (i as usize) % selected_algorithms_for_config.len();
                    selected_algorithms_for_config[algo_index].clone()
                } else {
                    String::from("PPO")
                };

                println!("=== LAUNCHING INSTANCE {} ===", i + 1);
                println!("Calculated port: base_port({}) + i({}) = {}", base_port, i, current_port);
                println!("Algorithm: {}", algorithm_name);

                let mut cmd = std::process::Command::new(env_path);

                // Set environment variables
                cmd.env("UNITY_BASE_PORT", current_port.to_string());
                cmd.env("UNITY_ALGORITHM", algorithm_name.clone());

                // Get configured hyperparameters for the algorithm
                let algo_config = algorithm_configs.get(&algorithm_name).cloned().unwrap_or_else(|| {
                    match algorithm_name.as_str() {
                        "PPO" => AlgoConfig::ppo(),
                        "SAC" => AlgoConfig::sac(),
                        "TD3" => AlgoConfig::td3(),
                        "TDSAc" => AlgoConfig::tdsac(),
                        "TQC" => AlgoConfig::tqc(),
                        "CrossQ" => AlgoConfig::crossq(),
                        "DrQV2" => AlgoConfig::drqv2(),
                        "PPO_ET" => AlgoConfig::ppo_et(),
                        "PPO_CE" => AlgoConfig::ppo_ce(),
                        _ => AlgoConfig::ppo(), // default
                    }
                });

                // Update with global default values if not explicitly configured
                // This ensures default values defined in Settings section are applied
                if !algorithm_configs.contains_key(&algorithm_name) {
                }

                // Pass hyperparameters as JSON environment variable (this may need proper serialization)
                cmd.env("UNITY_ALGORITHM_CONFIG", format!("{:?}", algo_config));

                cmd.arg(format!("--base-port={}", current_port));
                println!("Argument --base-port={}", current_port);
                println!("Environment variables: UNITY_BASE_PORT={}, UNITY_ALGORITHM={}, UNITY_ALGORITHM_CONFIG=...", current_port, algorithm_name);
                cmd
                   .arg(format!("--algorithm={}", algorithm_name))
                   .arg(format!("--num-areas={}", num_areas))
                   .arg(format!("--timeout-wait={}", timeout_wait))
                   .arg(format!("--seed={}", seed))
                   .arg(format!("--max-lifetime-restarts={}", max_lifetime_restarts))
                   .arg(format!("--restarts-rate-limit-n={}", restarts_rate_limit_n))
                   .arg(format!("--restarts-rate-limit-period-s={}", restarts_rate_limit_period_s));

                // Screen parameters: use native Unity in visual mode, custom in headless mode
                if !headless && !engine_settings.no_graphics {
                    cmd.arg("-screen-width")
                       .arg(format!("{}", engine_settings.width))
                       .arg("-screen-height")
                       .arg(format!("{}", engine_settings.height))
                       .arg("-screen-fullscreen")
                       .arg("0");
                } else {
                    // In headless mode, use custom parameters
                    cmd.arg(format!("--width={}", engine_settings.width))
                       .arg(format!("--height={}", engine_settings.height));
                }

                cmd.arg(format!("--quality-level={}", engine_settings.quality_level))
                   .arg(format!("--time-scale={}", engine_settings.time_scale))
                   .arg(format!("--target-framerate={}", engine_settings.target_frame_rate))
                   .arg(format!("--capture-frame-rate={}", engine_settings.capture_frame_rate));

                if headless {
                    cmd.arg("--headless");
                }

                if visual_progress {
                    cmd.arg("--visual-progress");
                }

                if engine_settings.no_graphics {
                    cmd.arg("--no-graphics");
                }

                if !run_id.is_empty() {
                    cmd.arg(format!("--run-id={}", run_id));
                }

                // Add player log with environment number
                let log_file_path = std::path::Path::new(results_path).join("logs").join(format!("player-{}.log", i + 1));
                println!("Log file path: {:?}", log_file_path);
                
                cmd.arg("-logFile")
                   .arg(log_file_path)
                   .arg(format!("--device={}", device));

                // Configure piping for log filtering
                cmd.stdout(Stdio::piped())
                   .stderr(Stdio::piped());

                match cmd.spawn() {
                        Ok(mut child) => {
                            // Spawn threads to filter stdout/stderr
                            if let Some(stdout) = child.stdout.take() {
                                thread::spawn(move || {
                                    let reader = BufReader::new(stdout);
                                    for line in reader.lines() {
                                        if let Ok(l) = line {
                                            // Filter out UnityMemory logs
                                            if !l.contains("[UnityMemory]") && !l.contains("memorysetup-") {
                                                println!("{}", l);
                                            }
                                        }
                                    }
                                });
                            }

                            if let Some(stderr) = child.stderr.take() {
                                thread::spawn(move || {
                                    let reader = BufReader::new(stderr);
                                    for line in reader.lines() {
                                        if let Ok(l) = line {
                                            if !l.contains("[UnityMemory]") && !l.contains("memorysetup-") {
                                                eprintln!("{}", l);
                                            }
                                        }
                                    }
                                });
                            }

                            self.unity_processes.push(child);
                            self.launched_ports.push(current_port);
                            self.port_algorithm_map.insert(current_port, algorithm_name.clone());
                            println!("Unity environment instance {} launched successfully on port {}", i + 1, current_port);
                        }
                        Err(e) => {
                            eprintln!("Error launching Unity environment instance {}: {}", i + 1, e);
                            eprintln!("Details: The file path may be incorrect or the file may have additional security restrictions.");
                        }
                    }
            }

            // Start TCP servers for each launched port
            if !self.unity_processes.is_empty() {
                println!("Total of {} Unity instance(s) launched successfully", self.unity_processes.len());
            } else {
                eprintln!("No Unity instances were launched successfully");
            }
        } else {
            eprintln!("Unity environment path not specified");
            return Err("Unity environment path not specified".to_string());
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
    ) -> Result<(), String> {
        use std::os::unix::fs::PermissionsExt;

        // Check and adjust permissions of the actual executable
        match std::fs::metadata(executable_path) {
            Ok(exec_metadata) => {
                let exec_permissions = exec_metadata.permissions();
                let exec_mode = exec_permissions.mode();
                println!("Real executable permissions: {:o}", exec_mode);

                if exec_mode & 0o111 == 0 {
                    let mut new_exec_permissions = exec_metadata.permissions();
                    new_exec_permissions.set_mode(exec_mode | 0o111);
                    match std::fs::set_permissions(executable_path, new_exec_permissions) {
                        Ok(_) => {
                            println!("Execution permissions set for executable: {}", executable_path);
                        }
                        Err(e) => {
                            eprintln!("Error setting Unity executable permissions: {}", e);
                            return Err(format!("Error setting Unity executable permissions: {}", e));
                        }
                    }
                } else {
                    println!("Unity executable already has execution permissions");
                }
            }
            Err(e) => {
                eprintln!("Error getting Unity executable metadata: {}", e);
                return Err(format!("Error getting Unity executable metadata: {}", e));
            }
        }

        // Execute multiple instances of the executable based on total_env
        println!("Launching {} instances of Unity environment", total_env);

        // Resolve executable path (Handle macOS .app bundle)
        let resolved_path = if executable_path.ends_with(".app") {
            let path = std::path::Path::new(executable_path);
            if let Some(file_stem) = path.file_stem() {
                if let Some(stem_str) = file_stem.to_str() {
                    let binary_path = path.join("Contents/MacOS").join(stem_str);
                    if binary_path.exists() {
                        binary_path.to_string_lossy().to_string()
                    } else {
                        eprintln!("⚠️ Warning: Expected binary not found at {:?}. Trying to find *any* executable in Contents/MacOS...", binary_path);
                        executable_path.to_string()
                    }
                } else {
                    executable_path.to_string()
                }
            } else {
                executable_path.to_string()
            }
        } else {
            executable_path.to_string()
        };

        let exec_cmd_path = &resolved_path; 
        
        // Ensure executable permissions
        if let Ok(metadata) = std::fs::metadata(exec_cmd_path) {
            let mut permissions = metadata.permissions();
            if permissions.mode() & 0o111 == 0 {
                println!("🔧 Fix: Adding +x permission to {}", exec_cmd_path);
                permissions.set_mode(0o755);
                if let Err(e) = std::fs::set_permissions(exec_cmd_path, permissions) {
                     eprintln!("❌ Failed to set permissions for {}: {}", exec_cmd_path, e);
                }
            }
        } else {
             eprintln!("❌ Error: Executable file not found at {}", exec_cmd_path);
        }

        if let Ok(cwd) = std::env::current_dir() {
            println!("📂 Current Working Directory: {:?}", cwd);
        } else {
            eprintln!("⚠️ Could not determine CWD.");
        }

        // Create instances of Unity environment
        let num_to_launch = if visual_progress { total_env + 1 } else { total_env };
        println!("Launching {} instances of Unity environment (Visual Progress: {})", num_to_launch, visual_progress);

        for i in 0..num_to_launch {
            let is_dummy = visual_progress && i == num_to_launch - 1;
            let current_port = base_port + (i as u16);

            // Distribute algorithms
            let algorithm_name = if is_dummy {
                String::from("DUMMY")
            } else if !selected_algorithms_for_config.is_empty() {
                let algo_index = (i as usize) % selected_algorithms_for_config.len();
                selected_algorithms_for_config[algo_index].clone()
            } else {
                String::from("PPO")
            };

            println!("=== LAUNCHING INSTANCE {} {} ===", i + 1, if is_dummy { "[DUMMY]" } else { "" });
            
            let mut cmd = std::process::Command::new(exec_cmd_path);
            
            // Set unique environment variables
            cmd.env("UNITY_BASE_PORT", current_port.to_string());
            cmd.env("UNITY_ALGORITHM", algorithm_name.clone());

            let algo_config = if is_dummy {
                AlgoConfig::ppo()
            } else {
                algorithm_configs.get(&algorithm_name).cloned().unwrap_or_else(|| {
                    match algorithm_name.as_str() {
                        "PPO" => AlgoConfig::ppo(),
                        "SAC" => AlgoConfig::sac(),
                        "TD3" => AlgoConfig::td3(),
                        "POCA" => AlgoConfig::poca(),
                        _ => AlgoConfig::ppo(),
                    }
                })
            };

            cmd.env("UNITY_ALGORITHM_CONFIG", format!("{:?}", algo_config));

            cmd.arg(format!("--base-port={}", current_port))
               .arg(format!("--algorithm={}", algorithm_name))
               .arg(format!("--num-areas={}", num_areas))
               .arg(format!("--timeout-wait={}", timeout_wait))
               .arg(format!("--seed={}", seed))
               .arg(format!("--max-lifetime-restarts={}", max_lifetime_restarts))
               .arg(format!("--restarts-rate-limit-n={}", restarts_rate_limit_n))
               .arg(format!("--restarts-rate-limit-period-s={}", restarts_rate_limit_period_s));

            if is_dummy {
                cmd.arg("-screen-width").arg("800")
                   .arg("-screen-height").arg("600")
                   .arg("-screen-fullscreen").arg("0");
            } else {
                if headless { cmd.arg("--headless"); }
                if engine_settings.no_graphics { cmd.arg("--no-graphics"); }
                
                cmd.arg(format!("--width={}", engine_settings.width))
                   .arg(format!("--height={}", engine_settings.height));
            }

            cmd.arg(format!("--quality-level={}", engine_settings.quality_level))
               .arg(format!("-speed={}", if is_dummy { 1.0 } else { engine_settings.time_scale }))
               .arg(format!("--time-scale={}", if is_dummy { 1.0 } else { engine_settings.time_scale }))
               .arg(format!("--target-framerate={}", engine_settings.target_frame_rate))
               .arg(format!("--capture-frame-rate={}", engine_settings.capture_frame_rate));

            if !run_id.is_empty() {
                cmd.arg(format!("--run-id={}", run_id));
            }

            let log_file_name = if is_dummy { "player-dummy.log".to_string() } else { format!("player-{}.log", i + 1) };
            let log_file_path = std::path::Path::new(results_path).join("logs").join(log_file_name);
            cmd.arg("-logFile").arg(log_file_path).arg(format!("--device={}", device));

            cmd.stdout(Stdio::piped()).stderr(Stdio::piped());

            match cmd.spawn() {
                Ok(mut child) => {
                    if let Some(stdout) = child.stdout.take() {
                        thread::spawn(move || {
                            let reader = BufReader::new(stdout);
                            for _ in reader.lines() {}
                        });
                    }
                    if let Some(stderr) = child.stderr.take() {
                        thread::spawn(move || {
                            let reader = BufReader::new(stderr);
                            for _ in reader.lines() {}
                        });
                    }

                    self.unity_processes.push(child);
                    self.launched_ports.push(current_port);
                    self.port_algorithm_map.insert(current_port, algorithm_name.clone());
                }
                Err(e) => eprintln!("Error launching instance: {}", e),
            }
        }
        Ok(())
    }

    pub fn stop_training(&mut self) {
        self.shutdown_requested.store(true, Ordering::SeqCst);
        self.tcp_server_handles.clear();
        for (i, mut process) in self.unity_processes.drain(..) .enumerate() {
            let _ = process.kill();
            let _ = process.wait();
        }
        self.launched_ports.clear();
        self.port_algorithm_map.clear();
        self.is_training_started = false;
    }
}
