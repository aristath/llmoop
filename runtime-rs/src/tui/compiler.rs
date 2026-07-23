use std::collections::BTreeMap;
use std::env;
use std::ffi::OsString;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, ExitStatus, Stdio};
use std::sync::mpsc::{self, Receiver};
use std::thread;

use serde::{Deserialize, Serialize};
use serde_json::Value;

const COMPILER_EVENT_SCHEMA: &str = "nerve.compiler_event.v1";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CompilerJobKind {
    Discovery,
    Compilation,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CompilerEvent {
    pub schema: String,
    pub sequence: u64,
    #[serde(rename = "type")]
    pub kind: String,
    #[serde(flatten)]
    pub payload: BTreeMap<String, Value>,
}

impl CompilerEvent {
    pub fn value(&self, key: &str) -> Option<&Value> {
        self.payload.get(key)
    }

    pub fn nested_string(&self, object: &str, key: &str) -> Option<&str> {
        self.value(object)?.get(key)?.as_str()
    }

    pub fn progress(&self) -> Option<(u64, u64)> {
        Some((
            self.value("current")?.as_u64()?,
            self.value("total")?.as_u64()?,
        ))
    }

    pub fn current_item(&self) -> Option<&str> {
        ["component_id", "component_id", "tensor_name", "shader_name"]
            .into_iter()
            .find_map(|key| self.value(key).and_then(Value::as_str))
    }

    pub fn diagnostics(&self) -> Vec<String> {
        self.value("diagnostics")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(|diagnostic| {
                diagnostic
                    .get("message")
                    .and_then(Value::as_str)
                    .map(ToOwned::to_owned)
            })
            .collect()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CompilerLaunch {
    program: OsString,
    prefix_args: Vec<OsString>,
    working_directory: PathBuf,
}

impl CompilerLaunch {
    pub fn from_environment() -> Result<Self, String> {
        let working_directory = compiler_workspace()?;
        if let Some(program) = env::var_os("NERVE_COMPILER_BIN") {
            return Ok(Self {
                program,
                prefix_args: Vec::new(),
                working_directory,
            });
        }
        let python = env::var_os("NERVE_PYTHON").unwrap_or_else(|| {
            let local = working_directory.join(".venv/bin/python");
            if local.is_file() {
                local.into_os_string()
            } else {
                OsString::from("python3")
            }
        });
        Ok(Self {
            program: python,
            prefix_args: vec![OsString::from("-m"), OsString::from("nerve")],
            working_directory,
        })
    }

    pub fn new(
        program: impl Into<OsString>,
        prefix_args: impl IntoIterator<Item = OsString>,
        working_directory: impl Into<PathBuf>,
    ) -> Self {
        Self {
            program: program.into(),
            prefix_args: prefix_args.into_iter().collect(),
            working_directory: working_directory.into(),
        }
    }

    pub fn working_directory(&self) -> &Path {
        &self.working_directory
    }

    pub fn start_discovery(&self, model_dir: &Path) -> Result<CompilerJob, String> {
        self.spawn(
            CompilerJobKind::Discovery,
            [
                OsString::from("--discover-model"),
                model_dir.as_os_str().to_owned(),
            ],
        )
    }

    pub fn start_compilation(&self, model_dir: &Path) -> Result<CompilerJob, String> {
        self.spawn(
            CompilerJobKind::Compilation,
            [
                OsString::from("--compile-model"),
                model_dir.as_os_str().to_owned(),
            ],
        )
    }

    fn spawn(
        &self,
        kind: CompilerJobKind,
        args: impl IntoIterator<Item = OsString>,
    ) -> Result<CompilerJob, String> {
        let mut command = Command::new(&self.program);
        command
            .args(&self.prefix_args)
            .args(args)
            .arg("--compiler-events-jsonl")
            .current_dir(&self.working_directory)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        let mut child = command.spawn().map_err(|error| {
            format!(
                "could not start compiler {:?}: {error}",
                self.program.to_string_lossy()
            )
        })?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| "compiler stdout pipe was not created".to_string())?;
        let stderr = child
            .stderr
            .take()
            .ok_or_else(|| "compiler stderr pipe was not created".to_string())?;
        let (sender, receiver) = mpsc::channel();
        let stdout_sender = sender.clone();
        thread::spawn(move || {
            for line in BufReader::new(stdout).lines() {
                let message = match line {
                    Ok(line) => match serde_json::from_str::<CompilerEvent>(&line) {
                        Ok(event) if event.schema == COMPILER_EVENT_SCHEMA => {
                            CompilerMessage::Event(event)
                        }
                        Ok(event) => CompilerMessage::ProtocolError(format!(
                            "unsupported compiler event schema {:?}",
                            event.schema
                        )),
                        Err(error) => CompilerMessage::ProtocolError(format!(
                            "invalid compiler event: {error}; output was {line:?}"
                        )),
                    },
                    Err(error) => CompilerMessage::ProtocolError(format!(
                        "could not read compiler event stream: {error}"
                    )),
                };
                if stdout_sender.send(message).is_err() {
                    break;
                }
            }
        });
        thread::spawn(move || {
            for line in BufReader::new(stderr).lines() {
                match line {
                    Ok(line) if !line.trim().is_empty() => {
                        if sender.send(CompilerMessage::Diagnostic(line)).is_err() {
                            break;
                        }
                    }
                    Ok(_) => {}
                    Err(error) => {
                        let _ = sender.send(CompilerMessage::Diagnostic(format!(
                            "could not read compiler diagnostics: {error}"
                        )));
                        break;
                    }
                }
            }
        });
        Ok(CompilerJob {
            kind,
            child,
            receiver,
            cancel_requested: false,
            terminal_event_received: false,
        })
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum CompilerMessage {
    Event(CompilerEvent),
    Diagnostic(String),
    ProtocolError(String),
}

pub struct CompilerJob {
    kind: CompilerJobKind,
    child: Child,
    receiver: Receiver<CompilerMessage>,
    cancel_requested: bool,
    terminal_event_received: bool,
}

impl CompilerJob {
    pub fn kind(&self) -> CompilerJobKind {
        self.kind
    }

    pub fn cancel_requested(&self) -> bool {
        self.cancel_requested
    }

    pub fn drain_messages(&mut self) -> Vec<CompilerMessage> {
        let mut messages = Vec::new();
        while let Ok(message) = self.receiver.try_recv() {
            if let CompilerMessage::Event(event) = &message
                && matches!(event.kind.as_str(), "Completed" | "Failed" | "Cancelled")
            {
                self.terminal_event_received = true;
            }
            messages.push(message);
        }
        messages
    }

    pub fn try_status(&mut self) -> Result<Option<ExitStatus>, String> {
        self.child
            .try_wait()
            .map_err(|error| format!("could not inspect compiler process: {error}"))
    }

    pub fn terminal_event_received(&self) -> bool {
        self.terminal_event_received
    }

    pub fn cancel(&mut self) -> Result<(), String> {
        if self.cancel_requested {
            return Ok(());
        }
        self.cancel_requested = true;
        send_terminate(self.child.id()).or_else(|_| {
            self.child
                .kill()
                .map_err(|error| format!("could not stop compiler process: {error}"))
        })
    }
}

impl Drop for CompilerJob {
    fn drop(&mut self) {
        if self.child.try_wait().ok().flatten().is_none() {
            let _ = send_terminate(self.child.id());
            let _ = self.child.wait();
        }
    }
}

#[cfg(unix)]
fn send_terminate(pid: u32) -> Result<(), String> {
    let status = Command::new("kill")
        .arg("-TERM")
        .arg(pid.to_string())
        .status()
        .map_err(|error| format!("could not request compiler cancellation: {error}"))?;
    if status.success() {
        Ok(())
    } else {
        Err(format!(
            "compiler cancellation command exited with {status}"
        ))
    }
}

#[cfg(not(unix))]
fn send_terminate(_pid: u32) -> Result<(), String> {
    Err("graceful compiler cancellation is not available on this platform".to_string())
}

fn compiler_workspace() -> Result<PathBuf, String> {
    if let Some(path) = env::var_os("NERVE_WORKSPACE") {
        let path = PathBuf::from(path);
        if is_compiler_workspace(&path) {
            return Ok(path);
        }
        return Err(format!(
            "NERVE_WORKSPACE does not contain the NERVE compiler: {}",
            path.display()
        ));
    }
    if let Ok(current) = env::current_dir() {
        for ancestor in current.ancestors() {
            if is_compiler_workspace(ancestor) {
                return Ok(ancestor.to_path_buf());
            }
        }
    }
    let source_workspace = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .map(Path::to_path_buf);
    if let Some(path) = source_workspace.filter(|path| is_compiler_workspace(path)) {
        return Ok(path);
    }
    Err("could not locate the NERVE compiler workspace; set NERVE_WORKSPACE".to_string())
}

fn is_compiler_workspace(path: &Path) -> bool {
    path.join("nerve/__main__.py").is_file() && path.join("nerve/cli.py").is_file()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::time::{Duration, Instant};

    #[test]
    fn compiler_event_exposes_progress_and_nested_package_path() {
        let event: CompilerEvent = serde_json::from_str(
            r#"{"schema":"nerve.compiler_event.v1","sequence":7,"type":"ComponentLoweringStarted","current":3,"total":12,"component_id":"layer_02"}"#,
        )
        .unwrap();
        assert_eq!(event.progress(), Some((3, 12)));
        assert_eq!(event.current_item(), Some("layer_02"));

        let completed: CompilerEvent = serde_json::from_str(
            r#"{"schema":"nerve.compiler_event.v1","sequence":8,"type":"Completed","package":{"package_manifest":"/tmp/package/vulkan_resident_package.json"}}"#,
        )
        .unwrap();
        assert_eq!(
            completed.nested_string("package", "package_manifest"),
            Some("/tmp/package/vulkan_resident_package.json")
        );
    }

    #[test]
    fn source_checkout_is_a_compiler_workspace() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
        assert!(is_compiler_workspace(root));
    }

    #[test]
    fn rust_client_consumes_real_python_discovery_protocol() {
        let source = env::temp_dir().join(format!(
            "nerve-tui-discovery-{}-{}",
            std::process::id(),
            Instant::now().elapsed().as_nanos()
        ));
        let _ = fs::remove_dir_all(&source);
        fs::create_dir_all(&source).unwrap();
        fs::write(
            source.join("config.json"),
            r#"{"model_type":"protocol_fixture","architectures":["FixtureCircuit"]}"#,
        )
        .unwrap();
        fs::write(source.join("model.safetensors"), b"fixture").unwrap();
        fs::write(source.join("tokenizer.json"), "{}").unwrap();
        fs::write(
            source.join("tokenizer_config.json"),
            r#"{"chat_template":"{{ messages }}"}"#,
        )
        .unwrap();

        let mut job = CompilerLaunch::from_environment()
            .unwrap()
            .start_discovery(&source)
            .unwrap();
        let deadline = Instant::now() + Duration::from_secs(10);
        let mut events = Vec::new();
        let status = loop {
            for message in job.drain_messages() {
                if let CompilerMessage::Event(event) = message {
                    events.push(event);
                }
            }
            if let Some(status) = job.try_status().unwrap() {
                break status;
            }
            assert!(Instant::now() < deadline, "compiler discovery timed out");
            thread::sleep(Duration::from_millis(10));
        };
        for message in job.drain_messages() {
            if let CompilerMessage::Event(event) = message {
                events.push(event);
            }
        }

        assert!(status.success());
        assert_eq!(
            events
                .iter()
                .map(|event| event.kind.as_str())
                .collect::<Vec<_>>(),
            ["DiscoveryStarted", "SourceDiscovered", "Completed"]
        );
        assert_eq!(
            events[1].nested_string("source", "model_type"),
            Some("protocol_fixture")
        );
        fs::remove_dir_all(source).unwrap();
    }
}
