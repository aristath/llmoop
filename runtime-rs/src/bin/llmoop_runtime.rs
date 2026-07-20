use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::time::Instant;

use chrono::{DateTime, FixedOffset, Local};
use llmoop_runtime::{
    CircuitPort, PedalCablePlacement, PedalPlacement, RUNTIME_DEFAULT_LOGICAL_DEVICE_ID,
    RUNTIME_TOPOLOGY_SCHEMA, RuntimeAvailableDevice, RuntimeBoundDevice, RuntimeCableRouteTarget,
    RuntimeCableRoutes, RuntimeCompiledPedalboardSummary, RuntimeDeviceBindings,
    RuntimeDeviceSliceReport, RuntimeDeviceTickPlanReport, RuntimeEffectivePedalboardTopology,
    RuntimeLocalCableBufferReport, RuntimePackageInspectionReport, RuntimePatchControls,
    RuntimePatchDuplicateAfterControl, RuntimePatchInspectionReport, RuntimePatchPlacementReport,
    RuntimePatchSourceChainEntry, RuntimePedalPortSummary, RuntimePlacedPedalTimingSummaryReport,
    RuntimePlacedPromptRunReport, RuntimePlacedTransportReport, RuntimePlacementReport,
    RuntimePromptBenchmarkReport, RuntimePromptBenchmarkRunReport,
    RuntimePromptBenchmarkTransportTotalsReport, RuntimePromptBenchmarkU64MetricReport,
    RuntimePromptBenchmarkUsizeMetricReport, RuntimePromptTimingReport,
    RuntimeRemoteCableBufferReport, RuntimeSourcePedal, RuntimeTokenizerOptionsReport,
    RuntimeTopologyReport, VulkanComputeDevice, VulkanComputeDeviceCatalog,
    VulkanComputeDeviceInfo, VulkanResidentHfTokenizerTextCodec,
    VulkanResidentInProcessPlacedPromptEngine, VulkanResidentInProcessPlacedPromptStream,
    VulkanResidentModelPackageDeviceSlice, VulkanResidentModelPackageManifest,
    VulkanResidentRuntimeModel, VulkanResidentTokenInputEvent, VulkanResidentTokenTextCodec,
    VulkanReusableKernelArtifactManifest, discover_runtime_devices,
};
use minijinja::{Environment, Error as TemplateError, ErrorKind as TemplateErrorKind};
use serde::Serialize;

#[derive(Clone, Debug, PartialEq, Eq)]
struct Args {
    package_manifest: Option<PathBuf>,
    prompt: Option<String>,
    chat: bool,
    inspect_runtime: bool,
    inspect_package: bool,
    inspect_patch: bool,
    inspect_placement: bool,
    inspect_device_slice: Option<String>,
    default_device_id: Option<String>,
    pedal_devices: BTreeMap<String, String>,
    device_bindings: BTreeMap<String, String>,
    duplicate_after: Vec<(String, String)>,
    source_chain: Option<Vec<(String, String)>>,
    max_new_tokens: usize,
    speculative_draft_tokens: usize,
    context_size: Option<usize>,
    vulkan_device_index: Option<usize>,
    random_seed: u32,
    add_special_tokens: bool,
    skip_special_tokens: bool,
    generated_only: bool,
    profile: bool,
    profile_runs: usize,
    json: bool,
}

struct PromptRunContext<'a> {
    args: &'a Args,
    package_manifest: &'a Path,
    manifest_dir: &'a Path,
    tokenizer_dir: &'a Path,
    prompt: &'a str,
    prompt_ids: &'a [u32],
    scheduled_token_activations: usize,
    capacity: usize,
    codec: &'a VulkanResidentHfTokenizerTextCodec,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            package_manifest: None,
            prompt: None,
            chat: false,
            inspect_runtime: false,
            inspect_package: false,
            inspect_patch: false,
            inspect_placement: false,
            inspect_device_slice: None,
            default_device_id: None,
            pedal_devices: BTreeMap::new(),
            device_bindings: BTreeMap::new(),
            duplicate_after: Vec::new(),
            source_chain: None,
            max_new_tokens: 65_536,
            speculative_draft_tokens: 2,
            context_size: None,
            vulkan_device_index: None,
            random_seed: 0,
            add_special_tokens: true,
            skip_special_tokens: true,
            generated_only: false,
            profile: false,
            profile_runs: 1,
            json: false,
        }
    }
}

fn main() {
    if let Err(error) = run() {
        eprintln!("llmoop-runtime error: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn Error>> {
    if std::env::args()
        .skip(1)
        .any(|arg| arg == "--help" || arg == "-h")
    {
        print_usage();
        return Ok(());
    }

    let args = parse_args().map_err(|error| io::Error::new(io::ErrorKind::InvalidInput, error))?;
    let package_manifest = args.package_manifest.as_ref().ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            "--package is required; run `python -m llmoop --compile-model <MODEL_DIR>` first",
        )
    })?;
    let manifest_dir = package_manifest
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."));
    if args.inspect_runtime {
        let manifest = VulkanResidentModelPackageManifest::from_json_file(package_manifest)?;
        return inspect_runtime_topology(&args, package_manifest, &manifest_dir, manifest);
    }
    if args.inspect_package {
        let manifest = VulkanResidentModelPackageManifest::from_json_file(package_manifest)?;
        return inspect_package(&args, package_manifest, &manifest_dir, manifest);
    }
    if args.inspect_patch {
        let manifest = VulkanResidentModelPackageManifest::from_json_file(package_manifest)?;
        return inspect_patch(&args, package_manifest, &manifest_dir, manifest);
    }
    let runtime_model = runtime_model(&args, package_manifest)?;
    if args.inspect_placement {
        return inspect_placement(&args, package_manifest, &manifest_dir, runtime_model);
    }
    if let Some(device_id) = args.inspect_device_slice.as_deref() {
        return inspect_device_slice(
            &args,
            package_manifest,
            &manifest_dir,
            runtime_model,
            device_id,
        );
    }
    let tokenizer_dir = tokenizer_dir_from_package(package_manifest)?;
    let codec = VulkanResidentHfTokenizerTextCodec::from_model_dir(&tokenizer_dir)?
        .with_add_special_tokens(args.add_special_tokens)
        .with_skip_special_tokens(args.skip_special_tokens);
    if args.chat {
        let capacity = choose_chat_runtime_context_size(package_manifest, args.context_size)?;
        return run_placed_chat(
            &args,
            &manifest_dir,
            &tokenizer_dir,
            runtime_model,
            capacity,
            &codec,
            args.prompt.as_deref(),
        );
    }
    let prompt = args
        .prompt
        .as_ref()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "--prompt is required"))?;
    let prompt_ids = codec.encode_text(prompt)?;
    let scheduled_token_activations = prompt_ids
        .len()
        .checked_add(args.max_new_tokens)
        .ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "prompt token count plus --max-new-tokens overflowed usize",
            )
        })?;
    let capacity =
        choose_runtime_context_size(package_manifest, args.context_size, prompt_ids.len())?;
    let context = PromptRunContext {
        args: &args,
        package_manifest,
        manifest_dir: &manifest_dir,
        tokenizer_dir: &tokenizer_dir,
        prompt,
        prompt_ids: &prompt_ids,
        scheduled_token_activations,
        capacity,
        codec: &codec,
    };

    if args.profile_runs > 1 {
        return run_placed_prompt_benchmark(&context, runtime_model);
    }
    run_placed_prompt(&context, runtime_model)
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
struct RuntimeChatMessage {
    role: String,
    content: String,
}

#[derive(Clone, Debug)]
struct RuntimeChatSession {
    formatter: RuntimeChatFormatter,
    messages: Vec<RuntimeChatMessage>,
}

fn chat_transcript_codec(
    tokenizer_dir: &Path,
) -> Result<VulkanResidentHfTokenizerTextCodec, Box<dyn Error>> {
    Ok(
        VulkanResidentHfTokenizerTextCodec::from_model_dir(tokenizer_dir)?
            .with_add_special_tokens(false)
            .with_skip_special_tokens(false),
    )
}

impl RuntimeChatSession {
    fn from_tokenizer_dir(tokenizer_dir: &Path) -> Result<Self, Box<dyn Error>> {
        Ok(Self {
            formatter: RuntimeChatFormatter::from_tokenizer_dir(tokenizer_dir)?,
            messages: Vec::new(),
        })
    }

    fn render_user_prompt_token_delta<C>(
        &self,
        user_content: &str,
        codec: &C,
    ) -> Result<Vec<u32>, Box<dyn Error>>
    where
        C: VulkanResidentTokenTextCodec,
    {
        if self.messages.is_empty() {
            let messages = vec![RuntimeChatMessage {
                role: "user".to_string(),
                content: user_content.to_string(),
            }];
            let formatted = self.formatter.format_messages(&messages, true)?;
            return Ok(codec.encode_text(&formatted)?);
        }

        const ASSISTANT_CONTENT_PROBE: &str = "LLMOOP_PREVIOUS_ASSISTANT_CONTENT_PROBE_6C70A8";
        let mut probe_history = self.messages.clone();
        let previous_assistant = probe_history.last_mut().ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "chat history is unexpectedly empty",
            )
        })?;
        if previous_assistant.role != "assistant" {
            return Err(Box::new(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "resident chat history must end with an assistant turn, found {:?}",
                    previous_assistant.role
                ),
            )));
        }
        previous_assistant.content = ASSISTANT_CONTENT_PROBE.to_string();
        let formatted_history = self.formatter.format_messages(&probe_history, false)?;
        let history_token_ids = codec.encode_text(&formatted_history)?;

        probe_history.push(RuntimeChatMessage {
            role: "user".to_string(),
            content: user_content.to_string(),
        });
        let formatted_continuation = self.formatter.format_messages(&probe_history, true)?;
        let continuation_token_ids = codec.encode_text(&formatted_continuation)?;
        Ok(incremental_chat_token_delta(
            &history_token_ids,
            &continuation_token_ids,
        )?)
    }

    fn commit_assistant_turn(&mut self, user_content: &str, assistant_content: &str) {
        self.messages.push(RuntimeChatMessage {
            role: "user".to_string(),
            content: user_content.to_string(),
        });
        self.messages.push(RuntimeChatMessage {
            role: "assistant".to_string(),
            content: assistant_content.to_string(),
        });
    }
}

fn incremental_chat_token_delta(
    rendered_history_token_ids: &[u32],
    rendered_continuation_token_ids: &[u32],
) -> Result<Vec<u32>, io::Error> {
    if !rendered_continuation_token_ids.starts_with(rendered_history_token_ids) {
        let common_prefix_len = rendered_history_token_ids
            .iter()
            .zip(rendered_continuation_token_ids)
            .take_while(|(history, continuation)| history == continuation)
            .count();
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "chat template rewrote previously resident turn framing at token {common_prefix_len}; incremental continuation requires the completed assistant probe to remain an exact prefix"
            ),
        ));
    }
    Ok(rendered_continuation_token_ids[rendered_history_token_ids.len()..].to_vec())
}

#[derive(Clone, Debug)]
struct RuntimeChatFormatter {
    template_source: String,
    template_variables: serde_json::Map<String, serde_json::Value>,
    render_time: DateTime<FixedOffset>,
}

impl RuntimeChatFormatter {
    fn from_tokenizer_dir(tokenizer_dir: &Path) -> Result<Self, Box<dyn Error>> {
        let template_path = tokenizer_dir.join("chat_template.jinja");
        let template = fs::read_to_string(&template_path).map_err(|error| {
            io::Error::new(
                error.kind(),
                format!(
                    "chat mode requires a supported chat template; failed to read {:?}: {error}",
                    template_path
                ),
            )
        })?;
        let formatter = Self {
            template_source: normalize_chat_template_for_runtime(&template),
            template_variables: tokenizer_template_variables(tokenizer_dir)?,
            render_time: Local::now().fixed_offset(),
        };
        formatter.format_messages(
            &[RuntimeChatMessage {
                role: "user".to_string(),
                content: "template validation".to_string(),
            }],
            true,
        )?;
        Ok(formatter)
    }

    fn format_messages(
        &self,
        messages: &[RuntimeChatMessage],
        add_generation_prompt: bool,
    ) -> Result<String, Box<dyn Error>> {
        let mut environment = Environment::new();
        environment
            .set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        environment.add_function(
            "raise_exception",
            |message: String| -> Result<String, TemplateError> {
                Err(TemplateError::new(
                    TemplateErrorKind::InvalidOperation,
                    message,
                ))
            },
        );
        let render_time = self.render_time;
        environment.add_function("strftime_now", move |format: String| {
            render_time.format(&format).to_string()
        });
        environment.add_template("chat", &self.template_source)?;

        let mut context = self.template_variables.clone();
        context.insert("messages".to_string(), serde_json::to_value(messages)?);
        context.insert(
            "add_generation_prompt".to_string(),
            serde_json::Value::Bool(add_generation_prompt),
        );
        Ok(environment.get_template("chat")?.render(context)?)
    }
}

fn normalize_chat_template_for_runtime(source: &str) -> String {
    let mut normalized = String::with_capacity(source.len());
    let mut cursor = 0usize;
    while let Some(relative_start) = source[cursor..].find("{%") {
        let start = cursor + relative_start;
        let tag_body_start = start + 2;
        let Some(relative_end) = source[tag_body_start..].find("%}") else {
            break;
        };
        let end = tag_body_start + relative_end;
        let tag_body = &source[tag_body_start..end];
        let statement = tag_body.trim().trim_matches('-').trim();
        normalized.push_str(&source[cursor..start]);
        if matches!(statement, "generation" | "endgeneration") {
            normalized.push_str(if tag_body.starts_with('-') {
                "{#-"
            } else {
                "{#"
            });
            normalized.push_str(statement);
            normalized.push_str(if tag_body.ends_with('-') { "-#}" } else { "#}" });
        } else {
            normalized.push_str(&source[start..end + 2]);
        }
        cursor = end + 2;
    }
    normalized.push_str(&source[cursor..]);
    normalized
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct RuntimeChatTurn {
    generated_token_ids: Vec<u32>,
    streamed: bool,
}

fn run_chat_repl<C, T, F>(
    initial_prompt: Option<&str>,
    mut chat_session: RuntimeChatSession,
    codec: &C,
    transcript_codec: &T,
    stop_token_ids: &[u32],
    mut submit: F,
) -> Result<(), Box<dyn Error>>
where
    C: VulkanResidentTokenTextCodec,
    T: VulkanResidentTokenTextCodec,
    F: FnMut(usize, &[u32]) -> Result<RuntimeChatTurn, Box<dyn Error>>,
{
    println!("Type a message and press Enter. Type /exit, /quit, exit, or quit to stop.");
    let mut turn_index = 0usize;
    if let Some(initial_prompt) = initial_prompt
        && !initial_prompt.trim().is_empty()
    {
        if !submit_chat_turn(
            &mut chat_session,
            codec,
            transcript_codec,
            stop_token_ids,
            &mut submit,
            turn_index,
            initial_prompt,
        )? {
            return Ok(());
        }
        turn_index = turn_index.saturating_add(1);
    }

    let stdin = io::stdin();
    let mut line = String::new();
    loop {
        print!("you> ");
        io::stdout().flush()?;
        line.clear();
        if stdin.read_line(&mut line)? == 0 {
            println!();
            break;
        }
        let input_text = line.trim_end_matches(['\r', '\n']);
        let command = input_text.trim();
        if command.eq_ignore_ascii_case("exit")
            || command.eq_ignore_ascii_case("quit")
            || command.eq_ignore_ascii_case("/exit")
            || command.eq_ignore_ascii_case("/quit")
        {
            break;
        }
        if command.is_empty() {
            continue;
        }

        if !submit_chat_turn(
            &mut chat_session,
            codec,
            transcript_codec,
            stop_token_ids,
            &mut submit,
            turn_index,
            input_text,
        )? {
            break;
        }
        turn_index = turn_index.saturating_add(1);
    }
    Ok(())
}

fn submit_chat_turn<C, T, F>(
    chat_session: &mut RuntimeChatSession,
    codec: &C,
    transcript_codec: &T,
    stop_token_ids: &[u32],
    submit: &mut F,
    turn_index: usize,
    input_text: &str,
) -> Result<bool, Box<dyn Error>>
where
    C: VulkanResidentTokenTextCodec,
    T: VulkanResidentTokenTextCodec,
    F: FnMut(usize, &[u32]) -> Result<RuntimeChatTurn, Box<dyn Error>>,
{
    let prompt_delta = chat_session.render_user_prompt_token_delta(input_text, transcript_codec)?;
    match submit(turn_index, &prompt_delta) {
        Ok(turn) => {
            let generated_text = codec.decode_tokens(&turn.generated_token_ids)?;
            let assistant_content_ids =
                assistant_content_token_ids(&turn.generated_token_ids, stop_token_ids);
            let assistant_content = transcript_codec.decode_tokens(assistant_content_ids)?;
            if turn.streamed {
                println!();
            } else {
                print_chat_response(&generated_text);
            }
            chat_session.commit_assistant_turn(input_text, &assistant_content);
            Ok(true)
        }
        Err(error) => Err(error),
    }
}

fn assistant_content_token_ids<'a>(
    generated_token_ids: &'a [u32],
    stop_token_ids: &[u32],
) -> &'a [u32] {
    let mut content_len = generated_token_ids.len();
    while content_len > 0 && stop_token_ids.contains(&generated_token_ids[content_len - 1]) {
        content_len -= 1;
    }
    &generated_token_ids[..content_len]
}

fn print_chat_response(text: &str) {
    print!("llm> ");
    print_text(text);
}

fn tokenizer_template_variables(
    tokenizer_dir: &Path,
) -> Result<serde_json::Map<String, serde_json::Value>, Box<dyn Error>> {
    let path = tokenizer_dir.join("tokenizer_config.json");
    if !path.is_file() {
        return Ok(serde_json::Map::new());
    }
    let config: serde_json::Value = serde_json::from_slice(&fs::read(&path)?)?;
    let object = config.as_object().ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("tokenizer config {path:?} must contain a JSON object"),
        )
    })?;
    Ok(object
        .iter()
        .map(|(key, value)| {
            let value = if key.ends_with("_token") {
                value
                    .get("content")
                    .and_then(serde_json::Value::as_str)
                    .map(|content| serde_json::Value::String(content.to_string()))
                    .unwrap_or_else(|| value.clone())
            } else {
                value.clone()
            };
            (key.clone(), value)
        })
        .collect())
}

fn chat_stop_token_ids_from_manifest(
    manifest_dir: &Path,
    tokenizer_dir: &Path,
    manifest: &VulkanResidentModelPackageManifest,
    formatter: &RuntimeChatFormatter,
) -> Result<Vec<u32>, Box<dyn Error>> {
    let config_path = manifest_dir.join(&manifest.config_path);
    let eos_values = if config_path.is_file() {
        let config: serde_json::Value = serde_json::from_slice(&fs::read(&config_path)?)?;
        let raw_eos = config.get("eos_token_id");
        if let Some(id) = raw_eos.and_then(serde_json::Value::as_u64) {
            vec![id]
        } else if let Some(ids) = raw_eos.and_then(serde_json::Value::as_array) {
            ids.iter()
                .filter_map(serde_json::Value::as_u64)
                .collect::<Vec<_>>()
        } else {
            Vec::new()
        }
    } else {
        Vec::new()
    };
    let mut stop_token_ids = eos_values
        .into_iter()
        .map(|id| {
            u32::try_from(id).map_err(|_| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("eos_token_id {id} does not fit in u32"),
                )
            })
        })
        .collect::<Result<Vec<_>, _>>()
        .map_err(Box::<dyn Error>::from)?;

    let tokenizer_config_path = tokenizer_dir.join("tokenizer_config.json");
    if tokenizer_config_path.is_file() {
        let tokenizer_config: serde_json::Value =
            serde_json::from_slice(&fs::read(&tokenizer_config_path)?)?;
        let eos_token = tokenizer_config.get("eos_token").and_then(|value| {
            value
                .as_str()
                .or_else(|| value.get("content").and_then(serde_json::Value::as_str))
        });
        if let Some(eos_token) = eos_token {
            let stop_codec = VulkanResidentHfTokenizerTextCodec::from_model_dir(tokenizer_dir)?
                .with_add_special_tokens(false)
                .with_skip_special_tokens(false);
            let encoded = stop_codec.encode_text(eos_token)?;
            let [token_id] = encoded.as_slice() else {
                return Err(Box::new(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "chat tokenizer eos_token {eos_token:?} must encode to exactly one token, got {encoded:?}"
                    ),
                )));
            };
            if !stop_token_ids.contains(token_id) {
                stop_token_ids.push(*token_id);
            }
        }
    }
    if let Some(token_id) = model_owned_assistant_turn_stop_token_id(tokenizer_dir, formatter)?
        && !stop_token_ids.contains(&token_id)
    {
        stop_token_ids.push(token_id);
    }
    Ok(stop_token_ids)
}

fn model_owned_assistant_turn_stop_token_id(
    tokenizer_dir: &Path,
    formatter: &RuntimeChatFormatter,
) -> Result<Option<u32>, Box<dyn Error>> {
    const ASSISTANT_SENTINEL: &str = "LLMOOP_ASSISTANT_TURN_CONTENT_SENTINEL_7F3A9C";
    let rendered = formatter.format_messages(
        &[
            RuntimeChatMessage {
                role: "user".to_string(),
                content: "Discover the model-owned assistant turn delimiter.".to_string(),
            },
            RuntimeChatMessage {
                role: "assistant".to_string(),
                content: ASSISTANT_SENTINEL.to_string(),
            },
        ],
        false,
    )?;
    let sentinel_start = rendered.rfind(ASSISTANT_SENTINEL).ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            "chat template did not preserve the synthetic assistant content used to discover its turn delimiter",
        )
    })?;
    let suffix = &rendered[sentinel_start + ASSISTANT_SENTINEL.len()..];
    let codec = VulkanResidentHfTokenizerTextCodec::from_model_dir(tokenizer_dir)?
        .with_add_special_tokens(false)
        .with_skip_special_tokens(false);
    let suffix_token_ids = codec.encode_text(suffix)?;
    let special_token_ids = tokenizer_special_token_ids(tokenizer_dir)?;
    Ok(first_special_token_id(
        &suffix_token_ids,
        &special_token_ids,
    ))
}

fn tokenizer_special_token_ids(tokenizer_dir: &Path) -> Result<BTreeSet<u32>, Box<dyn Error>> {
    let path = tokenizer_dir.join("tokenizer.json");
    let tokenizer: serde_json::Value = serde_json::from_slice(&fs::read(&path)?)?;
    tokenizer
        .get("added_tokens")
        .and_then(serde_json::Value::as_array)
        .into_iter()
        .flatten()
        .filter(|token| {
            token
                .get("special")
                .and_then(serde_json::Value::as_bool)
                .unwrap_or(false)
        })
        .filter_map(|token| token.get("id").and_then(serde_json::Value::as_u64))
        .map(|id| {
            u32::try_from(id).map_err(|_| {
                Box::<dyn Error>::from(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("special token id {id} in {path:?} does not fit in u32"),
                ))
            })
        })
        .collect()
}

fn first_special_token_id(token_ids: &[u32], special_token_ids: &BTreeSet<u32>) -> Option<u32> {
    token_ids
        .iter()
        .copied()
        .find(|token_id| special_token_ids.contains(token_id))
}

fn run_placed_chat(
    args: &Args,
    manifest_dir: &Path,
    tokenizer_dir: &Path,
    runtime_model: VulkanResidentRuntimeModel,
    capacity: usize,
    codec: &VulkanResidentHfTokenizerTextCodec,
    initial_prompt: Option<&str>,
) -> Result<(), Box<dyn Error>> {
    let setup_start = Instant::now();
    let chat_session = RuntimeChatSession::from_tokenizer_dir(tokenizer_dir)?;
    let stop_token_ids = chat_stop_token_ids_from_manifest(
        manifest_dir,
        tokenizer_dir,
        &runtime_model.package,
        &chat_session.formatter,
    )?;
    let transcript_codec = chat_transcript_codec(tokenizer_dir)?;
    let logical_device_ids = runtime_model.placement_device_ids();
    let bound_devices = runtime_bound_vulkan_devices(args, &logical_device_ids)?;
    let stream = VulkanResidentInProcessPlacedPromptStream::from_runtime_model_for_bound_devices(
        bound_devices.devices.clone(),
        manifest_dir,
        runtime_model,
        Some(capacity),
        args.random_seed,
        args.speculative_draft_tokens,
    )?;
    let mut engine = VulkanResidentInProcessPlacedPromptEngine::new();
    let stream_snapshot = engine.add_stream("main", stream)?;
    println!(
        "llmoop chat ready: placed_in_process, devices={:?}, context_size={}, setup_ms={:.3}",
        stream_snapshot.device_ids,
        stream_snapshot.context_window_activations,
        nanos_to_millis(elapsed_nanos_u64(setup_start))
    );

    run_chat_repl(
        initial_prompt,
        chat_session,
        codec,
        &transcript_codec,
        &stop_token_ids,
        |turn_index, token_ids| {
            print!("llm> ");
            io::stdout().flush()?;
            let mut decoder = codec.decode_stream();
            let mut output_error = None;
            let mut event = VulkanResidentTokenInputEvent::new(
                format!("chat_{turn_index}"),
                token_ids.to_vec(),
                args.max_new_tokens,
            )
            .with_origin("cli_chat");
            if !stop_token_ids.is_empty() {
                event = event.with_stop_tokens(stop_token_ids.clone());
            }
            let run = engine.submit_input_event_until_idle_with_output(
                "main",
                event,
                |output_event| {
                    if output_error.is_some() {
                        return;
                    }
                    match decoder.step(output_event.output_event.token_id) {
                        Ok(Some(text)) => {
                            print!("{text}");
                            if let Err(error) = io::stdout().flush() {
                                output_error = Some(error.to_string());
                            }
                        }
                        Ok(None) => {}
                        Err(error) => output_error = Some(error.to_string()),
                    }
                },
            )?;
            if let Some(error) = output_error {
                return Err(Box::new(io::Error::new(io::ErrorKind::InvalidData, error)));
            }
            Ok(RuntimeChatTurn {
                generated_token_ids: run.generated_token_ids,
                streamed: true,
            })
        },
    )
}

fn run_placed_prompt(
    context: &PromptRunContext<'_>,
    runtime_model: VulkanResidentRuntimeModel,
) -> Result<(), Box<dyn Error>> {
    let report = execute_placed_prompt_run(context, runtime_model)?;
    print_placed_prompt_report(context.args, &report)
}

fn execute_placed_prompt_run(
    context: &PromptRunContext<'_>,
    runtime_model: VulkanResidentRuntimeModel,
) -> Result<RuntimePlacedPromptRunReport, Box<dyn Error>> {
    let PromptRunContext {
        args,
        package_manifest,
        manifest_dir,
        tokenizer_dir,
        prompt,
        prompt_ids,
        scheduled_token_activations,
        capacity,
        codec,
        ..
    } = context;
    let setup_start = Instant::now();
    let logical_device_ids = runtime_model.placement_device_ids();
    let placement = runtime_model_placement(manifest_dir, &runtime_model)?;
    let bound_devices = runtime_bound_vulkan_devices(args, &logical_device_ids)?;
    let stream = VulkanResidentInProcessPlacedPromptStream::from_runtime_model_for_bound_devices(
        bound_devices.devices.clone(),
        manifest_dir,
        runtime_model,
        Some(*capacity),
        args.random_seed,
        args.speculative_draft_tokens,
    )?;
    let mut engine = VulkanResidentInProcessPlacedPromptEngine::new();
    let stream_snapshot = engine.add_stream("main", stream)?;
    let setup_time_ns = elapsed_nanos_u64(setup_start);
    let run_start = Instant::now();
    let input_event =
        VulkanResidentTokenInputEvent::new("prompt", prompt_ids.to_vec(), args.max_new_tokens);
    let input_event_id = input_event.id.clone();
    let submitted_run = engine.submit_input_event_until_idle("main", input_event)?;
    let run_time_ns = elapsed_nanos_u64(run_start);
    let run = submitted_run
        .engine_run
        .input_runs
        .into_iter()
        .find(|run| run.stream_id == "main" && run.submitted_run.input_event.id == input_event_id)
        .ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::NotFound,
                "placed prompt engine run loop did not return the submitted prompt event run",
            )
        })?
        .submitted_run
        .session_run
        .run;
    let generated_text = codec.decode_tokens(&run.generated_token_ids)?;
    let output_text = codec.decode_tokens(&run.output_token_ids)?;
    let total_scheduler_turns = run.scheduler_turn_count;
    let completed_stage_deltas = vec![run.completed_stage_count];
    let tick_count = run.tick_count;
    let generated_token_count = run.generated_token_ids.len();
    let timing = runtime_prompt_timing_report(
        setup_time_ns,
        run_time_ns,
        generated_token_count,
        tick_count,
        total_scheduler_turns,
    );
    let pedal_timings = Vec::new();
    let pedal_timing_summaries = Vec::new();
    let transport_stats_by_tick = Vec::new();
    let transport_published_packet_count = run.transport_stats.published_packet_count;
    let transport_published_byte_count = run.transport_stats.published_byte_count;
    let transport_received_packet_count = run.transport_stats.received_packet_count;
    let transport_received_byte_count = run.transport_stats.received_byte_count;
    let transport_direct_copy_count = run.transport_stats.direct_copy_count;
    let transport_direct_copy_byte_count = run.transport_stats.direct_copy_byte_count;
    let transport_direct_receive_count = run.transport_stats.direct_receive_count;
    let transport_direct_receive_byte_count = run.transport_stats.direct_receive_byte_count;

    Ok(RuntimePlacedPromptRunReport {
        ok: true,
        execution_mode: "placed_in_process".to_string(),
        package_manifest: package_manifest.to_path_buf(),
        tokenizer_dir: tokenizer_dir.to_path_buf(),
        input_device_id: stream_snapshot.input_device_id.clone(),
        output_device_id: stream_snapshot.output_device_id.clone(),
        device_count: stream_snapshot.device_ids.len(),
        device_ids: stream_snapshot.device_ids.clone(),
        bound_devices: bound_devices_report(&bound_devices),
        cable_routes: bound_cable_routes_report(&bound_devices, &placement.cables),
        runtime_patch: runtime_patch_report(args),
        device_bindings: runtime_device_bindings_report(args, &stream_snapshot.device_ids),
        hosted_pedal_count: stream_snapshot.hosted_pedal_count,
        context_window_activations: stream_snapshot.context_window_activations,
        scheduled_token_activations: *scheduled_token_activations,
        tokenizer: tokenizer_options_report(args),
        prompt_text: prompt.to_string(),
        prompt_ids: run.prompt_token_ids.clone(),
        generated_ids: run.generated_token_ids.clone(),
        generated_text: generated_text.clone(),
        output_text: output_text.clone(),
        stop_reason: run.stop_reason.clone(),
        tick_count,
        scheduler_turns: total_scheduler_turns,
        completed_stage_deltas,
        transport: RuntimePlacedTransportReport {
            published_packet_count: transport_published_packet_count,
            published_byte_count: transport_published_byte_count,
            received_packet_count: transport_received_packet_count,
            received_byte_count: transport_received_byte_count,
            direct_copy_count: transport_direct_copy_count,
            direct_copy_byte_count: transport_direct_copy_byte_count,
            direct_receive_count: transport_direct_receive_count,
            direct_receive_byte_count: transport_direct_receive_byte_count,
            by_tick: transport_stats_by_tick,
        },
        timing,
        pedal_timings,
        pedal_timing_summaries,
        speculative_cycle_count: run.speculative_decode.cycle_count,
        proposed_draft_token_count: run.speculative_decode.proposed_draft_token_count,
        accepted_draft_token_count: run.speculative_decode.accepted_draft_token_count,
        speculative_emitted_token_count: run.speculative_decode.emitted_token_count,
        speculative_draft_time_ns: run.speculative_decode.draft_time_ns,
        speculative_target_verification_time_ns: run.speculative_decode.target_verification_time_ns,
        speculative_draft_catch_up_time_ns: run.speculative_decode.draft_catch_up_time_ns,
    })
}

fn print_placed_prompt_report(
    args: &Args,
    report: &RuntimePlacedPromptRunReport,
) -> Result<(), Box<dyn Error>> {
    if args.json {
        println!("{}", serde_json::to_string_pretty(report)?);
    } else if args.generated_only {
        print_text(&report.generated_text);
    } else {
        print_text(&report.output_text);
        if args.profile {
            print_prompt_timing_profile(&report.timing);
            print_speculative_profile(report);
            print_placed_pedal_timing_profile(&report.pedal_timing_summaries, 5);
        }
    }
    Ok(())
}

fn print_speculative_profile(report: &RuntimePlacedPromptRunReport) {
    if report.speculative_cycle_count == 0 {
        return;
    }
    let acceptance = if report.proposed_draft_token_count == 0 {
        0.0
    } else {
        100.0 * report.accepted_draft_token_count as f64 / report.proposed_draft_token_count as f64
    };
    println!("speculative:");
    println!("  cycles={}", report.speculative_cycle_count);
    println!(
        "  drafts proposed={} accepted={} acceptance={acceptance:.2}%",
        report.proposed_draft_token_count, report.accepted_draft_token_count
    );
    println!(
        "  emitted_tokens={}",
        report.speculative_emitted_token_count
    );
    println!(
        "  draft_ms={:.3}",
        nanos_to_millis(report.speculative_draft_time_ns)
    );
    println!(
        "  target_verification_ms={:.3}",
        nanos_to_millis(report.speculative_target_verification_time_ns)
    );
    println!(
        "  draft_catch_up_ms={:.3}",
        nanos_to_millis(report.speculative_draft_catch_up_time_ns)
    );
}

fn run_placed_prompt_benchmark(
    context: &PromptRunContext<'_>,
    runtime_model: VulkanResidentRuntimeModel,
) -> Result<(), Box<dyn Error>> {
    let mut runs = Vec::with_capacity(context.args.profile_runs);
    for _ in 0..context.args.profile_runs {
        runs.push(execute_placed_prompt_run(context, runtime_model.clone())?);
    }
    let first = runs
        .first()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "--profile-runs is empty"))?;
    let benchmark_runs = runs
        .iter()
        .enumerate()
        .map(|(run_index, run)| placed_benchmark_run_report(run_index, run))
        .collect::<Vec<_>>();
    let benchmark = runtime_prompt_benchmark_report(
        context,
        &first.execution_mode,
        first.device_ids.clone(),
        first.device_bindings.clone(),
        benchmark_runs,
    );
    print_prompt_benchmark_report(context.args, &benchmark)
}

fn placed_benchmark_run_report(
    run_index: usize,
    run: &RuntimePlacedPromptRunReport,
) -> RuntimePromptBenchmarkRunReport {
    RuntimePromptBenchmarkRunReport {
        run_index,
        execution_mode: run.execution_mode.clone(),
        stop_reason: run.stop_reason.clone(),
        generated_token_count: run.timing.generated_token_count,
        tick_count: run.timing.tick_count,
        scheduler_turn_count: run.timing.scheduler_turn_count,
        setup_time_ns: run.timing.setup_time_ns,
        run_time_ns: run.timing.run_time_ns,
        total_time_ns: run.timing.total_time_ns,
        generated_tokens_per_second: generated_tokens_per_second(
            run.timing.generated_token_count,
            run.timing.run_time_ns,
        ),
        transport: Some(benchmark_transport_totals_from_report(&run.transport)),
        pedal_timing_summaries: run.pedal_timing_summaries.clone(),
    }
}

fn runtime_prompt_benchmark_report(
    context: &PromptRunContext<'_>,
    execution_mode: &str,
    device_ids: Vec<String>,
    device_bindings: RuntimeDeviceBindings,
    runs: Vec<RuntimePromptBenchmarkRunReport>,
) -> RuntimePromptBenchmarkReport {
    let PromptRunContext {
        args,
        package_manifest,
        tokenizer_dir,
        prompt,
        prompt_ids,
        ..
    } = context;
    let setup_values = runs.iter().map(|run| run.setup_time_ns).collect::<Vec<_>>();
    let run_values = runs.iter().map(|run| run.run_time_ns).collect::<Vec<_>>();
    let total_values = runs.iter().map(|run| run.total_time_ns).collect::<Vec<_>>();
    let generated_token_values = runs
        .iter()
        .map(|run| run.generated_token_count)
        .collect::<Vec<_>>();
    let tick_values = runs.iter().map(|run| run.tick_count).collect::<Vec<_>>();
    let scheduler_turn_values = runs
        .iter()
        .map(|run| run.scheduler_turn_count)
        .collect::<Vec<_>>();
    let mut stop_reasons = BTreeMap::new();
    for run in &runs {
        *stop_reasons.entry(run.stop_reason.clone()).or_insert(0) += 1;
    }
    let total_generated_tokens = generated_token_values.iter().sum::<usize>();
    let total_run_time_ns = run_values.iter().sum::<u64>();

    RuntimePromptBenchmarkReport {
        ok: true,
        execution_mode: execution_mode.to_string(),
        package_manifest: package_manifest.to_path_buf(),
        tokenizer_dir: tokenizer_dir.to_path_buf(),
        runtime_patch: runtime_patch_report(args),
        device_bindings,
        device_count: device_ids.len(),
        device_ids,
        profile_runs: runs.len(),
        prompt_text: prompt.to_string(),
        prompt_ids: prompt_ids.to_vec(),
        max_new_tokens: args.max_new_tokens,
        setup_time_ns: benchmark_u64_metric(&setup_values),
        run_time_ns: benchmark_u64_metric(&run_values),
        total_time_ns: benchmark_u64_metric(&total_values),
        generated_token_count: benchmark_usize_metric(&generated_token_values),
        tick_count: benchmark_usize_metric(&tick_values),
        scheduler_turn_count: benchmark_usize_metric(&scheduler_turn_values),
        generated_tokens_per_second: generated_tokens_per_second(
            total_generated_tokens,
            total_run_time_ns,
        ),
        stop_reasons,
        transport_totals: aggregate_benchmark_transport_totals(&runs),
        pedal_timing_summaries: aggregate_benchmark_pedal_timing_summaries(&runs),
        runs,
    }
}

fn benchmark_u64_metric(values: &[u64]) -> RuntimePromptBenchmarkU64MetricReport {
    let total = values.iter().sum::<u64>();
    RuntimePromptBenchmarkU64MetricReport {
        total,
        min: values.iter().copied().min().unwrap_or(0),
        max: values.iter().copied().max().unwrap_or(0),
        average: if values.is_empty() {
            0.0
        } else {
            total as f64 / values.len() as f64
        },
    }
}

fn benchmark_usize_metric(values: &[usize]) -> RuntimePromptBenchmarkUsizeMetricReport {
    let total = values.iter().sum::<usize>();
    RuntimePromptBenchmarkUsizeMetricReport {
        total,
        min: values.iter().copied().min().unwrap_or(0),
        max: values.iter().copied().max().unwrap_or(0),
        average: if values.is_empty() {
            0.0
        } else {
            total as f64 / values.len() as f64
        },
    }
}

fn generated_tokens_per_second(generated_token_count: usize, run_time_ns: u64) -> Option<f64> {
    if run_time_ns == 0 {
        None
    } else {
        Some(generated_token_count as f64 / (run_time_ns as f64 / 1_000_000_000.0))
    }
}

fn benchmark_transport_totals_from_report(
    transport: &RuntimePlacedTransportReport,
) -> RuntimePromptBenchmarkTransportTotalsReport {
    RuntimePromptBenchmarkTransportTotalsReport {
        published_packet_count: transport.published_packet_count,
        published_byte_count: transport.published_byte_count,
        received_packet_count: transport.received_packet_count,
        received_byte_count: transport.received_byte_count,
        direct_copy_count: transport.direct_copy_count,
        direct_copy_byte_count: transport.direct_copy_byte_count,
        direct_receive_count: transport.direct_receive_count,
        direct_receive_byte_count: transport.direct_receive_byte_count,
    }
}

fn aggregate_benchmark_transport_totals(
    runs: &[RuntimePromptBenchmarkRunReport],
) -> Option<RuntimePromptBenchmarkTransportTotalsReport> {
    let mut total = RuntimePromptBenchmarkTransportTotalsReport {
        published_packet_count: 0,
        published_byte_count: 0,
        received_packet_count: 0,
        received_byte_count: 0,
        direct_copy_count: 0,
        direct_copy_byte_count: 0,
        direct_receive_count: 0,
        direct_receive_byte_count: 0,
    };
    let mut seen = false;
    for transport in runs.iter().filter_map(|run| run.transport.as_ref()) {
        seen = true;
        total.published_packet_count = total
            .published_packet_count
            .saturating_add(transport.published_packet_count);
        total.published_byte_count = total
            .published_byte_count
            .saturating_add(transport.published_byte_count);
        total.received_packet_count = total
            .received_packet_count
            .saturating_add(transport.received_packet_count);
        total.received_byte_count = total
            .received_byte_count
            .saturating_add(transport.received_byte_count);
        total.direct_copy_count = total
            .direct_copy_count
            .saturating_add(transport.direct_copy_count);
        total.direct_copy_byte_count = total
            .direct_copy_byte_count
            .saturating_add(transport.direct_copy_byte_count);
        total.direct_receive_count = total
            .direct_receive_count
            .saturating_add(transport.direct_receive_count);
        total.direct_receive_byte_count = total
            .direct_receive_byte_count
            .saturating_add(transport.direct_receive_byte_count);
    }
    seen.then_some(total)
}

fn aggregate_benchmark_pedal_timing_summaries(
    runs: &[RuntimePromptBenchmarkRunReport],
) -> Vec<RuntimePlacedPedalTimingSummaryReport> {
    let mut summaries = BTreeMap::<(String, String), RuntimePlacedPedalTimingSummaryReport>::new();
    for run in runs {
        for timing in &run.pedal_timing_summaries {
            let entry = summaries
                .entry((timing.device_id.clone(), timing.pedal_id.clone()))
                .or_insert_with(|| RuntimePlacedPedalTimingSummaryReport {
                    device_id: timing.device_id.clone(),
                    pedal_id: timing.pedal_id.clone(),
                    tick_count: 0,
                    dispatch_count: 0,
                    total_run_time_ns: 0,
                    average_tick_time_ns: None,
                    average_dispatch_time_ns: None,
                });
            entry.tick_count += timing.tick_count;
            entry.dispatch_count += timing.dispatch_count;
            entry.total_run_time_ns = entry
                .total_run_time_ns
                .saturating_add(timing.total_run_time_ns);
        }
    }
    let mut summaries = summaries.into_values().collect::<Vec<_>>();
    for summary in &mut summaries {
        summary.average_tick_time_ns = average_nanos(summary.total_run_time_ns, summary.tick_count);
        summary.average_dispatch_time_ns =
            average_nanos(summary.total_run_time_ns, summary.dispatch_count);
    }
    summaries.sort_by(|left, right| {
        right
            .total_run_time_ns
            .cmp(&left.total_run_time_ns)
            .then_with(|| left.device_id.cmp(&right.device_id))
            .then_with(|| left.pedal_id.cmp(&right.pedal_id))
    });
    summaries
}

fn print_prompt_benchmark_report(
    args: &Args,
    report: &RuntimePromptBenchmarkReport,
) -> Result<(), Box<dyn Error>> {
    if args.json {
        println!("{}", serde_json::to_string_pretty(report)?);
        return Ok(());
    }

    println!("benchmark:");
    println!("  execution_mode={}", report.execution_mode);
    println!("  runs={}", report.profile_runs);
    println!("  devices={}", report.device_ids.join(","));
    println!(
        "  setup_ms avg={:.3} min={:.3} max={:.3}",
        nanos_to_millis_f64(report.setup_time_ns.average),
        nanos_to_millis(report.setup_time_ns.min),
        nanos_to_millis(report.setup_time_ns.max)
    );
    println!(
        "  run_ms avg={:.3} min={:.3} max={:.3}",
        nanos_to_millis_f64(report.run_time_ns.average),
        nanos_to_millis(report.run_time_ns.min),
        nanos_to_millis(report.run_time_ns.max)
    );
    println!(
        "  total_ms avg={:.3} min={:.3} max={:.3}",
        nanos_to_millis_f64(report.total_time_ns.average),
        nanos_to_millis(report.total_time_ns.min),
        nanos_to_millis(report.total_time_ns.max)
    );
    println!(
        "  generated_tokens total={} avg={:.3}",
        report.generated_token_count.total, report.generated_token_count.average
    );
    if let Some(tokens_per_second) = report.generated_tokens_per_second {
        println!("  generated_tokens_per_second={tokens_per_second:.3}");
    }
    if !report.stop_reasons.is_empty() {
        println!("stop_reasons:");
        for (reason, count) in &report.stop_reasons {
            println!("  {reason}={count}");
        }
    }
    if let Some(transport) = &report.transport_totals {
        println!("transport_totals:");
        println!("  published_packets={}", transport.published_packet_count);
        println!("  received_packets={}", transport.received_packet_count);
        println!("  direct_copies={}", transport.direct_copy_count);
    }
    print_placed_pedal_timing_profile(&report.pedal_timing_summaries, 5);
    Ok(())
}

fn inspect_runtime_topology(
    args: &Args,
    package_manifest: &Path,
    manifest_dir: &Path,
    manifest: VulkanResidentModelPackageManifest,
) -> Result<(), Box<dyn Error>> {
    let default_device_id = args
        .default_device_id
        .as_deref()
        .unwrap_or(RUNTIME_DEFAULT_LOGICAL_DEVICE_ID);
    let available_devices = inspect_available_devices(
        default_device_id,
        runtime_report_default_vulkan_physical_device_index(args),
    );
    let source_graph = manifest.resolved_source_graph(manifest_dir.to_path_buf())?;
    let patch = manifest.runtime_patch_from_controls(
        args.default_device_id.as_deref(),
        &args.pedal_devices,
        &args.duplicate_after,
        args.source_chain.as_deref(),
    )?;
    let effective_graph = source_graph.instantiate_runtime_patch(&patch)?;
    let placement = effective_graph.placement_plan(&patch.placement_spec())?;
    let placement_device_ids = placement_device_ids(&placement.pedals);
    let runtime_routes = runtime_cable_routes_report(args, &placement.cables);
    let device_bindings = runtime_device_bindings_report(args, &placement_device_ids);
    let source_pedals = source_pedals_report(&manifest);
    let payload = RuntimeTopologyReport {
        ok: true,
        schema: RUNTIME_TOPOLOGY_SCHEMA.to_string(),
        package_manifest: package_manifest.to_path_buf(),
        package_root: manifest_dir.to_path_buf(),
        package_id: manifest.package_id.clone(),
        compiled_schema: manifest.schema.clone(),
        config_path: manifest.config_path.clone(),
        tokenizer: serde_json::to_value(&manifest.tokenizer)?,
        available_devices,
        compiled: RuntimeCompiledPedalboardSummary {
            wiring: manifest.circuit_graph.wiring.clone(),
            source_pedal_count: source_pedals.len(),
            source_pedals,
            max_context_activations: manifest.max_context_activations,
        },
        runtime_patch_controls: runtime_patch_report(args),
        runtime_patch: patch,
        effective: RuntimeEffectivePedalboardTopology {
            wiring: placement.wiring,
            pedal_count: placement.pedals.len(),
            cable_count: placement.cables.len(),
            local_cable_count: placement.local_cable_count,
            cross_device_cable_count: placement.cross_device_cable_count,
            device_count: placement_device_ids.len(),
            device_ids: placement_device_ids,
            device_bindings,
            cable_routes: runtime_routes,
            pedals: placement.pedals,
            cables: placement.cables,
        },
    };

    if args.json {
        println!("{}", serde_json::to_string_pretty(&payload)?);
    } else {
        println!("package_id={}", payload.package_id);
        println!("source_pedal_count={}", payload.compiled.source_pedal_count);
        println!("effective_pedal_count={}", payload.effective.pedal_count);
        println!("device_count={}", payload.effective.device_count);
        println!(
            "cross_device_cable_count={}",
            payload.effective.cross_device_cable_count
        );
        println!(
            "same_physical_target_cable_count={}",
            payload
                .effective
                .cable_routes
                .same_physical_target_cable_count
        );
        println!(
            "cross_physical_target_cable_count={}",
            payload
                .effective
                .cable_routes
                .cross_physical_target_cable_count
        );
    }

    Ok(())
}

fn inspect_package(
    args: &Args,
    package_manifest: &Path,
    manifest_dir: &Path,
    manifest: VulkanResidentModelPackageManifest,
) -> Result<(), Box<dyn Error>> {
    let default_device_id = args
        .default_device_id
        .as_deref()
        .unwrap_or(RUNTIME_DEFAULT_LOGICAL_DEVICE_ID);
    let available_devices = inspect_available_devices(
        default_device_id,
        runtime_report_default_vulkan_physical_device_index(args),
    );
    let source_pedals = source_pedals_report(&manifest);
    let source_pedal_count = source_pedals.len();
    let payload = RuntimePackageInspectionReport {
        ok: true,
        package_manifest: package_manifest.to_path_buf(),
        package_root: manifest_dir.to_path_buf(),
        schema: manifest.schema.clone(),
        package_id: manifest.package_id.clone(),
        config_path: manifest.config_path.clone(),
        tokenizer: serde_json::to_value(&manifest.tokenizer)?,
        compiled_wiring: manifest.circuit_graph.wiring.clone(),
        runtime_patch: runtime_patch_report(args),
        device_bindings: runtime_device_bindings_report(args, &[]),
        max_context_activations: manifest.max_context_activations,
        source_pedal_count,
        source_pedals,
        available_devices,
    };

    if args.json {
        println!("{}", serde_json::to_string_pretty(&payload)?);
    } else {
        println!("package_id={}", payload.package_id);
        println!("source_pedal_count={}", payload.source_pedal_count);
        println!("compiled_wiring={}", payload.compiled_wiring);
        for pedal in &payload.source_pedals {
            println!(
                "{} {} kernels={} state_ports={}",
                pedal.pedal_id, pedal.operator_type, pedal.kernel_count, pedal.state_port_count
            );
        }
    }

    Ok(())
}

fn source_pedals_report(manifest: &VulkanResidentModelPackageManifest) -> Vec<RuntimeSourcePedal> {
    let execution_by_pedal = manifest
        .pedal_executions
        .iter()
        .map(|execution| (execution.pedal_id.as_str(), execution))
        .collect::<BTreeMap<_, _>>();

    manifest
        .circuit_graph
        .pedals
        .iter()
        .enumerate()
        .map(|(pedal_index, pedal)| {
            let execution = execution_by_pedal.get(pedal.pedal_id.as_str());
            RuntimeSourcePedal {
                pedal_index,
                pedal_id: pedal.pedal_id.clone(),
                operator_type: pedal.operator_type.clone(),
                runtime_role: pedal.circuit.runtime_role,
                implementation: pedal.implementation.clone(),
                behavioral_role: pedal.behavioral_role.clone(),
                source_layer_index: pedal.circuit.source.source_layer_index,
                circuit_id: pedal.circuit.id.clone(),
                input_ports: pedal
                    .circuit
                    .boundary
                    .inputs
                    .iter()
                    .map(package_port_report)
                    .collect::<Vec<_>>(),
                output_ports: pedal
                    .circuit
                    .boundary
                    .outputs
                    .iter()
                    .map(package_port_report)
                    .collect::<Vec<_>>(),
                state_port_count: pedal.circuit.state_ports.len(),
                parameter_ref_count: pedal.params.refs.len(),
                node_count: pedal.circuit.nodes.len(),
                kernel_count: match pedal.runtime_role {
                    llmoop_runtime::CircuitRuntimeRole::SignalProcessor => execution
                        .map(|execution| execution.kernels.len())
                        .unwrap_or(0),
                    llmoop_runtime::CircuitRuntimeRole::InputTransducer => 1,
                    llmoop_runtime::CircuitRuntimeRole::OutputTransducer => 2,
                    llmoop_runtime::CircuitRuntimeRole::Sampler => manifest.sampler.kernels.len(),
                    llmoop_runtime::CircuitRuntimeRole::DraftProcessor
                    | llmoop_runtime::CircuitRuntimeRole::DraftInputAdapter
                    | llmoop_runtime::CircuitRuntimeRole::DraftOutputTransducer => 0,
                },
            }
        })
        .collect::<Vec<_>>()
}

fn inspect_available_devices(
    default_device_id: &str,
    selected_vulkan_device_index: Option<usize>,
) -> Vec<RuntimeAvailableDevice> {
    discover_runtime_devices(default_device_id, selected_vulkan_device_index)
}

fn inspect_patch(
    args: &Args,
    package_manifest: &Path,
    manifest_dir: &Path,
    manifest: VulkanResidentModelPackageManifest,
) -> Result<(), Box<dyn Error>> {
    let source_graph = manifest.resolved_source_graph(manifest_dir.to_path_buf())?;
    let patch = manifest.runtime_patch_from_controls(
        args.default_device_id.as_deref(),
        &args.pedal_devices,
        &args.duplicate_after,
        args.source_chain.as_deref(),
    )?;
    let effective_graph = source_graph.instantiate_runtime_patch(&patch)?;
    let placement = effective_graph.placement_plan(&patch.placement_spec())?;
    let placement_device_ids = placement_device_ids(&placement.pedals);
    let instance_count = patch.instances.len();
    let cable_count = placement.cables.len();
    let payload = RuntimePatchInspectionReport {
        ok: true,
        package_manifest: package_manifest.to_path_buf(),
        package_root: manifest_dir.to_path_buf(),
        package_id: manifest.package_id.clone(),
        compiled_source_pedal_count: source_graph.circuits.len(),
        runtime_patch_controls: runtime_patch_report(args),
        runtime_patch: patch,
        device_bindings: runtime_device_bindings_report(args, &placement_device_ids),
        effective_pedal_count: instance_count,
        effective_cable_count: cable_count,
        placement: RuntimePatchPlacementReport {
            schema: placement.schema,
            wiring: placement.wiring,
            local_cable_count: placement.local_cable_count,
            cross_device_cable_count: placement.cross_device_cable_count,
            runtime_routes: runtime_cable_routes_report(args, &placement.cables),
            pedals: placement.pedals,
            cables: placement.cables,
        },
    };

    if args.json {
        println!("{}", serde_json::to_string_pretty(&payload)?);
    } else {
        println!("package_id={}", payload.package_id);
        println!("effective_pedal_count={}", payload.effective_pedal_count);
        println!("effective_cable_count={}", payload.effective_cable_count);
        println!(
            "cross_device_cable_count={}",
            payload.placement.cross_device_cable_count
        );
        for pedal in &payload.placement.pedals {
            println!(
                "{} circuit={} device={}",
                pedal.pedal_id, pedal.circuit_id, pedal.device_id
            );
        }
    }

    Ok(())
}

fn package_port_report(port: &CircuitPort) -> RuntimePedalPortSummary {
    RuntimePedalPortSummary {
        id: port.id.clone(),
        signal: port.signal.clone(),
        shape: port.shape.clone(),
        source: port.source.clone(),
        pedal_port: port.pedal_port.clone(),
    }
}

fn inspect_device_slice(
    args: &Args,
    package_manifest: &Path,
    manifest_dir: &Path,
    runtime_model: VulkanResidentRuntimeModel,
    device_id: &str,
) -> Result<(), Box<dyn Error>> {
    let capacity = choose_runtime_context_size(package_manifest, args.context_size, 1)?;
    let logical_device_ids = vec![device_id.to_string()];
    let bound_devices = runtime_bound_vulkan_devices(args, &logical_device_ids)?;
    let device = bound_devices.devices.get(device_id).ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("logical device {device_id:?} was not mounted"),
        )
    })?;
    let payload = inspect_device_slice_payload(
        device,
        package_manifest,
        manifest_dir,
        runtime_model,
        device_id,
        capacity,
    )?;

    if args.json {
        println!("{}", serde_json::to_string_pretty(&payload)?);
    } else {
        println!("device_id={}", payload.device_id);
        println!("hosted_pedal_count={}", payload.hosted_pedal_count);
        println!("incoming_cable_count={}", payload.incoming_cable_count);
        println!("outgoing_cable_count={}", payload.outgoing_cable_count);
        println!("dispatch_count={}", payload.dispatch_count);
        println!("descriptor_count={}", payload.descriptor_count);
        println!("tick_stage_count={}", payload.tick_plan.stage_count);
    }

    Ok(())
}

fn inspect_placement(
    args: &Args,
    package_manifest: &Path,
    manifest_dir: &Path,
    runtime_model: VulkanResidentRuntimeModel,
) -> Result<(), Box<dyn Error>> {
    let capacity = choose_runtime_context_size(package_manifest, args.context_size, 1)?;
    let device_ids = runtime_model.placement_device_ids();
    let placement = runtime_model_placement(manifest_dir, &runtime_model)?;
    let bound_devices = runtime_bound_vulkan_devices(args, &device_ids)?;
    let slices = device_ids
        .iter()
        .map(|device_id| {
            let device = bound_devices.devices.get(device_id).ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!("logical device {device_id:?} was not mounted"),
                )
            })?;
            inspect_device_slice_payload(
                device,
                package_manifest,
                manifest_dir,
                runtime_model.clone(),
                device_id,
                capacity,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    let payload = RuntimePlacementReport {
        ok: true,
        package_manifest: package_manifest.to_path_buf(),
        context_window_activations: capacity,
        runtime_patch: runtime_patch_report(args),
        device_bindings: runtime_device_bindings_report(args, &device_ids),
        bound_devices: bound_devices_report(&bound_devices),
        cable_routes: bound_cable_routes_report(&bound_devices, &placement.cables),
        device_count: device_ids.len(),
        device_ids,
        devices: slices,
    };

    if args.json {
        println!("{}", serde_json::to_string_pretty(&payload)?);
    } else {
        println!("device_count={}", payload.device_count);
        for device in &payload.devices {
            println!(
                "{} pedals={} incoming={} outgoing={} dispatches={}",
                device.device_id,
                device.hosted_pedal_count,
                device.incoming_cable_count,
                device.outgoing_cable_count,
                device.dispatch_count
            );
        }
    }

    Ok(())
}

fn inspect_device_slice_payload(
    device: &VulkanComputeDevice,
    package_manifest: &Path,
    manifest_dir: &Path,
    runtime_model: VulkanResidentRuntimeModel,
    device_id: &str,
    capacity: usize,
) -> Result<RuntimeDeviceSliceReport, Box<dyn Error>> {
    let slice = VulkanResidentModelPackageDeviceSlice::from_runtime_model_for_device(
        device,
        manifest_dir,
        runtime_model,
        device_id,
        Some(capacity),
    )?;
    let mounted = slice.create_mounted_stream_circuit(device)?;
    let reusable_manifest = VulkanReusableKernelArtifactManifest::new(
        slice
            .loaded_manifest()
            .artifacts
            .iter()
            .map(|artifact| artifact.artifact.clone())
            .collect(),
    );
    let mounted_bound = mounted.mounted_placed_bound_dispatch_plan(&reusable_manifest)?;
    let tick_plan = mounted.stream_tick_plan(&reusable_manifest)?;
    let resident_plan = &mounted.placed_plan.placed_resident_plan;
    let loaded_kernel_artifact_count = slice.loaded_manifest().artifacts.len();

    Ok(RuntimeDeviceSliceReport {
        ok: true,
        package_manifest: package_manifest.to_path_buf(),
        device_name: device.device_name().to_string(),
        device_id: slice.device_id,
        context_window_activations: capacity,
        hosted_pedals: resident_plan.hosted_pedal_ids.clone(),
        local_cables: resident_plan
            .local_cables
            .iter()
            .map(|cable| RuntimeLocalCableBufferReport {
                cable_index: cable.cable_index,
                signal: cable.signal.clone(),
                source_pedal_id: cable.source_pedal_id.clone(),
                destination_pedal_id: cable.destination_pedal_id.clone(),
                device_id: cable.source_device_id.clone(),
                byte_capacity: mounted
                    .cable_io
                    .local_cable_buffer(cable.cable_index)
                    .map(|buffer| buffer.byte_capacity),
            })
            .collect::<Vec<_>>(),
        incoming_cables: resident_plan
            .incoming_cables
            .iter()
            .map(|cable| RuntimeRemoteCableBufferReport {
                cable_index: cable.cable_index,
                signal: cable.signal.clone(),
                source_device_id: cable.source_device_id.clone(),
                source_pedal_id: cable.source_pedal_id.clone(),
                destination_device_id: cable.destination_device_id.clone(),
                destination_pedal_id: cable.destination_pedal_id.clone(),
                byte_capacity: mounted
                    .cable_io
                    .incoming_buffer(cable.cable_index)
                    .map(|buffer| buffer.byte_capacity),
            })
            .collect::<Vec<_>>(),
        outgoing_cables: resident_plan
            .outgoing_cables
            .iter()
            .map(|cable| RuntimeRemoteCableBufferReport {
                cable_index: cable.cable_index,
                signal: cable.signal.clone(),
                source_device_id: cable.source_device_id.clone(),
                source_pedal_id: cable.source_pedal_id.clone(),
                destination_device_id: cable.destination_device_id.clone(),
                destination_pedal_id: cable.destination_pedal_id.clone(),
                byte_capacity: mounted
                    .cable_io
                    .outgoing_buffer(cable.cable_index)
                    .map(|buffer| buffer.byte_capacity),
            })
            .collect::<Vec<_>>(),
        hosted_pedal_count: slice.hosted_pedal_count,
        incoming_cable_count: slice.incoming_cable_count,
        outgoing_cable_count: slice.outgoing_cable_count,
        permanent_parameter_count: slice.permanent_parameter_count,
        permanent_parameter_bytes: slice.permanent_parameter_bytes,
        reusable_kernel_word_count: slice.reusable_kernel_word_count,
        loaded_kernel_artifact_count,
        dispatch_count: mounted_bound.dispatches.len(),
        descriptor_count: mounted_bound.total_descriptor_count,
        model_boundary_descriptor_count: mounted_bound.model_boundary_descriptor_count,
        incoming_cable_descriptor_count: mounted_bound.incoming_cable_descriptor_count,
        outgoing_cable_descriptor_count: mounted_bound.outgoing_cable_descriptor_count,
        tick_plan: RuntimeDeviceTickPlanReport {
            stage_count: tick_plan.stage_count,
            receive_stage_count: tick_plan.receive_stage_count,
            dispatch_stage_count: tick_plan.dispatch_stage_count,
            publish_stage_count: tick_plan.publish_stage_count,
            local_cable_read_count: tick_plan.local_cable_read_count,
            local_cable_write_count: tick_plan.local_cable_write_count,
            incoming_cable_read_count: tick_plan.incoming_cable_read_count,
            outgoing_cable_write_count: tick_plan.outgoing_cable_write_count,
            model_input_read_count: tick_plan.model_input_read_count,
            model_output_write_count: tick_plan.model_output_write_count,
            can_execute: tick_plan.can_execute,
        },
    })
}

fn placement_device_ids(pedals: &[PedalPlacement]) -> Vec<String> {
    let mut device_ids = pedals
        .iter()
        .map(|pedal| pedal.device_id.clone())
        .collect::<Vec<_>>();
    device_ids.sort();
    device_ids.dedup();
    device_ids
}

fn runtime_model_placement(
    manifest_dir: &Path,
    runtime_model: &VulkanResidentRuntimeModel,
) -> Result<llmoop_runtime::StreamCircuitPlacementPlan, Box<dyn Error>> {
    let graph = runtime_model.resolved_graph(manifest_dir.to_path_buf())?;
    graph
        .placement_plan(&runtime_model.placement)
        .map_err(|error| Box::new(error) as Box<dyn Error>)
}

fn tokenizer_dir_from_package(package_manifest: &Path) -> Result<PathBuf, Box<dyn Error>> {
    let manifest = VulkanResidentModelPackageManifest::from_json_file(package_manifest)?;
    let manifest_dir = package_manifest
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."));
    let tokenizer_dir = resolve_package_path(&manifest_dir, &manifest.tokenizer.path);
    if !tokenizer_dir.join("tokenizer.json").is_file() {
        return Err(Box::new(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "compiled package declares tokenizer at {}, but tokenizer.json is missing",
                tokenizer_dir.display()
            ),
        )));
    }
    Ok(tokenizer_dir)
}

fn resolve_package_path(manifest_dir: &Path, raw_path: &str) -> PathBuf {
    let path = Path::new(raw_path);
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        manifest_dir.join(path)
    }
}

fn runtime_model(
    args: &Args,
    package_manifest: &Path,
) -> Result<VulkanResidentRuntimeModel, Box<dyn Error>> {
    let manifest = VulkanResidentModelPackageManifest::from_json_file(package_manifest)?;
    Ok(manifest.mount_runtime_patch_controls(
        args.default_device_id.as_deref(),
        &args.pedal_devices,
        &args.duplicate_after,
        args.source_chain.as_deref(),
    )?)
}

struct RuntimeBoundVulkanDevices {
    devices: BTreeMap<String, Rc<VulkanComputeDevice>>,
    physical_device_indices: BTreeMap<String, usize>,
    physical_device_ids: BTreeMap<String, String>,
}

fn runtime_physical_device_bindings_in(
    args: &Args,
    logical_device_ids: &[String],
    available_devices: &[VulkanComputeDeviceInfo],
) -> Result<BTreeMap<String, usize>, io::Error> {
    let default_physical_device_index = if let Some(index) = args.vulkan_device_index {
        index
    } else {
        available_devices
            .iter()
            .find(|device| device.selected_by_default)
            .or_else(|| available_devices.first())
            .map(|device| device.physical_device_index)
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::NotFound,
                    "no Vulkan compute-capable physical devices are available",
                )
            })?
    };
    let mut logical_device_ids = logical_device_ids.to_vec();
    logical_device_ids.sort();
    logical_device_ids.dedup();
    logical_device_ids
        .into_iter()
        .map(|logical_device_id| {
            let physical_device_index = runtime_mount_physical_device_index(
                args,
                &logical_device_id,
                default_physical_device_index,
                available_devices,
            )?;
            Ok((logical_device_id, physical_device_index))
        })
        .collect()
}

fn runtime_bound_vulkan_devices(
    args: &Args,
    logical_device_ids: &[String],
) -> Result<RuntimeBoundVulkanDevices, Box<dyn Error>> {
    let device_catalog = VulkanComputeDeviceCatalog::discover()?;
    let available_devices = device_catalog.available_compute_devices();
    let requested_bindings =
        runtime_physical_device_bindings_in(args, logical_device_ids, available_devices)?;
    let mut devices = BTreeMap::new();
    let mut physical_devices: BTreeMap<usize, Rc<VulkanComputeDevice>> = BTreeMap::new();
    let mut physical_device_indices = BTreeMap::new();
    let mut physical_device_ids = BTreeMap::new();

    for (logical_device_id, physical_device_index) in requested_bindings {
        let available_device = available_devices
            .iter()
            .find(|device| device.physical_device_index == physical_device_index)
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::NotFound,
                    format!(
                        "Vulkan physical device index {physical_device_index} is not available"
                    ),
                )
            })?;
        let device = if let Some(device) = physical_devices.get(&physical_device_index) {
            Rc::clone(device)
        } else {
            let device = Rc::new(device_catalog.open_device_uuid(available_device.device_uuid)?);
            physical_devices.insert(physical_device_index, Rc::clone(&device));
            device
        };
        devices.insert(logical_device_id.clone(), device);
        physical_device_indices.insert(logical_device_id.clone(), physical_device_index);
        physical_device_ids.insert(
            logical_device_id.clone(),
            available_device.physical_device_id.clone(),
        );
    }

    Ok(RuntimeBoundVulkanDevices {
        devices,
        physical_device_indices,
        physical_device_ids,
    })
}

fn runtime_default_vulkan_physical_device_index() -> Result<usize, Box<dyn Error>> {
    let devices = VulkanComputeDevice::available_compute_devices()?;
    devices
        .iter()
        .find(|device| device.selected_by_default)
        .or_else(|| devices.first())
        .map(|device| device.physical_device_index)
        .ok_or_else(|| {
            Box::new(io::Error::new(
                io::ErrorKind::NotFound,
                "no Vulkan compute-capable physical devices are available",
            )) as Box<dyn Error>
        })
}

fn bound_devices_report(bound_devices: &RuntimeBoundVulkanDevices) -> Vec<RuntimeBoundDevice> {
    bound_devices
        .devices
        .iter()
        .map(|(logical_device_id, device)| {
            let physical_device_index = bound_devices
                .physical_device_indices
                .get(logical_device_id)
                .copied();
            RuntimeBoundDevice {
                device_id: logical_device_id.clone(),
                target: bound_devices
                    .physical_device_ids
                    .get(logical_device_id)
                    .cloned(),
                physical_device_index,
                device_name: device.device_name().to_string(),
            }
        })
        .collect::<Vec<_>>()
}

fn runtime_cable_routes_report(args: &Args, cables: &[PedalCablePlacement]) -> RuntimeCableRoutes {
    RuntimeCableRoutes::from_cables(cables, |device_id| {
        runtime_target_for_logical_device(args, device_id)
    })
}

fn bound_cable_routes_report(
    bound_devices: &RuntimeBoundVulkanDevices,
    cables: &[PedalCablePlacement],
) -> RuntimeCableRoutes {
    RuntimeCableRoutes::from_cables(cables, |device_id| {
        let physical_device_index = bound_devices
            .physical_device_indices
            .get(device_id)
            .copied();
        RuntimeCableRouteTarget {
            target: bound_devices.physical_device_ids.get(device_id).cloned(),
            physical_device_index,
            binding_source: "mounted".to_string(),
        }
    })
}

fn runtime_mount_physical_device_index(
    args: &Args,
    logical_device_id: &str,
    default_physical_device_index: usize,
    available_devices: &[VulkanComputeDeviceInfo],
) -> Result<usize, io::Error> {
    if let Some(target) = args.device_bindings.get(logical_device_id) {
        return resolve_runtime_vulkan_physical_device_ref_in(target, available_devices)
            .map_err(|error| io::Error::new(io::ErrorKind::InvalidInput, error))?
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!(
                        "logical device {logical_device_id:?} is bound to unsupported target {target:?}; local mounted execution supports vulkan:N or cpuN targets"
                    ),
                )
            });
    }
    match resolve_runtime_vulkan_physical_device_ref_in(logical_device_id, available_devices)
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidInput, error))?
    {
        Some(index) => Ok(index),
        None if logical_device_id.contains(':') => Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "logical device id {logical_device_id:?} looks like an unsupported direct runtime target; local mounted execution supports vulkan:N or cpuN targets"
            ),
        )),
        None => Ok(default_physical_device_index),
    }
}

fn runtime_target_for_logical_device(
    args: &Args,
    logical_device_id: &str,
) -> RuntimeCableRouteTarget {
    if let Some(target) = args.device_bindings.get(logical_device_id) {
        let physical_device_index = resolve_runtime_vulkan_physical_device_ref(target)
            .ok()
            .flatten();
        return RuntimeCableRouteTarget {
            target: Some(target.clone()),
            physical_device_index,
            binding_source: "explicit".to_string(),
        };
    }
    match resolve_runtime_vulkan_physical_device_ref(logical_device_id) {
        Ok(Some(index)) => RuntimeCableRouteTarget {
            target: Some(logical_device_id.to_string()),
            physical_device_index: Some(index),
            binding_source: "device_id".to_string(),
        },
        Ok(None) | Err(_) if logical_device_id.contains(':') => RuntimeCableRouteTarget {
            target: Some(logical_device_id.to_string()),
            physical_device_index: None,
            binding_source: "device_id".to_string(),
        },
        Ok(None) | Err(_) => {
            let default_physical_device_index =
                runtime_report_default_vulkan_physical_device_index(args);
            let target = default_physical_device_index.map(|index| format!("vulkan:{index}"));
            RuntimeCableRouteTarget {
                physical_device_index: default_physical_device_index,
                target,
                binding_source: if args.vulkan_device_index.is_some() {
                    "process_default".to_string()
                } else {
                    "runtime_default".to_string()
                },
            }
        }
    }
}

fn runtime_report_default_vulkan_physical_device_index(args: &Args) -> Option<usize> {
    args.vulkan_device_index
        .or_else(|| {
            args.default_device_id.as_deref().and_then(|device_id| {
                resolve_runtime_vulkan_physical_device_ref(device_id)
                    .ok()
                    .flatten()
            })
        })
        .or_else(|| runtime_default_vulkan_physical_device_index().ok())
}

fn elapsed_nanos_u64(start: Instant) -> u64 {
    u64::try_from(start.elapsed().as_nanos()).unwrap_or(u64::MAX)
}

fn average_nanos(total_ns: u64, count: usize) -> Option<u64> {
    if count == 0 {
        None
    } else {
        Some(total_ns / u64::try_from(count).unwrap_or(u64::MAX))
    }
}

fn runtime_prompt_timing_report(
    setup_time_ns: u64,
    run_time_ns: u64,
    generated_token_count: usize,
    tick_count: usize,
    scheduler_turn_count: usize,
) -> RuntimePromptTimingReport {
    RuntimePromptTimingReport {
        setup_time_ns,
        run_time_ns,
        total_time_ns: setup_time_ns.saturating_add(run_time_ns),
        generated_token_count,
        tick_count,
        scheduler_turn_count,
        average_generated_token_time_ns: average_nanos(run_time_ns, generated_token_count),
        average_tick_time_ns: average_nanos(run_time_ns, tick_count),
        average_scheduler_turn_time_ns: average_nanos(run_time_ns, scheduler_turn_count),
    }
}

fn tokenizer_options_report(args: &Args) -> RuntimeTokenizerOptionsReport {
    RuntimeTokenizerOptionsReport {
        add_special_tokens: args.add_special_tokens,
        skip_special_tokens: args.skip_special_tokens,
    }
}

fn runtime_patch_report(args: &Args) -> RuntimePatchControls {
    RuntimePatchControls {
        default_device_id: args.default_device_id.clone(),
        pedal_devices: args.pedal_devices.clone(),
        source_chain: args.source_chain.as_ref().map(|source_chain| {
            source_chain
                .iter()
                .map(
                    |(instance_id, source_pedal_id)| RuntimePatchSourceChainEntry {
                        instance_id: instance_id.clone(),
                        source_pedal_id: source_pedal_id.clone(),
                    },
                )
                .collect::<Vec<_>>()
        }),
        duplicate_after: args
            .duplicate_after
            .iter()
            .map(
                |(after_instance_id, new_instance_id)| RuntimePatchDuplicateAfterControl {
                    after_instance_id: after_instance_id.clone(),
                    new_instance_id: new_instance_id.clone(),
                },
            )
            .collect::<Vec<_>>(),
    }
}

fn runtime_device_bindings_report(
    args: &Args,
    logical_device_ids: &[String],
) -> RuntimeDeviceBindings {
    let all_logical_devices_are_explicitly_bound = logical_device_ids
        .iter()
        .all(|device_id| args.device_bindings.contains_key(device_id));
    let default_physical_device_index = if all_logical_devices_are_explicitly_bound {
        args.vulkan_device_index.or_else(|| {
            args.default_device_id
                .as_deref()
                .and_then(|device_id| parse_vulkan_physical_device_ref(device_id).ok().flatten())
        })
    } else {
        runtime_report_default_vulkan_physical_device_index(args)
    };
    RuntimeDeviceBindings::from_vulkan_targets(
        logical_device_ids,
        &args.device_bindings,
        default_physical_device_index,
        resolve_runtime_vulkan_physical_device_ref,
    )
}

fn choose_runtime_context_size(
    package_manifest: &Path,
    requested_context_size: Option<usize>,
    minimum_context_size: usize,
) -> Result<usize, Box<dyn Error>> {
    let manifest = VulkanResidentModelPackageManifest::from_json_file(package_manifest)?;
    Ok(resolve_runtime_context_size(
        manifest.max_context_activations,
        requested_context_size,
        minimum_context_size,
    )?)
}

fn resolve_runtime_context_size(
    max_context_size: usize,
    requested_context_size: Option<usize>,
    minimum_context_size: usize,
) -> io::Result<usize> {
    if max_context_size == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "compiled package declares a zero maximum context size",
        ));
    }

    let context_size = requested_context_size.unwrap_or(max_context_size);
    if context_size > max_context_size {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "requested context size {context_size} exceeds the model maximum ({max_context_size})"
            ),
        ));
    }
    if context_size < minimum_context_size {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "context size {context_size} cannot hold the {minimum_context_size}-token prompt"
            ),
        ));
    }

    Ok(context_size)
}

fn choose_chat_runtime_context_size(
    package_manifest: &Path,
    requested_context_size: Option<usize>,
) -> Result<usize, Box<dyn Error>> {
    choose_runtime_context_size(package_manifest, requested_context_size, 0)
}

fn parse_args() -> Result<Args, String> {
    let mut parsed = Args::default();
    let mut raw = std::env::args().skip(1);

    while let Some(arg) = raw.next() {
        match arg.as_str() {
            "--package" => {
                parsed.package_manifest = Some(PathBuf::from(next_value(&mut raw, &arg)?));
            }
            "--prompt" => {
                parsed.prompt = Some(next_value(&mut raw, "--prompt")?);
            }
            "--chat" => {
                parsed.chat = true;
            }
            "--inspect-runtime" => {
                parsed.inspect_runtime = true;
            }
            "--inspect-package" => {
                parsed.inspect_package = true;
            }
            "--inspect-patch" => {
                parsed.inspect_patch = true;
            }
            "--inspect-placement" => {
                parsed.inspect_placement = true;
            }
            "--inspect-device-slice" => {
                parsed.inspect_device_slice = Some(next_value(&mut raw, "--inspect-device-slice")?);
            }
            "--device" => {
                parsed.default_device_id = Some(next_value(&mut raw, &arg)?);
            }
            "--place-pedal" => {
                let assignment = next_value(&mut raw, &arg)?;
                let (pedal_id, device_id) = parse_pedal_device_assignment(&assignment)?;
                if parsed
                    .pedal_devices
                    .insert(pedal_id.clone(), device_id)
                    .is_some()
                {
                    return Err(format!(
                        "duplicate runtime placement for pedal {pedal_id:?}"
                    ));
                }
            }
            "--bind-device" => {
                let assignment = next_value(&mut raw, &arg)?;
                let (device_id, target) = parse_device_binding_assignment(&assignment)?;
                if parsed
                    .device_bindings
                    .insert(device_id.clone(), target)
                    .is_some()
                {
                    return Err(format!(
                        "duplicate runtime device binding for logical device {device_id:?}"
                    ));
                }
            }
            "--duplicate-after" => {
                let assignment = next_value(&mut raw, "--duplicate-after")?;
                parsed
                    .duplicate_after
                    .push(parse_duplicate_after_assignment(&assignment)?);
            }
            "--chain" => {
                let chain = parse_source_chain(&next_value(&mut raw, &arg)?)?;
                if parsed.source_chain.replace(chain).is_some() {
                    return Err("--chain may only be supplied once".to_string());
                }
            }
            "--max-new-tokens" => {
                parsed.max_new_tokens = parse_next(&mut raw, "--max-new-tokens")?;
            }
            "--speculative-draft-tokens" => {
                parsed.speculative_draft_tokens =
                    parse_next(&mut raw, "--speculative-draft-tokens")?;
            }
            "--context-size" => {
                parsed.context_size = Some(parse_next(&mut raw, "--context-size")?);
            }
            "--vulkan-device-index" => {
                parsed.vulkan_device_index = Some(parse_next(&mut raw, "--vulkan-device-index")?);
            }
            "--seed" => {
                parsed.random_seed = parse_next(&mut raw, "--seed")?;
            }
            "--no-special-tokens" => {
                parsed.add_special_tokens = false;
            }
            "--keep-special-tokens" => {
                parsed.skip_special_tokens = false;
            }
            "--generated-only" => {
                parsed.generated_only = true;
            }
            "--profile" => {
                parsed.profile = true;
            }
            "--profile-runs" => {
                parsed.profile_runs = parse_next(&mut raw, "--profile-runs")?;
            }
            "--json" => {
                parsed.json = true;
            }
            _ => {
                return Err(format!("unknown argument {arg:?}\n\n{}", usage()));
            }
        }
    }

    if matches!(parsed.prompt.as_deref(), Some("")) {
        return Err("--prompt must not be empty".to_string());
    }
    let inspect_mode_count = usize::from(parsed.inspect_runtime)
        + usize::from(parsed.inspect_package)
        + usize::from(parsed.inspect_patch)
        + usize::from(parsed.inspect_placement)
        + usize::from(parsed.inspect_device_slice.is_some());
    if inspect_mode_count > 1 {
        return Err(
            "--inspect-runtime, --inspect-package, --inspect-patch, --inspect-placement, and --inspect-device-slice are mutually exclusive"
                .to_string(),
        );
    }
    if parsed.chat && inspect_mode_count > 0 {
        return Err("--chat cannot be combined with inspect modes".to_string());
    }
    if parsed.chat && parsed.profile {
        return Err("--profile is not supported with --chat".to_string());
    }
    if parsed.chat && parsed.profile_runs != 1 {
        return Err("--profile-runs is not supported with --chat".to_string());
    }
    if parsed.chat && parsed.json {
        return Err("--json is not supported with --chat yet".to_string());
    }
    if matches!(parsed.inspect_device_slice.as_deref(), Some("")) {
        return Err("--inspect-device-slice must not be empty".to_string());
    }
    if matches!(parsed.default_device_id.as_deref(), Some("")) {
        return Err("--device must not be empty".to_string());
    }
    if parsed.max_new_tokens == 0 {
        return Err("--max-new-tokens must be at least 1".to_string());
    }
    if matches!(parsed.context_size, Some(0)) {
        return Err("--context-size must be at least 1".to_string());
    }
    if parsed.profile_runs == 0 {
        return Err("--profile-runs must be at least 1".to_string());
    }

    Ok(parsed)
}

fn parse_pedal_device_assignment(raw: &str) -> Result<(String, String), String> {
    let (pedal_id, device_id) = raw
        .split_once('=')
        .ok_or_else(|| format!("invalid runtime placement {raw:?}; expected PEDAL_ID=DEVICE_ID"))?;
    let pedal_id = pedal_id.trim();
    let device_id = device_id.trim();
    if pedal_id.is_empty() {
        return Err(format!(
            "invalid runtime placement {raw:?}; pedal id must not be empty"
        ));
    }
    if device_id.is_empty() {
        return Err(format!(
            "invalid runtime placement {raw:?}; device id must not be empty"
        ));
    }
    Ok((pedal_id.to_string(), device_id.to_string()))
}

fn parse_device_binding_assignment(raw: &str) -> Result<(String, String), String> {
    let (device_id, target) = raw.split_once('=').ok_or_else(|| {
        format!("invalid runtime device binding {raw:?}; expected LOGICAL_DEVICE_ID=TARGET")
    })?;
    let device_id = device_id.trim();
    let target = target.trim();
    if device_id.is_empty() {
        return Err(format!(
            "invalid runtime device binding {raw:?}; logical device id must not be empty"
        ));
    }
    if target.is_empty() {
        return Err(format!(
            "invalid runtime device binding {raw:?}; target must not be empty"
        ));
    }
    validate_runtime_device_target_syntax(target)?;
    Ok((device_id.to_string(), target.to_string()))
}

fn validate_runtime_device_target_syntax(raw: &str) -> Result<(), String> {
    if raw.starts_with("vulkan-uuid:") {
        parse_vulkan_device_uuid_ref(raw)?;
    } else if raw.starts_with("vulkan") {
        if parse_vulkan_physical_device_ref(raw)?.is_none() {
            return Err(format!(
                "invalid Vulkan physical device reference {raw:?}; expected vulkan:N"
            ));
        }
    } else if raw.starts_with("cpu") {
        parse_cpu_runtime_device_ref(raw)?;
    }
    Ok(())
}

fn resolve_runtime_vulkan_physical_device_ref(raw: &str) -> Result<Option<usize>, String> {
    if let Some(index) = parse_vulkan_physical_device_ref(raw)? {
        return Ok(Some(index));
    }
    let device_uuid = parse_vulkan_device_uuid_ref(raw)?;
    let cpu_ordinal = parse_cpu_runtime_device_ref(raw)?;
    if device_uuid.is_none() && cpu_ordinal.is_none() {
        return Ok(None);
    }
    let available_devices = VulkanComputeDevice::available_compute_devices()
        .map_err(|error| format!("failed to discover Vulkan devices: {error}"))?;
    resolve_runtime_vulkan_physical_device_ref_in(raw, &available_devices)
}

fn resolve_runtime_vulkan_physical_device_ref_in(
    raw: &str,
    available_devices: &[VulkanComputeDeviceInfo],
) -> Result<Option<usize>, String> {
    if let Some(index) = parse_vulkan_physical_device_ref(raw)? {
        return Ok(Some(index));
    }
    if let Some(device_uuid) = parse_vulkan_device_uuid_ref(raw)? {
        return available_devices
            .iter()
            .find(|device| device.device_uuid == device_uuid)
            .map(|device| Some(device.physical_device_index))
            .ok_or_else(|| format!("Vulkan device reference {raw:?} is not available"));
    }
    if let Some(cpu_ordinal) = parse_cpu_runtime_device_ref(raw)? {
        return available_devices
            .iter()
            .filter(|device| device.device_type == "cpu")
            .nth(cpu_ordinal)
            .map(|device| Some(device.physical_device_index))
            .ok_or_else(|| format!("CPU runtime device cpu{cpu_ordinal} is not available"));
    }
    Ok(None)
}

fn parse_vulkan_device_uuid_ref(raw: &str) -> Result<Option<[u8; 16]>, String> {
    let Some(encoded) = raw.strip_prefix("vulkan-uuid:") else {
        return Ok(None);
    };
    if encoded.len() != 32 {
        return Err(format!(
            "invalid Vulkan device UUID reference {raw:?}; expected vulkan-uuid followed by 32 hexadecimal digits"
        ));
    }
    let mut device_uuid = [0u8; 16];
    for (index, byte) in device_uuid.iter_mut().enumerate() {
        let offset = index * 2;
        *byte = u8::from_str_radix(&encoded[offset..offset + 2], 16)
            .map_err(|error| format!("invalid Vulkan device UUID reference {raw:?}: {error}"))?;
    }
    Ok(Some(device_uuid))
}

fn parse_vulkan_physical_device_ref(raw: &str) -> Result<Option<usize>, String> {
    if let Some(index) = raw.strip_prefix("vulkan:") {
        if index.is_empty() {
            return Err(format!(
                "invalid Vulkan physical device reference {raw:?}; expected vulkan:N"
            ));
        }
        return index
            .parse::<usize>()
            .map(Some)
            .map_err(|error| format!("invalid Vulkan physical device reference {raw:?}: {error}"));
    }
    Ok(None)
}

fn parse_cpu_runtime_device_ref(raw: &str) -> Result<Option<usize>, String> {
    if raw == "cpu" {
        return Ok(Some(0));
    }
    if let Some(index) = raw.strip_prefix("cpu:") {
        if index.is_empty() {
            return Err(format!(
                "invalid CPU runtime device reference {raw:?}; expected cpuN or cpu:N"
            ));
        }
        return index
            .parse::<usize>()
            .map(Some)
            .map_err(|error| format!("invalid CPU runtime device reference {raw:?}: {error}"));
    }
    if let Some(index) = raw.strip_prefix("cpu") {
        if index.is_empty() {
            return Err(format!(
                "invalid CPU runtime device reference {raw:?}; expected cpuN or cpu:N"
            ));
        }
        if index.chars().all(|ch| ch.is_ascii_digit()) {
            return index
                .parse::<usize>()
                .map(Some)
                .map_err(|error| format!("invalid CPU runtime device reference {raw:?}: {error}"));
        }
        return Err(format!(
            "invalid CPU runtime device reference {raw:?}; expected cpuN or cpu:N"
        ));
    }
    Ok(None)
}

fn parse_duplicate_after_assignment(raw: &str) -> Result<(String, String), String> {
    let (after_instance_id, new_instance_id) = raw.split_once('=').ok_or_else(|| {
        format!("invalid runtime duplicate {raw:?}; expected AFTER_INSTANCE_ID=NEW_INSTANCE_ID")
    })?;
    let after_instance_id = after_instance_id.trim();
    let new_instance_id = new_instance_id.trim();
    if after_instance_id.is_empty() {
        return Err(format!(
            "invalid runtime duplicate {raw:?}; source instance id must not be empty"
        ));
    }
    if new_instance_id.is_empty() {
        return Err(format!(
            "invalid runtime duplicate {raw:?}; new instance id must not be empty"
        ));
    }
    Ok((after_instance_id.to_string(), new_instance_id.to_string()))
}

fn parse_source_chain(raw: &str) -> Result<Vec<(String, String)>, String> {
    let separator = if raw.contains("->") { "->" } else { "," };
    let mut chain = Vec::new();
    let mut instance_ids = std::collections::BTreeSet::new();

    for raw_item in raw.split(separator) {
        let raw_item = raw_item.trim();
        if raw_item.is_empty() {
            return Err(format!(
                "invalid runtime chain {raw:?}; chain items must not be empty"
            ));
        }
        let (instance_id, source_pedal_id) =
            if let Some((instance_id, source_pedal_id)) = raw_item.split_once('=') {
                (instance_id.trim(), source_pedal_id.trim())
            } else {
                (raw_item, raw_item)
            };
        if instance_id.is_empty() {
            return Err(format!(
                "invalid runtime chain item {raw_item:?}; instance id must not be empty"
            ));
        }
        if source_pedal_id.is_empty() {
            return Err(format!(
                "invalid runtime chain item {raw_item:?}; source pedal id must not be empty"
            ));
        }
        if !instance_ids.insert(instance_id.to_string()) {
            return Err(format!(
                "invalid runtime chain {raw:?}; duplicate instance id {instance_id:?}"
            ));
        }
        chain.push((instance_id.to_string(), source_pedal_id.to_string()));
    }

    if chain.is_empty() {
        return Err("runtime chain must contain at least one pedal".to_string());
    }

    Ok(chain)
}

fn parse_next<T: std::str::FromStr>(
    raw: &mut impl Iterator<Item = String>,
    flag: &str,
) -> Result<T, String>
where
    T::Err: std::fmt::Display,
{
    let value = next_value(raw, flag)?;
    value
        .parse::<T>()
        .map_err(|error| format!("invalid value {value:?} for {flag}: {error}"))
}

fn next_value(raw: &mut impl Iterator<Item = String>, flag: &str) -> Result<String, String> {
    raw.next()
        .ok_or_else(|| format!("{flag} requires a value\n\n{}", usage()))
}

fn print_text(text: &str) {
    print!("{text}");
    if !text.ends_with('\n') {
        println!();
    }
}

fn print_prompt_timing_profile(timing: &RuntimePromptTimingReport) {
    println!("profile:");
    println!("  setup_ms={:.3}", nanos_to_millis(timing.setup_time_ns));
    println!("  run_ms={:.3}", nanos_to_millis(timing.run_time_ns));
    println!("  total_ms={:.3}", nanos_to_millis(timing.total_time_ns));
    println!("  generated_tokens={}", timing.generated_token_count);
    println!("  ticks={}", timing.tick_count);
    println!("  scheduler_turns={}", timing.scheduler_turn_count);
    if let Some(average) = timing.average_generated_token_time_ns {
        println!("  avg_generated_token_ms={:.3}", nanos_to_millis(average));
    }
    if let Some(average) = timing.average_tick_time_ns {
        println!("  avg_tick_ms={:.3}", nanos_to_millis(average));
    }
    if let Some(average) = timing.average_scheduler_turn_time_ns {
        println!("  avg_scheduler_turn_ms={:.3}", nanos_to_millis(average));
    }
}

fn print_placed_pedal_timing_profile(
    summaries: &[RuntimePlacedPedalTimingSummaryReport],
    max_rows: usize,
) {
    if summaries.is_empty() || max_rows == 0 {
        return;
    }
    println!("top_pedals:");
    for summary in summaries.iter().take(max_rows) {
        println!(
            "  {}:{} total_ms={:.3} ticks={} dispatches={} avg_tick_ms={} avg_dispatch_ms={}",
            summary.device_id,
            summary.pedal_id,
            nanos_to_millis(summary.total_run_time_ns),
            summary.tick_count,
            summary.dispatch_count,
            optional_nanos_to_millis(summary.average_tick_time_ns),
            optional_nanos_to_millis(summary.average_dispatch_time_ns)
        );
    }
}

fn optional_nanos_to_millis(value: Option<u64>) -> String {
    value
        .map(|nanos| format!("{:.3}", nanos_to_millis(nanos)))
        .unwrap_or_else(|| "n/a".to_string())
}

fn nanos_to_millis(nanos: u64) -> f64 {
    nanos as f64 / 1_000_000.0
}

fn nanos_to_millis_f64(nanos: f64) -> f64 {
    nanos / 1_000_000.0
}

fn print_usage() {
    println!("{}", usage());
}

fn usage() -> &'static str {
    "Usage: llmoop-runtime --package <COMPILED_PACKAGE.json> (--prompt <TEXT> | --chat) [OPTIONS]

Options:
  --package <PATH>           Compiled resident model package manifest. Required.
  --prompt <TEXT>            External text event to inject into the resident stream.
                             With --chat, this is the optional first message.
  --chat                     Start an interactive resident text session.
  --device <DEVICE_ID>       Default logical device for this runtime patch.
  --place-pedal <PEDAL=DEV>  Assign one runtime pedal instance to a logical device.
  --bind-device <DEV=TARGET> Bind a logical device to a discovered Vulkan device ID.
  --chain <ITEM[,ITEM...]>    Runtime source chain. ITEM is SOURCE or INSTANCE=SOURCE.
  --duplicate-after <AFTER=NEW>
                             Duplicate runtime pedal instance AFTER with id NEW.
  --inspect-runtime          Preview UI-ready package, patch, placement, device, and route facts.
  --inspect-package          Summarize the compiled source pedal kit and available devices.
  --inspect-patch            Preview the effective runtime patch without mounting devices.
  --inspect-placement        Mount and summarize every logical device slice in the runtime patch.
  --inspect-device-slice <DEVICE_ID>
                             Mount and summarize only the runtime patch pedals assigned to DEVICE_ID.
  --max-new-tokens <N>       Generation stop condition, independent of context size. Default: 65536
  --speculative-draft-tokens <N>
                             MTP draft tokens proposed per verification cycle. Default: 2; 0 disables MTP.
  --context-size <N>         Runtime transient-state window. Default: auto, up to the model maximum.
  --vulkan-device-index <N>  Use Vulkan physical device index N as the default local target.
  --seed <U32>               Explicit sampler randomness seed. Default: 0
  --no-special-tokens        Do not add tokenizer special tokens to raw --prompt input.
                             Chat templates always own their complete special-token framing.
  --keep-special-tokens      Keep tokenizer special tokens in decoded output text.
  --generated-only           Print only newly generated text instead of prompt + generated text.
  --profile                  Print human-readable timing and top-pedal summaries.
  --profile-runs <N>         Run N fresh prompt trials and report aggregate benchmark stats.
  --json                     Print a machine-readable run report.
  -h, --help                 Show this help.

Example:
  python -m llmoop --compile-model <MODEL_DIR>
  cargo run --manifest-path runtime-rs/Cargo.toml --features 'vulkan tokenizers' --bin llmoop-runtime -- --package packages/model_xxx/vulkan_resident_package.json --chat"
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    use chrono::{FixedOffset, TimeZone};
    use tokenizers::models::wordlevel::WordLevel;
    use tokenizers::pre_tokenizers::whitespace::Whitespace;
    use tokenizers::processors::template::TemplateProcessing;
    use tokenizers::{AddedToken, Tokenizer};

    use llmoop_runtime::{
        VulkanComputeDeviceInfo, VulkanResidentHfTokenizerTextCodec, VulkanResidentTokenTextCodec,
        VulkanResidentTokenTextCodecError,
    };

    use super::{
        Args, RuntimeChatFormatter, RuntimeChatMessage, RuntimeChatSession,
        assistant_content_token_ids, chat_transcript_codec, incremental_chat_token_delta,
        model_owned_assistant_turn_stop_token_id, normalize_chat_template_for_runtime,
        parse_device_binding_assignment, parse_source_chain, parse_vulkan_device_uuid_ref,
        resolve_runtime_context_size, resolve_runtime_vulkan_physical_device_ref,
        runtime_device_bindings_report, runtime_physical_device_bindings_in,
    };

    fn formatter(template_source: &str) -> RuntimeChatFormatter {
        RuntimeChatFormatter {
            template_source: template_source.to_string(),
            template_variables: serde_json::Map::new(),
            render_time: FixedOffset::east_opt(0)
                .unwrap()
                .with_ymd_and_hms(2026, 7, 18, 12, 0, 0)
                .unwrap(),
        }
    }

    #[derive(Clone, Copy)]
    struct CharacterCodec;

    impl VulkanResidentTokenTextCodec for CharacterCodec {
        fn encode_text(&self, text: &str) -> Result<Vec<u32>, VulkanResidentTokenTextCodecError> {
            Ok(text.chars().map(u32::from).collect())
        }

        fn decode_tokens(
            &self,
            token_ids: &[u32],
        ) -> Result<String, VulkanResidentTokenTextCodecError> {
            token_ids
                .iter()
                .map(|token_id| {
                    char::from_u32(*token_id).ok_or_else(|| {
                        VulkanResidentTokenTextCodecError::new(format!(
                            "invalid character token {token_id}"
                        ))
                    })
                })
                .collect()
        }
    }

    #[test]
    fn chat_template_tokenization_does_not_inject_post_processor_special_tokens() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let tokenizer_dir = std::env::temp_dir().join(format!(
            "llmoop-chat-tokenizer-specials-{}-{unique}",
            std::process::id()
        ));
        fs::create_dir_all(&tokenizer_dir).unwrap();

        let mut tokenizer = Tokenizer::new(WordLevel::default());
        tokenizer
            .add_special_tokens([AddedToken::from("<bos>", true)])
            .unwrap();
        tokenizer
            .add_tokens([AddedToken::from("hello", false)])
            .unwrap();
        let bos_id = tokenizer.token_to_id("<bos>").unwrap();
        let hello_id = tokenizer.token_to_id("hello").unwrap();
        tokenizer.with_pre_tokenizer(Some(Whitespace));
        tokenizer.with_post_processor(Some(
            TemplateProcessing::builder()
                .try_single("<bos> $A")
                .unwrap()
                .special_tokens(vec![("<bos>", bos_id)])
                .build()
                .unwrap(),
        ));
        tokenizer
            .save(tokenizer_dir.join("tokenizer.json"), false)
            .unwrap();

        let raw_codec = VulkanResidentHfTokenizerTextCodec::from_model_dir(&tokenizer_dir)
            .unwrap()
            .with_add_special_tokens(true);
        let chat_codec = chat_transcript_codec(&tokenizer_dir).unwrap();

        assert_eq!(
            raw_codec.encode_text("hello").unwrap(),
            vec![bos_id, hello_id]
        );
        assert_eq!(chat_codec.encode_text("hello").unwrap(), vec![hello_id]);

        fs::remove_dir_all(tokenizer_dir).unwrap();
    }

    #[test]
    fn context_defaults_to_model_capacity_and_rejects_impossible_requests() {
        assert_eq!(
            resolve_runtime_context_size(131_072, None, 65_536).unwrap(),
            131_072
        );
        assert_eq!(
            resolve_runtime_context_size(131_072, Some(8_192), 4_096).unwrap(),
            8_192
        );

        let too_small = resolve_runtime_context_size(131_072, Some(4_096), 4_097).unwrap_err();
        assert_eq!(too_small.kind(), std::io::ErrorKind::InvalidInput);
        assert!(too_small.to_string().contains("cannot hold"));

        let too_large = resolve_runtime_context_size(32_768, Some(65_536), 1).unwrap_err();
        assert_eq!(too_large.kind(), std::io::ErrorKind::InvalidInput);
        assert!(too_large.to_string().contains("exceeds the model maximum"));

        let zero_model = resolve_runtime_context_size(0, None, 0).unwrap_err();
        assert_eq!(zero_model.kind(), std::io::ErrorKind::InvalidData);
    }

    #[test]
    fn runtime_device_binding_parser_validates_syntax_without_device_discovery() {
        assert_eq!(
            parse_device_binding_assignment("gpu0 = vulkan:5").unwrap(),
            ("gpu0".to_string(), "vulkan:5".to_string())
        );
        assert_eq!(
            parse_device_binding_assignment("remote = lan:worker-a").unwrap(),
            ("remote".to_string(), "lan:worker-a".to_string())
        );
        assert_eq!(
            resolve_runtime_vulkan_physical_device_ref("vulkan:7").unwrap(),
            Some(7)
        );

        for invalid in [
            "gpu0=vulkan:",
            "gpu0=vulkan-latest",
            "cpu0=cpu:",
            "cpu0=cpuish",
            "gpu0=vulkan-uuid:abcd",
        ] {
            assert!(
                parse_device_binding_assignment(invalid).is_err(),
                "accepted invalid binding {invalid:?}"
            );
        }
    }

    #[test]
    fn runtime_physical_bindings_distinguish_logical_from_physical_placement() {
        let available_devices = vec![
            VulkanComputeDeviceInfo {
                physical_device_index: 2,
                physical_device_id: "vulkan-uuid:00000000000000000000000000000002".to_string(),
                device_uuid: [2; 16],
                device_name: "GPU 2".to_string(),
                device_type: "discrete_gpu".to_string(),
                vendor_id: 1,
                device_id: 2,
                api_version: 1,
                driver_version: 1,
                compute_queue_family_indices: vec![0],
                memory_heaps: Vec::new(),
                selected_by_default: true,
            },
            VulkanComputeDeviceInfo {
                physical_device_index: 3,
                physical_device_id: "vulkan-uuid:00000000000000000000000000000003".to_string(),
                device_uuid: [3; 16],
                device_name: "GPU 3".to_string(),
                device_type: "discrete_gpu".to_string(),
                vendor_id: 1,
                device_id: 3,
                api_version: 1,
                driver_version: 1,
                compute_queue_family_indices: vec![0],
                memory_heaps: Vec::new(),
                selected_by_default: false,
            },
        ];
        let logical_device_ids = vec!["board_a".to_string(), "board_b".to_string()];
        let colocated = runtime_physical_device_bindings_in(
            &Args::default(),
            &logical_device_ids,
            &available_devices,
        )
        .unwrap();
        assert_eq!(colocated.get("board_a"), Some(&2));
        assert_eq!(colocated.get("board_b"), Some(&2));
        assert_eq!(colocated.values().collect::<BTreeSet<_>>().len(), 1);

        let mut split_args = Args::default();
        split_args
            .device_bindings
            .insert("board_b".to_string(), "vulkan:3".to_string());
        let split = runtime_physical_device_bindings_in(
            &split_args,
            &logical_device_ids,
            &available_devices,
        )
        .unwrap();
        assert_eq!(split.get("board_a"), Some(&2));
        assert_eq!(split.get("board_b"), Some(&3));
        assert_eq!(split.values().collect::<BTreeSet<_>>().len(), 2);
    }

    #[test]
    fn fully_explicit_device_binding_report_does_not_request_an_unused_default_gpu() {
        let logical_device_ids = vec!["board_a".to_string(), "board_b".to_string()];
        let mut args = Args::default();
        args.device_bindings
            .insert("board_a".to_string(), "vulkan:2".to_string());
        args.device_bindings
            .insert("board_b".to_string(), "vulkan:3".to_string());

        let report = runtime_device_bindings_report(&args, &logical_device_ids);

        assert_eq!(report.process_vulkan_device_index, None);
        assert_eq!(report.default_vulkan_device_index, None);
        assert_eq!(report.requested_vulkan_device_indices, vec![2, 3]);
    }

    #[test]
    fn runtime_source_chain_parser_preserves_duplicates_only_with_unique_instances() {
        assert_eq!(
            parse_source_chain("layer_0 -> repeat=layer_0 -> layer_1").unwrap(),
            vec![
                ("layer_0".to_string(), "layer_0".to_string()),
                ("repeat".to_string(), "layer_0".to_string()),
                ("layer_1".to_string(), "layer_1".to_string()),
            ]
        );
        assert!(parse_source_chain("layer_0,layer_0").is_err());
        assert!(parse_source_chain("layer_0,,layer_1").is_err());
        assert!(parse_source_chain("repeat=").is_err());
    }

    #[test]
    fn stable_vulkan_device_uuid_references_are_parsed_without_discovery() {
        assert_eq!(
            parse_vulkan_device_uuid_ref("vulkan-uuid:000000000a0000000000000000000000").unwrap(),
            Some([0, 0, 0, 0, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        );
        assert!(
            parse_vulkan_device_uuid_ref("vulkan-uuid:not-a-device")
                .unwrap_err()
                .contains("32 hexadecimal digits")
        );
        assert_eq!(parse_vulkan_device_uuid_ref("vulkan:3").unwrap(), None);
    }

    #[test]
    fn model_template_controls_pipe_turn_role_names() {
        let mut formatter = formatter(
            "{%- for message in messages %}{{- '<|turn>' + ('model' if message.role == 'assistant' else message.role) + '\n' + (message.content | trim) + '<turn|>\n' }}{%- endfor %}{%- if add_generation_prompt %}{{- '<|turn>model\n' }}{%- endif %}",
        );
        formatter.template_variables.insert(
            "bos_token".to_string(),
            serde_json::Value::String("<bos>".to_string()),
        );
        let messages = vec![
            RuntimeChatMessage {
                role: "user".to_string(),
                content: "Hello".to_string(),
            },
            RuntimeChatMessage {
                role: "assistant".to_string(),
                content: "Hi there".to_string(),
            },
            RuntimeChatMessage {
                role: "user".to_string(),
                content: "Remember me".to_string(),
            },
        ];

        assert_eq!(
            formatter.format_messages(&messages, true).unwrap(),
            "<|turn>user\nHello<turn|>\n<|turn>model\nHi there<turn|>\n<|turn>user\nRemember me<turn|>\n<|turn>model\n"
        );
    }

    #[test]
    fn model_template_keeps_default_reasoning_branch() {
        let formatter = formatter(
            "{%- for message in messages %}{{- '<|im_start|>' + message.role + '\n' + message.content + '<|im_end|>\n' }}{%- endfor %}{%- if add_generation_prompt %}{{- '<|im_start|>assistant\n' }}{%- if enable_thinking is defined and enable_thinking is false %}{{- '<think>\n\n</think>\n\n' }}{%- else %}{{- '<think>\n' }}{%- endif %}{%- endif %}",
        );

        assert_eq!(
            formatter
                .format_messages(
                    &[RuntimeChatMessage {
                        role: "user".to_string(),
                        content: "Solve this".to_string(),
                    }],
                    true,
                )
                .unwrap(),
            "<|im_start|>user\nSolve this<|im_end|>\n<|im_start|>assistant\n<think>\n"
        );
    }

    #[test]
    fn hugging_face_generation_metadata_preserves_rendered_content_and_trimming() {
        let normalized = normalize_chat_template_for_runtime(
            "before \n{%- generation -%}\nassistant content\n{%- endgeneration -%}\n after",
        );
        let formatter = formatter(&normalized);

        assert_eq!(
            formatter.format_messages(&[], false).unwrap(),
            "beforeassistant contentafter"
        );
    }

    #[test]
    fn model_template_can_supply_a_dated_default_system_turn() {
        let formatter = formatter(
            "{%- if messages[0].role == 'system' %}{%- set loop_messages = messages[1:] %}{%- else %}{{- '<|start_of_role|>system<|end_of_role|>Current Date: ' + strftime_now('%B %d, %Y') + '.<|end_of_text|>\n' }}{%- set loop_messages = messages %}{%- endif %}{%- for message in loop_messages %}{{- '<|start_of_role|>' + message.role + '<|end_of_role|>' + message.content + '<|end_of_text|>\n' }}{%- if loop.last and add_generation_prompt %}{{- '<|start_of_role|>assistant<|end_of_role|>' }}{%- endif %}{%- endfor %}",
        );

        assert_eq!(
            formatter
                .format_messages(
                    &[RuntimeChatMessage {
                        role: "user".to_string(),
                        content: "Hello".to_string(),
                    }],
                    true,
                )
                .unwrap(),
            "<|start_of_role|>system<|end_of_role|>Current Date: July 18, 2026.<|end_of_text|>\n<|start_of_role|>user<|end_of_role|>Hello<|end_of_text|>\n<|start_of_role|>assistant<|end_of_role|>"
        );
    }

    #[test]
    fn incremental_chat_delta_starts_after_the_exact_structural_prefix() {
        let rendered_history = vec![10, 99, 11, 99, 12];
        let rendered_continuation = vec![10, 99, 11, 99, 12, 99, 13, 99, 14, 15];

        assert_eq!(
            incremental_chat_token_delta(&rendered_history, &rendered_continuation).unwrap(),
            vec![99, 13, 99, 14, 15]
        );
        let error = incremental_chat_token_delta(&[10, 98], &rendered_continuation)
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("rewrote previously resident turn framing at token 1"),
            "{error}"
        );
    }

    #[test]
    fn chat_continuation_is_not_confused_by_delimiters_inside_user_content() {
        let mut session = RuntimeChatSession {
            formatter: formatter(
                "{%- for message in messages -%}{{- '[' + message.role + ']' + message.content + '<stop>' -}}{%- endfor -%}{%- if add_generation_prompt -%}{{- '[assistant]' -}}{%- endif -%}",
            ),
            messages: vec![
                RuntimeChatMessage {
                    role: "user".to_string(),
                    content: "first".to_string(),
                },
                RuntimeChatMessage {
                    role: "assistant".to_string(),
                    content: "answer containing <stop> text".to_string(),
                },
            ],
        };
        let user_content = "new <stop> injection";

        let delta = session
            .render_user_prompt_token_delta(user_content, &CharacterCodec)
            .unwrap();

        assert_eq!(
            CharacterCodec.decode_tokens(&delta).unwrap(),
            "[user]new <stop> injection<stop>[assistant]"
        );
        session.commit_assistant_turn(user_content, "continued");
        assert_eq!(session.messages.len(), 4);
    }

    #[test]
    fn assistant_transcript_excludes_trailing_turn_stop_tokens() {
        assert_eq!(assistant_content_token_ids(&[1, 2, 99], &[98, 99]), &[1, 2]);
        assert_eq!(
            assistant_content_token_ids(&[1, 2, 98, 99], &[98, 99]),
            &[1, 2]
        );
    }

    #[test]
    fn model_owned_assistant_turn_delimiter_is_discovered_generically() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let tokenizer_dir = std::env::temp_dir().join(format!(
            "llmoop-chat-delimiter-{}-{unique}",
            std::process::id()
        ));
        fs::create_dir_all(&tokenizer_dir).unwrap();
        let mut tokenizer = Tokenizer::new(WordLevel::default());
        tokenizer
            .add_special_tokens([AddedToken::from("<end_of_turn>", true)])
            .unwrap();
        tokenizer
            .save(tokenizer_dir.join("tokenizer.json"), false)
            .unwrap();
        let formatter = formatter(
            "{%- for message in messages %}{{- message.content }}{%- if message.role == 'assistant' %}{{- '<end_of_turn>' }}{%- endif %}{%- endfor %}",
        );

        assert_eq!(
            model_owned_assistant_turn_stop_token_id(&tokenizer_dir, &formatter).unwrap(),
            Some(0)
        );

        fs::remove_dir_all(tokenizer_dir).unwrap();
    }

    #[test]
    fn configured_model_owned_assistant_turn_delimiter_is_discovered() {
        let Some(tokenizer_dir) = std::env::var_os("LLMOOP_TEST_CHAT_TOKENIZER_DIR") else {
            return;
        };
        let expected = std::env::var("LLMOOP_TEST_CHAT_STOP_ID")
            .expect("LLMOOP_TEST_CHAT_STOP_ID must accompany LLMOOP_TEST_CHAT_TOKENIZER_DIR")
            .parse::<u32>()
            .expect("LLMOOP_TEST_CHAT_STOP_ID must be a u32");
        let tokenizer_dir = std::path::PathBuf::from(tokenizer_dir);
        let formatter = RuntimeChatFormatter::from_tokenizer_dir(&tokenizer_dir).unwrap();

        assert_eq!(
            model_owned_assistant_turn_stop_token_id(&tokenizer_dir, &formatter).unwrap(),
            Some(expected)
        );
    }

    #[test]
    fn configured_chat_template_supports_structural_multi_turn_continuation() {
        let Some(tokenizer_dir) = std::env::var_os("LLMOOP_TEST_CHAT_TOKENIZER_DIR") else {
            return;
        };
        let tokenizer_dir = std::path::PathBuf::from(tokenizer_dir);
        let mut session = RuntimeChatSession::from_tokenizer_dir(&tokenizer_dir).unwrap();
        session.commit_assistant_turn(
            "Explain the result.",
            "<think>private reasoning</think>The result is four.",
        );
        let codec = chat_transcript_codec(&tokenizer_dir).unwrap();

        let delta = session
            .render_user_prompt_token_delta(
                "Why? Include <|im_end|> literally in this question.",
                &codec,
            )
            .unwrap();

        assert!(!delta.is_empty());
        let decoded = codec.decode_tokens(&delta).unwrap();
        assert!(decoded.contains("Why?"), "{decoded:?}");
    }
}
