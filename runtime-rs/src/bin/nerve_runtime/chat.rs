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
    fn from_tokenizer_dir(
        tokenizer_dir: &Path,
        template_variables: &BTreeMap<String, serde_json::Value>,
    ) -> Result<Self, Box<dyn Error>> {
        Ok(Self {
            formatter: RuntimeChatFormatter::from_tokenizer_dir(tokenizer_dir, template_variables)?,
            messages: Vec::new(),
        })
    }

    fn render_user_prompt_token_delta<C>(
        &self,
        user_content: &str,
        codec: &C,
        stop_token_ids: &[u32],
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

        const ASSISTANT_CONTENT_PROBE: &str = "NERVE_PREVIOUS_ASSISTANT_CONTENT_PROBE_6C70A8";
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

        probe_history.push(RuntimeChatMessage {
            role: "user".to_string(),
            content: user_content.to_string(),
        });
        let formatted_continuation = self.formatter.format_messages(&probe_history, true)?;

        let history_probe_offset = formatted_history
            .rfind(ASSISTANT_CONTENT_PROBE)
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    "chat template did not preserve the assistant content probe",
                )
            })?;
        let continuation_probe_offset = formatted_continuation
            .find(ASSISTANT_CONTENT_PROBE)
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    "chat template did not preserve the assistant content probe in the continuation",
                )
            })?;
        let history_suffix =
            &formatted_history[history_probe_offset + ASSISTANT_CONTENT_PROBE.len()..];
        let continuation_suffix =
            &formatted_continuation[continuation_probe_offset + ASSISTANT_CONTENT_PROBE.len()..];
        let history_suffix_token_ids = codec.encode_text(history_suffix)?;
        let continuation_suffix_token_ids = codec.encode_text(continuation_suffix)?;
        Ok(incremental_chat_token_delta(
            &history_suffix_token_ids,
            &continuation_suffix_token_ids,
            stop_token_ids,
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
    rendered_history_suffix_token_ids: &[u32],
    rendered_continuation_suffix_token_ids: &[u32],
    stop_token_ids: &[u32],
) -> Result<Vec<u32>, io::Error> {
    let stop_index = rendered_history_suffix_token_ids
        .iter()
        .position(|token_id| stop_token_ids.contains(token_id))
        .ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "chat template assistant turn suffix does not contain a configured stop token",
            )
        })?;
    let committed_history_suffix = &rendered_history_suffix_token_ids[..=stop_index];
    if !rendered_continuation_suffix_token_ids.starts_with(committed_history_suffix) {
        let common_prefix_len = rendered_history_suffix_token_ids
            .iter()
            .zip(rendered_continuation_suffix_token_ids)
            .take_while(|(history, continuation)| history == continuation)
            .count();
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "chat template rewrote the completed assistant turn suffix at token {common_prefix_len}"
            ),
        ));
    }
    Ok(rendered_continuation_suffix_token_ids[stop_index + 1..].to_vec())
}

#[derive(Clone, Debug)]
struct RuntimeChatFormatter {
    template_source: String,
    template_variables: serde_json::Map<String, serde_json::Value>,
    render_time: DateTime<FixedOffset>,
}

impl RuntimeChatFormatter {
    fn from_tokenizer_dir(
        tokenizer_dir: &Path,
        variable_overrides: &BTreeMap<String, serde_json::Value>,
    ) -> Result<Self, Box<dyn Error>> {
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
        let mut template_variables = tokenizer_template_variables(tokenizer_dir)?;
        template_variables.extend(
            variable_overrides
                .iter()
                .map(|(name, value)| (name.clone(), value.clone())),
        );
        let formatter = Self {
            template_source: normalize_chat_template_for_runtime(&template),
            template_variables,
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
    timing: RuntimePromptTimingReport,
    execution_counters: VulkanResidentExecutionCounters,
    prefix_state_cache: VulkanResidentPlacedPrefixStateCacheStats,
    speculative_cycle_count: usize,
    proposed_draft_token_count: usize,
    accepted_draft_token_count: usize,
    speculative_emitted_token_count: usize,
    speculative_draft_time_ns: u64,
    speculative_target_verification_time_ns: u64,
    speculative_draft_catch_up_time_ns: u64,
    resident_feedback: RuntimeFeedbackExecutionReport,
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
    let prompt_delta = chat_session.render_user_prompt_token_delta(
        input_text,
        transcript_codec,
        stop_token_ids,
    )?;
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
            print_runtime_timing_stats("stats", &turn.timing);
            print_runtime_execution_counters(&turn.execution_counters);
            print_runtime_prefix_state_cache_stats(&turn.prefix_state_cache);
            print_runtime_speculative_stats(
                turn.speculative_cycle_count,
                turn.proposed_draft_token_count,
                turn.accepted_draft_token_count,
                turn.speculative_emitted_token_count,
                turn.speculative_draft_time_ns,
                turn.speculative_target_verification_time_ns,
                turn.speculative_draft_catch_up_time_ns,
            );
            print_runtime_feedback_stats(&turn.resident_feedback);
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
    const ASSISTANT_SENTINEL: &str = "NERVE_ASSISTANT_TURN_CONTENT_SENTINEL_7F3A9C";
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
