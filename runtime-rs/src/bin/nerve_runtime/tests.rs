#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, BTreeSet};
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    use chrono::{FixedOffset, TimeZone};
    use tokenizers::models::wordlevel::WordLevel;
    use tokenizers::pre_tokenizers::whitespace::Whitespace;
    use tokenizers::processors::template::TemplateProcessing;
    use tokenizers::{AddedToken, Tokenizer};

    use nerve_runtime::{
        VulkanComputeDeviceInfo, VulkanResidentHfTokenizerTextCodec, VulkanResidentTokenTextCodec,
        VulkanResidentTokenTextCodecError,
    };

    use super::{
        Args, RuntimeChatFormatter, RuntimeChatMessage, RuntimeChatSession,
        assistant_content_token_ids, chat_transcript_codec, incremental_chat_token_delta,
        model_owned_assistant_turn_stop_token_id, normalize_chat_template_for_runtime,
        parse_args_from, parse_chat_template_variable, parse_device_binding_assignment,
        parse_source_chain, parse_vulkan_device_uuid_ref, resolve_runtime_context_size,
        resolve_runtime_vulkan_physical_device_ref, runtime_device_bindings_report,
        runtime_physical_device_bindings_in,
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

    #[test]
    fn duplicate_default_device_is_rejected() {
        let error = parse_args_from(
            ["--device", "gpu0", "--device", "gpu1"]
                .into_iter()
                .map(str::to_string),
        )
        .unwrap_err();

        assert_eq!(error, "--device may only be supplied once");
    }

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
            "nerve-chat-tokenizer-specials-{}-{unique}",
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
    fn model_template_accepts_boolean_reasoning_control() {
        let mut formatter = formatter(
            "{%- if add_generation_prompt %}{%- if enable_thinking is false %}direct{%- else %}thinking{%- endif %}{%- endif %}",
        );
        formatter.template_variables.insert(
            "enable_thinking".to_string(),
            serde_json::Value::Bool(false),
        );

        assert_eq!(
            formatter
                .format_messages(
                    &[RuntimeChatMessage {
                        role: "user".to_string(),
                        content: "Answer directly".to_string(),
                    }],
                    true,
                )
                .unwrap(),
            "direct"
        );
    }

    #[test]
    fn chat_template_variables_require_json_values_and_jinja_names() {
        assert_eq!(
            parse_chat_template_variable("enable_thinking=false").unwrap(),
            (
                "enable_thinking".to_string(),
                serde_json::Value::Bool(false)
            )
        );
        assert_eq!(
            parse_chat_template_variable("tool_choice=\"auto\"").unwrap(),
            (
                "tool_choice".to_string(),
                serde_json::Value::String("auto".to_string())
            )
        );
        assert!(parse_chat_template_variable("enable-thinking=false").is_err());
        assert!(parse_chat_template_variable("enable_thinking=disabled").is_err());
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
    fn incremental_chat_delta_preserves_the_post_stop_separator() {
        let rendered_history_suffix = vec![50, 99, 10];
        let rendered_continuation_suffix = vec![50, 99, 10, 13, 14, 15];

        assert_eq!(
            incremental_chat_token_delta(
                &rendered_history_suffix,
                &rendered_continuation_suffix,
                &[99],
            )
            .unwrap(),
            vec![10, 13, 14, 15]
        );
        let error =
            incremental_chat_token_delta(&rendered_history_suffix, &[50, 98, 10, 13], &[99])
                .unwrap_err()
                .to_string();
        assert!(
            error.contains("rewrote the completed assistant turn suffix at token 1"),
            "{error}"
        );
        let error = incremental_chat_token_delta(&[50, 98], &[50, 98, 10], &[99])
            .unwrap_err()
            .to_string();
        assert!(error.contains("does not contain a configured stop token"));
    }

    #[test]
    fn incremental_chat_delta_ignores_history_only_tokens_after_the_turn_stop() {
        let rendered_history_suffix = vec![50, 99, 10, 100];
        let rendered_continuation_suffix = vec![50, 99, 10, 13, 14, 15];

        assert_eq!(
            incremental_chat_token_delta(
                &rendered_history_suffix,
                &rendered_continuation_suffix,
                &[99],
            )
            .unwrap(),
            vec![10, 13, 14, 15]
        );
    }

    #[test]
    fn chat_continuation_ignores_template_rewrites_before_assistant_content() {
        let mut session = RuntimeChatSession {
            formatter: formatter(
                "{%- for message in messages -%}{%- if message.role == 'user' -%}{{- '[user]' + message.content + '!' -}}{%- else -%}{{- '[assistant]' -}}{%- if loop.last -%}{{- '<preserved-thinking>' -}}{%- endif -%}{{- message.content + '!' -}}{%- endif -%}{%- endfor -%}{%- if add_generation_prompt -%}{{- '[assistant]<think>' -}}{%- endif -%}",
            ),
            messages: vec![
                RuntimeChatMessage {
                    role: "user".to_string(),
                    content: "first".to_string(),
                },
                RuntimeChatMessage {
                    role: "assistant".to_string(),
                    content: "answer".to_string(),
                },
            ],
        };

        let delta = session
            .render_user_prompt_token_delta("second", &CharacterCodec, &[u32::from('!')])
            .unwrap();

        assert_eq!(
            CharacterCodec.decode_tokens(&delta).unwrap(),
            "[user]second![assistant]<think>"
        );
        session.commit_assistant_turn("second", "continued");
        assert_eq!(session.messages.len(), 4);
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
            .render_user_prompt_token_delta(user_content, &CharacterCodec, &[u32::from('>')])
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
            "nerve-chat-delimiter-{}-{unique}",
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
        let Some(tokenizer_dir) = std::env::var_os("NERVE_TEST_CHAT_TOKENIZER_DIR") else {
            return;
        };
        let expected = std::env::var("NERVE_TEST_CHAT_STOP_ID")
            .expect("NERVE_TEST_CHAT_STOP_ID must accompany NERVE_TEST_CHAT_TOKENIZER_DIR")
            .parse::<u32>()
            .expect("NERVE_TEST_CHAT_STOP_ID must be a u32");
        let tokenizer_dir = std::path::PathBuf::from(tokenizer_dir);
        let formatter =
            RuntimeChatFormatter::from_tokenizer_dir(&tokenizer_dir, &BTreeMap::new()).unwrap();

        assert_eq!(
            model_owned_assistant_turn_stop_token_id(&tokenizer_dir, &formatter).unwrap(),
            Some(expected)
        );
    }

    #[test]
    fn configured_chat_template_supports_structural_multi_turn_continuation() {
        let Some(tokenizer_dir) = std::env::var_os("NERVE_TEST_CHAT_TOKENIZER_DIR") else {
            return;
        };
        let tokenizer_dir = std::path::PathBuf::from(tokenizer_dir);
        let mut session =
            RuntimeChatSession::from_tokenizer_dir(&tokenizer_dir, &BTreeMap::new()).unwrap();
        session.commit_assistant_turn(
            "Explain the result.",
            "<think>private reasoning</think>The result is four.",
        );
        let codec = chat_transcript_codec(&tokenizer_dir).unwrap();
        let stop_token_id =
            model_owned_assistant_turn_stop_token_id(&tokenizer_dir, &session.formatter)
                .unwrap()
                .expect("configured template must own an assistant turn stop token");

        let delta = session
            .render_user_prompt_token_delta(
                "Why? Include <|im_end|> literally in this question.",
                &codec,
                &[stop_token_id],
            )
            .unwrap();

        assert!(!delta.is_empty());
        let decoded = codec.decode_tokens(&delta).unwrap();
        assert!(decoded.contains("Why?"), "{decoded:?}");
    }

    #[test]
    fn configured_chat_template_honors_non_thinking_variable() {
        let Some(tokenizer_dir) = std::env::var_os("NERVE_TEST_CHAT_TOKENIZER_DIR") else {
            return;
        };
        let tokenizer_dir = std::path::PathBuf::from(tokenizer_dir);
        let variables = BTreeMap::from([(
            "enable_thinking".to_string(),
            serde_json::Value::Bool(false),
        )]);
        let session = RuntimeChatSession::from_tokenizer_dir(&tokenizer_dir, &variables).unwrap();
        let codec = chat_transcript_codec(&tokenizer_dir).unwrap();

        let prompt_ids = session
            .render_user_prompt_token_delta("Answer directly.", &codec, &[])
            .unwrap();
        let rendered = codec.decode_tokens(&prompt_ids).unwrap();

        assert!(rendered.contains("<think>\n\n</think>\n\n"), "{rendered:?}");
    }
}
