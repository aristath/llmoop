fn main() {
    if let Err(error) = run() {
        eprintln!("nerve-runtime error: {error}");
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
    if args.inspect_devices {
        return inspect_device_capabilities(&args);
    }
    let package_manifest = args.package_manifest.as_ref().ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            "--package is required; run `python -m nerve --compile-model <MODEL_DIR>` first",
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
    if args.inspect_graph {
        let manifest = VulkanResidentModelPackageManifest::from_json_file(package_manifest)?;
        return inspect_graph(&args, package_manifest, &manifest_dir, manifest);
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

    run_placed_prompt(&context, runtime_model)
}
