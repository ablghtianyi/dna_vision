def build_name_str(**kwargs) -> str:
    model_config = kwargs['model_config']
    training_config = kwargs['training_config']
    data_config = kwargs['data_config']
    model_type = model_config.model_type.lower()

    match model_type:
        case "vit":
            name_str = "patch{}_embed{}_head{}_plen{}_ratio{}_T{}_lr{}_B{}_wd{}_warm{}".format(
                            model_config.patch_size,
                            model_config.embed_dim,
                            model_config.num_heads,
                            model_config.max_path_len,
                            model_config.mlp_ratio,
                            training_config.num_epochs,
                            training_config.learning_rate,
                            training_config.batch_size,
                            training_config.weight_decay,
                            training_config.warmup_epochs
                        )
            if data_config.mixup is False:
                name_str = 'nomix_' + name_str
        case "dna_linear_nested_tf":
            n_tf = model_config.n_tf
            name_str = "bias_u{}_skip{}_patch{}_embed{}_head{}_ntf{}_plen{}_ds{}_mtopk{}_ratio{}_temp{}_T{}_lr{}_B{}_wd{}_warm{}".format(
                            model_config.bias_u,
                            model_config.skip_factor,
                            model_config.patch_size,
                            model_config.embed_dim,
                            model_config.num_heads,
                            n_tf,
                            model_config.max_path_len,
                            model_config.start_node_depth,
                            model_config.module_top_k,
                            model_config.mlp_ratio,
                            model_config.module_temperature,
                            training_config.num_epochs,
                            training_config.learning_rate,
                            training_config.batch_size,
                            training_config.weight_decay,
                            training_config.warmup_epochs
                        )
            if model_config.use_bias is True:
                name_str = 'lb_' + name_str
            if model_config.early_exit is True:
                assert model_config.hard_exit is False
                name_str = 'ex_' + name_str
            if model_config.hard_exit is True:
                assert model_config.early_exit is False
                name_str = 'hex_' + name_str
            if model_config.use_id is False:
                name_str = 'noid' + name_str
            if training_config.clip_norm > 0.0:
                name_str = name_str + f'_gn{training_config.clip_norm:.1f}'
            if 'nested' in model_type:
                name_str = f'nested_' + name_str
            if data_config.mixup is False:
                name_str = 'nomix_' + name_str
        case "dna_linear_nested_split":
            n_mlp = model_config.n_mlp
            name_str = "bias_u{}_skip{}_patch{}_embed{}_head{}_nattn{}_nmlp{}_nconv{}_plen{}_mtopk{}_ratio{}_temp{}_T{}_lr{}_B{}_wd{}_warm{}".format(
                            model_config.bias_u,
                            model_config.skip_factor,
                            model_config.patch_size,
                            model_config.embed_dim,
                            model_config.num_heads,
                            model_config.n_attn,
                            n_mlp,
                            model_config.n_conv,
                            model_config.max_path_len,
                            model_config.module_top_k,
                            model_config.mlp_ratio,
                            model_config.module_temperature,
                            training_config.num_epochs,
                            training_config.learning_rate,
                            training_config.batch_size,
                            training_config.weight_decay,
                            training_config.warmup_epochs
                        )
            if model_config.use_bias is True:
                name_str = 'lb_' + name_str
            if model_config.early_exit is True:
                assert model_config.hard_exit is False
                name_str = 'ex_' + name_str
            if model_config.hard_exit is True:
                assert model_config.early_exit is False
                name_str = 'hex_' + name_str
            if model_config.use_id is False:
                name_str = 'noid' + name_str
            if training_config.clip_norm > 0.0:
                name_str = name_str + f'_gn{training_config.clip_norm:.1f}'
            if 'nested' in model_type:
                name_str = f'nested_' + name_str
            if data_config.mixup is False:
                name_str = 'nomix_' + name_str
        
        case _:
            raise ValueError(f"Unknown model type: {model_type}.")
    
    return name_str