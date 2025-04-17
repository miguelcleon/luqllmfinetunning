from optimum.intel.openvino import OVConfig, OVModelForCausalLM

# where MODEL_PATH is your HF checkpoint
export_dir = "ov_model_ir"
MODEL_PATH = "trainingdata/gemma3_luq_model"
ov_config = OVConfig(precision="FP16")            # or "FP32"/"INT8" if you like
OVModelForCausalLM.from_pretrained(
    MODEL_PATH,
    export=True,
    ov_config=ov_config,
    save_directory=export_dir
)