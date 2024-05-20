model_name = "convert_dich"
num_train_epochs = 3

# For GPU P100 16GB
per_device_train_batch_size = 12
per_device_eval_batch_size = 12

checkpoint = "VietAI/vit5-base"

if "google-t5" in checkpoint:
    prefix = "translate English to French: "
else:
    prefix = "dịch sang tiếng Việt: "
