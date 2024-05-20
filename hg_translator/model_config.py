model_name = "convert_dich"
num_train_epochs = 30

per_device_train_batch_size = 8
per_device_eval_batch_size = 8

checkpoint = "VietAI/vit5-base"

if "google-t5" in checkpoint:
    prefix = "translate English to French: "
else:
    prefix = "dịch sang tiếng Việt: "