model_name = "tien-hiep"
num_train_epochs = 3

# For GPU P100 16GB
per_device_train_batch_size = 12
per_device_eval_batch_size = 12

# checkpoint = "VietAI/vit5-base"
checkpoint = "google/mt5-base"

# prefix = "dịch sang tiên hiệp: "
prefix = "translate from Chinese to Vietnamese: "


source_lang = "zh"
target_lang = "vi"
