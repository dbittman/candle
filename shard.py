from transformers import AutoModelForCausalLM, AutoTokenizer
device_map = {"": "cpu"}


orig_model_file_path = "mixtral-model/snapshots/ffe1a706bacbd5abddc5ff99432ee38f7e0662fb/"
model_file_path = "shard_dir"

print("from-pretrained")
model = AutoModelForCausalLM.from_pretrained(orig_model_file_path)
tokenizer = AutoTokenizer.from_pretrained(orig_model_file_path)

print("saving")

sd = model.state_dict()
new_sd = {}
for k, v in sd.items():
    # print(k)
    # if k == "lm_head.weight":
    #    new_sd["lm_head.weight"] = v
    # else:
    #    k = "model." + k
    new_sd[k] = v


model.save_pretrained(
    model_file_path, max_shard_size="768MB", state_dict=new_sd)


tokenizer.save_pretrained(model_file_path, max_shard_size="768MB")

# print("load checkpoint")
# model = load_checkpoint_and_dispatch(
#    model, checkpoint = model_file_path, device_map = device_map, no_split_module_classes = ['Block'])
