from huggingface_hub import snapshot_download, login

from huggingface_hub import snapshot_download
repo_id = "openslr/librispeech_asr"
local_path = snapshot_download(repo_id=repo_id, repo_type="dataset", resume_download=True, revision='main')

from datasets import load_dataset
dataset = load_dataset(local_path, 'clean', split='train.100')