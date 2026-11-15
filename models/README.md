# models/

The fine-tuned LoRA adapter is **not stored in git** (about 160 MB; GitHub rejects files over 100 MB).

Put the two adapter files here before building the API image or starting the API:

```
models/llama-2-aftercare-v4/
  adapter_config.json
  adapter_model.safetensors
```

`Dockerfile.api` copies exactly those two files into the image, and `NOTE_ADAPTER_PATH` points at the folder.

Where to keep the adapter: a private S3 bucket (`aws s3 cp` it down at deploy time) or a private Hugging Face repo.
The base model, `NousResearch/Llama-2-7b-chat-hf`, is downloaded automatically on the first note generation.
The adapter must be used with that same base model.
