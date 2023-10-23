# How to push any AWQ LLM to Replicate

1. Download a AWQ model from HuggingFace and place it in /models
2. Update the `<model_name>` in predict.py to match the file in /models
3. Create a model on Replicate (https://replicate.com/docs/guides/push-a-transformers-model)
4. Run `cog login`
5. Run `cog push r8.im/<your-username>/<your-model-name>`
6. `docker system prune` to cleanup temp images

Thanks to nateraw on https://github.com/nateraw/replicate-examples
