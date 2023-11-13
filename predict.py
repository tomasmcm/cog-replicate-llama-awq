import time
from vllm import LLM, SamplingParams
import torch
from cog import BasePredictor, Input

class Predictor(BasePredictor):

    def setup(self):
        self.llm = LLM(
            model="./models/TheBloke_Llama-2-7B-Chat-AWQ",
            quantization="awq",
            dtype="auto",
            gpu_memory_utilization=0.8,
            max_num_batched_tokens=32768,
            max_model_len=512,
            tensor_parallel_size=1
        )

    def predict(
        self,
        prompt: str = Input(
            description="Text prompt to send to the model."
        ),
        max_tokens: int = Input(
            description="Maximum number of tokens to generate per output sequence.",
            default=128,
        ),
        presence_penalty: float = Input(
            description="Float that penalizes new tokens based on whether they appear in the generated text so far. Values > 0 encourage the model to use new tokens, while values < 0 encourage the model to repeat tokens.",
            ge=-5,
            le=5,
            default=0.0,
        ),
        frequency_penalty: float = Input(
            description="Float that penalizes new tokens based on their frequency in the generated text so far. Values > 0 encourage the model to use new tokens, while values < 0 encourage the model to repeat tokens.",
            ge=-5,
            le=5,
            default=0.0,
        ),
        temperature: float = Input(
            description="Float that controls the randomness of the sampling. Lower values make the model more deterministic, while higher values make the model more random. Zero means greedy sampling.",
            ge=0.01,
            le=5,
            default=0.8,
        ),
        top_p: float = Input(
            description="Float that controls the cumulative probability of the top tokens to consider. Must be in (0, 1]. Set to 1 to consider all tokens.",
            ge=0.01,
            le=1.0,
            default=0.95,
        ),
        top_k: int = Input(
            description="Integer that controls the number of top tokens to consider. Set to -1 to consider all tokens.",
            default=-1,
        ),
        stop: str = Input(
            description="List of strings that stop the generation when they are generated. The returned output will not contain the stop strings.",
            default=None,
        )
    ) -> str:
        sampling_params = SamplingParams(
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            stop=stop,
            max_tokens=max_tokens
        )
        start = time.time()
        outputs = self.llm.generate([prompt], sampling_params)
        print(f"\nGenerated {len(outputs[0].outputs[0].token_ids)} tokens in {time.time() - start} seconds.")
        return outputs[0].outputs[0].text
