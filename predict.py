import time
from vllm import LLM, SamplingParams
import torch
from cog import BasePredictor, Input

class Predictor(BasePredictor):

    def setup(self):
        self.llm = LLM(
            model="./models/<model_name>",
            quantization="awq",
            dtype="auto",
            gpu_memory_utilization=0.8,
            max_num_batched_tokens=32768,
            max_model_len=512,
            tensor_parallel_size=1
        )

    def predict(
        self,
        prompt: str = Input(description=f"Text prompt to send to the model."),
        max_new_tokens: int = Input(
            description="The maximum number of tokens the model should generate as output.",
            default=128,
        ),
        temperature: float = Input(
            description="The value used to modulate the next token probabilities.",
            ge=0.01,
            le=5,
            default=0.8,
        ),
        top_p: float = Input(
            description="A probability threshold for generating the output. If < 1.0, only keep the top tokens with cumulative probability >= top_p (nucleus filtering). Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751).",
            ge=0.01,
            le=1.0,
            default=0.95
        ),
        top_k: int = Input(
            description="The number of highest probability tokens to consider for generating the output. If > 0, only keep the top k tokens with highest probability (top-k filtering).",
            default=50,
        ),
        presence_penalty: float = Input(
            description="Presence penalty",
            ge=0.01,
            le=5,
            default=1.0
        ),
    ) -> str:
        sampling_params = SamplingParams(
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            presence_penalty=presence_penalty
        )
        start = time.time()
        outputs = self.llm.generate([prompt], sampling_params)
        print(f"\nGenerated {len(outputs[0].outputs[0].token_ids)} tokens in {time.time() - start} seconds.")
        return outputs[0].outputs[0].text
