from abc import ABC, abstractmethod
import logging
from time import sleep
import statistics
import ollama

class BaseLLM(ABC):
    """
    Abstract base class for Large Language Model clients.
    Defines the common interface for all LLM clients.
    """
    @abstractmethod
    def __init__(self, model_settings: dict):
        self.model_name = model_settings.get("model_name")
        self.model_config = self._configure_model(model_settings)

    @abstractmethod
    def _configure_model(self, model_settings: dict):
        """Configures the model with specific settings."""
        pass

    @abstractmethod
    def generate(self, prompt: str, retries: int = 5, delay: float = 3.0):
        """Generates content using the LLM, with a retry mechanism."""
        pass

class OllamaClient(BaseLLM):
    """LLM client for local models served via Ollama."""

    def __init__(self, model_settings: dict):
        """
        Initializes the Ollama client.
        Assumes the Ollama service is running locally.
        """
        try:
            # Quick check to see if the Ollama service is running
            ollama.list()
        except ollama.ResponseError as e:
            raise ConnectionError("Could not connect to Ollama server. Is it running? Start command: 'ollama serve'") from e
        super().__init__(model_settings)

        # To calculate statistics
        self.token_history = {
            "inputs": [],
            "outputs": [],
        }

    def _configure_model(self, model_settings: dict):
        """
        Extracts model options from settings to be used in the API call.
        Ollama uses an 'options' dictionary for configuration.
        """
        options = {
            'temperature': model_settings.get('temperature'),
            'top_p': model_settings.get('top_p'),
            'top_k': model_settings.get('top_k'),
            'num_ctx': model_settings.get('num_ctx'),
            'repeat_penalty': model_settings.get('repeat_penalty'),
            'format': model_settings.get('format'),
        }
        # Filter out None values so we don't send empty parameters
        return {k: v for k, v in options.items() if v is not None}

    def generate(self, prompt: str = None, messages: list = None, retries: int = 5, delay: float = 3.0):
        if messages is None:
            if prompt is None:
                raise ValueError("You must pass either prompt or messages.")
            messages = [{"role": "user", "content": prompt}]

        backoff = delay

        for attempt in range(1, retries + 1):
            try:
                response = ollama.chat(
                    model=self.model_name,
                    messages=messages,
                    options=self.model_config
                )
                # The response from ollama.chat is a dictionary.
                processed_response = self.post_processing(response)

                # Statics
                input_tokens = response.get("prompt_eval_count", None)
                output_tokens = response.get("eval_count", None)
                if input_tokens is not None:
                    self.token_history["inputs"].append(input_tokens)
                if output_tokens is not None:
                    self.token_history["outputs"].append(output_tokens)

                return processed_response
            except Exception as e:
                logging.error(f"Error on Ollama call (attempt {attempt}/{retries}): {e}")
                if attempt < retries:
                    sleep(backoff)
                    backoff *= 1.5
                else:
                    logging.error(f"Ollama call failed after {retries} attempts. Stopping the program.")
                    raise e
                
    def post_processing(self, response: dict):
        content = response.get("message", {}).get("content")
        if content is None:
            logging.info(f"Ollama returned an empty response: {response}")
            return "[]"
       
        cleaned = content.strip() 

        # Remove Markdown code fences in a robust way
        if cleaned.startswith("```"):
            # Remove first line (which is ``` or ```json)
            lines = cleaned.split("\n")
            if lines:
                lines = lines[1:]  # drop first line
            # Remove last ``` if present
            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]
            cleaned = "\n".join(lines).strip()
        
        return cleaned

    def get_stats(self):
        inputs = self.token_history["inputs"]
        outputs = self.token_history["outputs"]
        tokens_stats = {
            "mean_input_tokens": round(statistics.mean(inputs), 2) if inputs else 0,
            "mean_output_tokens": round(statistics.mean(outputs), 2) if outputs else 0,
            "min_input_tokens": min(inputs) if inputs else 0,
            "max_input_tokens": max(inputs) if inputs else 0,
            "min_output_tokens": min(outputs) if outputs else 0,
            "max_output_tokens": max(outputs) if outputs else 0,
        }
        return tokens_stats

def get_llm_client(settings: dict) -> BaseLLM:
    model_name = settings["model_name"]
    
    if any(m in model_name for m in ["gemma"]):
        return OllamaClient(settings)
    else:
        raise ValueError(f"Model name not supported: {model_name}")
