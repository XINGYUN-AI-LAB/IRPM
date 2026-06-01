import openai
import time
import requests
import os
import yaml

API_MAX_RETRY = int(os.getenv("PPE_API_MAX_RETRY", "16"))
API_RETRY_SLEEP = int(os.getenv("PPE_API_RETRY_SLEEP", "10"))
API_ERROR_OUTPUT = "$ERROR$"


CUSTOM_API_MODEL_LIST = (
    ""
)

# Model-specific API URL mapping
# If a model is not in this dict, it will use CUSTOM_API_URL from environment or default
CUSTOM_API_MODEL_URL_MAP = {
    "": ""
}


def register(name, registry):
    def register_name(func):
        registry[name] = func
        return func

    return register_name

def make_config(config_file: str) -> dict:
    config_kwargs = {}
    with open(config_file, "r") as f:
        config_kwargs = yaml.load(f, Loader=yaml.SafeLoader)

    return config_kwargs

def chat_completion_openai(model, messages, temperature, max_tokens, api_dict=None):

    if api_dict:
        client = openai.OpenAI(
            base_url=api_dict["api_base"],
            api_key=api_dict["api_key"],
        )
    else:
        client = openai.OpenAI()

    output = API_ERROR_OUTPUT
    for _ in range(API_MAX_RETRY):
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=60,
            )
            output = completion.choices[0].message.content
            break
        except openai.RateLimitError as e:
            print(type(e), e)
            time.sleep(10)
        except openai.BadRequestError as e:
            print(messages)
            print(type(e), e)
        except openai.APITimeoutError as e:
            print(type(e), "The api request timed out")
        except KeyError as e:
            print(type(e), e)
            break
        except Exception as e:
            print(type(e), e)
            break
    return output


def chat_completion_nvidia(client, model, messages):
    output = API_ERROR_OUTPUT
    for _ in range(API_MAX_RETRY):
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=messages,
            )
            output = completion.choices[0].message[0].content
            break
        except openai.RateLimitError as e:
            print(type(e), e)
            time.sleep(10)
        except openai.BadRequestError as e:
            print(messages)
            print(type(e), e)
        except openai.APITimeoutError as e:
            print(type(e), "The api request timed out")
        except KeyError as e:
            print(type(e), e)
            break
        except openai.InternalServerError as e:
            continue
        except Exception as e:
            print(type(e), e)

    if output == API_ERROR_OUTPUT:
        print(f"Output Errored after {API_MAX_RETRY} tries.")
    return output

def chat_completion_nvidia_new(client, model, messages):
    output = "$ERROR$"
    for _ in range(16):
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=messages,
                )
            # print(completion)
            output = completion.choices[0].message.content
            break
        except openai.RateLimitError as e:
            print(type(e), e)
            time.sleep(10)
        except openai.BadRequestError as e:
            print(messages)
            print(type(e), e)
        except openai.APITimeoutError as e:
            print(type(e), "The api request timed out")
        except KeyError as e:
            print(type(e), e)
            break
        except openai.InternalServerError as e:
            continue
        except Exception as e:
            print(type(e), e)

    if output == "$ERROR$":
        print("Output Errored after 16 tries.")
    return output

def chat_completion_openai_azure(
    model, messages, temperature, max_tokens, api_dict=None
):
    from openai import AzureOpenAI

    api_base = api_dict["api_base"]
    client = AzureOpenAI(
        azure_endpoint=api_base,
        api_key=api_dict["api_key"],
        api_version=api_dict["api_version"],
        timeout=240,
        max_retries=2,
    )

    output = API_ERROR_OUTPUT
    for _ in range(API_MAX_RETRY):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                n=1,
                temperature=temperature,
                max_tokens=max_tokens,
                seed=42,
            )
            output = response.choices[0].message.content
            break
        except openai.RateLimitError as e:
            print(type(e), e)
            time.sleep(API_RETRY_SLEEP)
        except openai.BadRequestError as e:
            print(type(e), e)
            break
        except KeyError:
            print(type(e), e)
            break

    return output


def chat_completion_anthropic(model, messages, temperature, max_tokens, api_dict=None):
    import anthropic

    if api_dict:
        api_key = api_dict["api_key"]
    else:
        api_key = os.environ["ANTHROPIC_API_KEY"]

    sys_msg = ""
    if messages[0]["role"] == "system":
        sys_msg = messages[0]["content"]
        messages = messages[1:]

    output = API_ERROR_OUTPUT
    for _ in range(API_MAX_RETRY):
        try:
            c = anthropic.Anthropic(api_key=api_key)
            response = c.messages.create(
                model=model,
                messages=messages,
                stop_sequences=[anthropic.HUMAN_PROMPT],
                max_tokens=max_tokens,
                temperature=temperature,
                system=sys_msg,
            )
            output = response.content[0].text
            break
        except anthropic.APIError as e:
            print(type(e), e)
            time.sleep(API_RETRY_SLEEP)
        except Exception as e:
            print(e)
            break
    return output


def chat_completion_mistral(model, messages, temperature, max_tokens):
    from mistralai.client import MistralClient
    from mistralai.models.chat_completion import ChatMessage
    from mistralai.exceptions import MistralException

    api_key = os.environ["MISTRAL_API_KEY"]
    client = MistralClient(api_key=api_key)

    prompts = [
        ChatMessage(role=message["role"], content=message["content"])
        for message in messages
    ]

    output = API_ERROR_OUTPUT
    for _ in range(API_MAX_RETRY):
        try:
            chat_response = client.chat(
                model=model,
                messages=prompts,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            output = chat_response.choices[0].message.content
            break
        except MistralException as e:
            print(type(e), e)
            break

    return output


def http_completion_gemini(model, message, temperature, max_tokens):
    api_key = os.environ["GEMINI_API_KEY"]

    safety_settings = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
    ]

    output = API_ERROR_OUTPUT
    try:
        response = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}",
            json={
                "contents": [{"parts": [{"text": message}]}],
                "safetySettings": safety_settings,
                "generationConfig": {
                    "temperature": temperature,
                    "maxOutputTokens": max_tokens,
                },
            },
        )
    except Exception as e:
        print(f"**API REQUEST ERROR** Reason: {e}.")

    if response.status_code != 200:
        print(f"**API REQUEST ERROR** Reason: status code {response.status_code}.")

    output = response.json()["candidates"][0]["content"]["parts"][0]["text"]

    return output


def chat_completion_cohere(model, messages, temperature, max_tokens):
    import cohere

    co = cohere.Client(os.environ["COHERE_API_KEY"])
    assert len(messages) > 0

    template_map = {"system": "SYSTEM", "assistant": "CHATBOT", "user": "USER"}

    assert messages[-1]["role"] == "user"
    prompt = messages[-1]["content"]

    if len(messages) > 1:
        history = []
        for message in messages[:-1]:
            history.append(
                {"role": template_map[message["role"]], "message": message["content"]}
            )
    else:
        history = None

    output = API_ERROR_OUTPUT
    for _ in range(API_MAX_RETRY):
        try:
            response = co.chat(
                message=prompt,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                chat_history=history,
            )
            output = response.text
            break
        except cohere.core.api_error.ApiError as e:
            print(type(e), e)
            raise
        except Exception as e:
            print(type(e), e)
            break

    return output


# Custom API Client class to mimic OpenAI client interface
class CustomAPIClient:
    """
    Custom API client that mimics OpenAI client interface.
    Automatically converts max_tokens to maxTokens for custom API compatibility.
    """

    def __init__(self, api_key=None, base_url=None, timeout=100):
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout or int(os.getenv("CUSTOM_API_TIMEOUT", "100"))
        # Create a chat object with completions attribute to match OpenAI SDK structure
        self.chat = self.Chat(self)

    class Chat:
        """Mimics OpenAI's chat object structure"""

        def __init__(self, client):
            self._client = client
            self.completions = CustomAPIClient.ChatCompletions(client)

    class ChatCompletions:
        """Mimics OpenAI's chat.completions interface"""

        def __init__(self, client):
            self._client = client

        def create(self, model, messages, temperature=0, max_tokens=None, **kwargs):
            """
            Create a chat completion.
            Automatically converts max_tokens to maxTokens for custom API.
            """
            # Convert max_tokens to maxTokens
            payload = {
                "stream": False,
                "messages": messages,
                "temperature": temperature,
                "model": model,
            }
            if max_tokens is not None:
                payload["maxTokens"] = max_tokens  # Use maxTokens instead of max_tokens

            # Prepare headers
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._client.api_key}",
            }

            # Make the request
            url = f"{self._client.base_url}/chat/completions"
            response = requests.post(
                url, json=payload, headers=headers, timeout=self._client.timeout
            )
            response.raise_for_status()

            data = response.json()
            
            # Handle rate limiting response
            if "choices" in data and len(data["choices"]) > 0:
                content = data["choices"][0]["message"]["content"].strip()
                if content == "平台限流":
                    raise Exception("Rate limited: 平台限流")
            
            # Return a response object similar to OpenAI's response
            return CustomAPIClient.ChatCompletionResponse(data)

    class ChatCompletionResponse:
        """Mimics OpenAI's chat completion response object"""

        def __init__(self, data):
            self.data = data
            # Create choices list with message objects
            self.choices = [
                CustomAPIClient.Choice(choice_data) for choice_data in data.get("choices", [])
            ]

    class Choice:
        """Mimics OpenAI's choice object"""

        def __init__(self, choice_data):
            self.message = CustomAPIClient.Message(choice_data.get("message", {}))

    class Message:
        """Mimics OpenAI's message object"""

        def __init__(self, message_data):
            self.content = message_data.get("content", "")


def chat_completion_custom_api(model, messages, temperature, max_tokens, api_dict=None):
    """
    Chat completion function for custom API.
    Uses CustomAPIClient which provides OpenAI-compatible interface.
    """
    # Get API key and base URL from api_dict or environment variables
    if api_dict:
        api_key = api_dict.get("api_key")
        base_url = api_dict.get("api_base")
    else:
        api_key = None
        base_url = None
    
    # If base_url is not provided, check if model has a specific URL mapping
    if not base_url and model in CUSTOM_API_MODEL_URL_MAP:
        base_url = CUSTOM_API_MODEL_URL_MAP[model]
    
    client = CustomAPIClient(api_key=api_key, base_url=base_url)

    output = API_ERROR_OUTPUT
    for attempt in range(API_MAX_RETRY):
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            output = completion.choices[0].message.content
            break
        except Exception as e:
            # More informative logging so users can see hanging/timeout behavior.
            print(
                f"[custom_api] Error on attempt {attempt + 1}/{API_MAX_RETRY}: "
                f"{type(e).__name__}: {e}"
            )
            time.sleep(API_RETRY_SLEEP)
            if attempt == API_MAX_RETRY - 1:
                print(f"[custom_api] Output errored after {API_MAX_RETRY} tries.")
    
    return output


def get_generation(
    messages, temperature, api_type, api_dict, model_name, max_tokens=8192
):
    # Check if model is in custom API list
    if model_name in CUSTOM_API_MODEL_LIST:
        output = chat_completion_custom_api(
            model=model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            api_dict=api_dict,
        )
        return output

    match api_type:
        case "anthropic":
            output = chat_completion_anthropic(
                model=model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        case "mistral":
            output = chat_completion_mistral(
                model=model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        case "gemini":
            output = http_completion_gemini(
                model=model_name,
                message=messages[0]["content"],
                temperature=temperature,
                max_tokens=max_tokens,
            )
        case "azure":
            output = chat_completion_openai_azure(
                model=model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                api_dict=api_dict,
            )
        case "cohere":
            output = chat_completion_cohere(
                model=model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        case "local":
            # Import local inference module
            from utils.local_inference import chat_completion_local
            
            # vLLM automatically batches requests when multiple calls happen simultaneously
            # For optimal performance with local inference, consider reducing --parallel
            # to allow vLLM to batch requests more effectively
            output = chat_completion_local(
                model_path=model_name,  # model_name should be local path
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                device_map=None,  # Will use environment variable or default
                torch_dtype=None,  # Will use environment variable or default
            )
        case _:
            output = chat_completion_openai(
                model=model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                api_dict=api_dict,
            )
    return output
