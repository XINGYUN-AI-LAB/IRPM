"""
Local model inference module for PPE.
Supports direct loading of local models using vLLM for single-node multi-GPU inference.
"""
import os
import sys
import time
from typing import List, Dict, Optional, Union
from collections import deque

# Global cache for models and tokenizers
_model_cache: Dict[str, any] = {}
_tokenizer_cache: Dict[str, any] = {}
_cache_lock = None

# Cache for GPU configuration per model path
_gpu_config_cache: Dict[str, Dict[str, any]] = {}

# Batch processing queue for local inference
_batch_queue: Dict[str, deque] = {}
_batch_lock = None
_batch_size = int(os.getenv("PPE_LOCAL_BATCH_SIZE", "32"))  # Default batch size
_batch_timeout = float(os.getenv("PPE_LOCAL_BATCH_TIMEOUT", "0.1"))  # Max wait time in seconds

def _get_cache_lock():
    """Lazy initialization of lock for thread safety"""
    global _cache_lock
    if _cache_lock is None:
        import threading
        _cache_lock = threading.Lock()
    return _cache_lock

def _get_batch_lock():
    """Lazy initialization of batch processing lock"""
    global _batch_lock
    if _batch_lock is None:
        import threading
        _batch_lock = threading.Lock()
    return _batch_lock


def chat_completion_local(
    model_path: str,
    messages: List[Dict[str, str]],
    temperature: float,
    max_tokens: int,
    device_map: Optional[Union[str, Dict[str, List[int]]]] = None,
    torch_dtype: Optional[str] = None,
) -> str:
    """
    Local model inference using vLLM.
    
    This function loads models from local paths and performs inference directly,
    supporting multi-GPU setups via tensor_parallel_size.
    
    Args:
        model_path: Local path to the model directory
        messages: Chat messages in OpenAI format [{"role": "user/assistant/system", "content": "..."}]
        temperature: Sampling temperature (0.0 = deterministic)
        max_tokens: Maximum number of tokens to generate
        device_map: Device mapping strategy:
            - "cuda:0,1,2,3": Use specific GPUs (parsed to determine tensor_parallel_size)
            - "cuda:0": Use single GPU
            - None: Use environment variable PPE_LOCAL_DEVICE_MAP or default to single GPU
        torch_dtype: Model dtype (not used with vLLM, kept for compatibility)
            - None: Use environment variable PPE_LOCAL_TORCH_DTYPE (ignored)
    
    Returns:
        Generated text string
    """
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    
    # Thread-safe model loading with caching
    with _get_cache_lock():
        # Load model and tokenizer (cached per model_path)
        if model_path not in _model_cache:
            # Get configuration from environment or use defaults (only when loading model)
            if device_map is None:
                # Default to 8 GPUs if not specified via environment variable
                device_map = os.getenv("CUDA_VISIBLE_DEVICES", "cuda:0,1,2,3,4,5,6,7")
            
            # Parse GPU count from device_map (cached per model_path)
            if model_path not in _gpu_config_cache:
                num_gpus = 1
                device_ids = None
                if isinstance(device_map, str) and device_map.startswith("cuda:") and "," in device_map:
                    # Parse "cuda:0,1,2,3" -> 4 GPUs
                    try:
                        device_ids = [int(x.strip()) for x in device_map.split(":")[1].split(",")]
                        num_gpus = len(device_ids)
                    except (ValueError, IndexError) as e:
                        print(f"[local_inference] WARNING: Failed to parse device_map '{device_map}': {e}, using single GPU")
                        num_gpus = 1
                elif isinstance(device_map, str) and device_map.startswith("cuda:"):
                    # Single GPU like "cuda:0"
                    num_gpus = 1
                
                # Cache GPU config
                _gpu_config_cache[model_path] = {
                    "num_gpus": num_gpus,
                    "device_ids": device_ids,
                }
            else:
                # Use cached config
                num_gpus = _gpu_config_cache[model_path]["num_gpus"]
                device_ids = _gpu_config_cache[model_path]["device_ids"]
            
            # Set multiprocessing method for multi-GPU BEFORE loading model (required by vLLM)
            # CRITICAL: This must be set before any vLLM operations
            if num_gpus > 1:
                os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
                print(f"[local_inference] Set VLLM_WORKER_MULTIPROC_METHOD=spawn for multi-GPU (num_gpus={num_gpus})")
            
            # Validate environment variable if using multi-GPU
            if num_gpus > 1:
                if "VLLM_WORKER_MULTIPROC_METHOD" not in os.environ or os.environ["VLLM_WORKER_MULTIPROC_METHOD"] != "spawn":
                    print(f"[local_inference] WARNING: VLLM_WORKER_MULTIPROC_METHOD not set correctly for multi-GPU!")
                    print(f"[local_inference] Current value: {os.environ.get('VLLM_WORKER_MULTIPROC_METHOD', 'NOT SET')}")
                    print(f"[local_inference] This may cause torch.distributed initialization errors")
            
            # Get vLLM GPU utilization from environment variable
            # Higher values (0.95) maximize GPU memory usage and can improve throughput
            # Lower values (0.85-0.9) leave more headroom and may be more stable
            gpu_memory_utilization = float(os.getenv("PPE_LOCAL_VLLM_GPU_UTIL", "0.95"))
            
            # Get enforce_eager setting (disable flash attention for stability)
            # Set to "false" to enable flash attention (faster but may have issues with some models)
            enforce_eager = os.getenv("PPE_LOCAL_VLLM_ENFORCE_EAGER", "true").lower() == "true"
            
            # Get disable_custom_all_reduce setting (default False for better performance)
            disable_custom_all_reduce = os.getenv("PPE_LOCAL_VLLM_DISABLE_CUSTOM_ALL_REDUCE", "false").lower() == "true"
            
            # Print configuration only when loading model
            print(f"[local_inference] Loading model from {model_path}...")
            if device_ids:
                print(f"[local_inference] Multiple GPUs detected: {device_ids}, using tensor_parallel_size={num_gpus}")
            print(f"[local_inference] vLLM config: tensor_parallel_size={num_gpus}, gpu_memory_utilization={gpu_memory_utilization}, enforce_eager={enforce_eager}, disable_custom_all_reduce={disable_custom_all_reduce}")
            
            try:
                # Load model with vLLM
                # Note: VLLM_WORKER_MULTIPROC_METHOD should already be set above if num_gpus > 1
                # Use enforce_eager=True to avoid flash attention issues with some models
                # This may be slower but more stable
                llm_kwargs = {
                    "trust_remote_code": True,
                    "tensor_parallel_size": num_gpus,
                    "gpu_memory_utilization": gpu_memory_utilization,
                    "enforce_eager": enforce_eager,  # Disable flash attention to avoid assertion errors
                    "disable_custom_all_reduce": disable_custom_all_reduce,  # Configurable for performance
                }
                
                # Add max_model_len if specified (helps avoid memory issues)
                max_model_len = os.getenv("PPE_LOCAL_VLLM_MAX_MODEL_LEN", 8192)
                if max_model_len:
                    try:
                        llm_kwargs["max_model_len"] = int(max_model_len)
                        print(f"[local_inference] Using max_model_len={llm_kwargs['max_model_len']}")
                    except ValueError:
                        print(f"[local_inference] WARNING: Invalid max_model_len '{max_model_len}', ignoring")
                
                if enforce_eager:
                    print(f"[local_inference] Using enforce_eager=True (flash attention disabled for stability)")
                else:
                    print(f"[local_inference] WARNING: enforce_eager=False (flash attention enabled, may cause issues)")
                
                if disable_custom_all_reduce:
                    print(f"[local_inference] Using standard all-reduce (disable_custom_all_reduce=True)")
                else:
                    print(f"[local_inference] Using custom all-reduce for better performance (disable_custom_all_reduce=False)")
                
                # Validate model path before loading
                if not os.path.isabs(model_path):
                    raise ValueError(f"Model path must be absolute: {model_path}")
                if not os.path.exists(model_path):
                    raise FileNotFoundError(f"Model path does not exist: {model_path}")
                
                # Check if config.json exists (required for model loading)
                config_path = os.path.join(model_path, "config.json")
                if not os.path.exists(config_path):
                    raise FileNotFoundError(f"Model config.json not found at: {config_path}. Please ensure this is a valid model directory.")
                
                # Attempt to load model with error handling and fallback
                try:
                    model = LLM(model_path, **llm_kwargs)
                    
                    # Verify model was loaded with correct settings (only log if mismatch)
                    if hasattr(model, 'llm_engine') and hasattr(model.llm_engine, 'model_config'):
                        model_config = model.llm_engine.model_config
                        if hasattr(model_config, 'enforce_eager'):
                            actual_enforce_eager = model_config.enforce_eager
                            if actual_enforce_eager != enforce_eager:
                                print(f"[local_inference] WARNING: enforce_eager mismatch! Requested={enforce_eager}, Actual={actual_enforce_eager}")
                    
                    # Load tokenizer separately (for chat template)
                    tokenizer = AutoTokenizer.from_pretrained(
                        model_path,
                        trust_remote_code=True,
                    )
                    
                    _model_cache[model_path] = model
                    _tokenizer_cache[model_path] = tokenizer
                    print(f"[local_inference] Model loaded successfully with vLLM (tensor_parallel_size={num_gpus})")
                    
                except Exception as e:
                    # Check if this is a HuggingFace validation error (local path being treated as repo ID)
                    error_type = type(e).__name__
                    error_msg = str(e)
                    
                    if "HFValidationError" in error_type or "repo id must be" in error_msg.lower() or "repo_name" in error_msg.lower():
                        print(f"[local_inference] ERROR: vLLM is treating local path as HuggingFace Hub repo ID", flush=True)
                        print(f"[local_inference] Model path: {model_path}", flush=True)
                        print(f"[local_inference] Error: {error_msg}", flush=True)
                        print(f"[local_inference]", flush=True)
                        print(f"[local_inference] SOLUTION:", flush=True)
                        print(f"[local_inference]   1. Ensure the model path is an absolute path", flush=True)
                        print(f"[local_inference]   2. Verify the model directory contains config.json", flush=True)
                        print(f"[local_inference]   3. Check vLLM version compatibility (may need to upgrade)", flush=True)
                        print(f"[local_inference]   4. Try using a shorter path name (some vLLM versions have issues with long paths)", flush=True)
                        sys.stdout.flush()
                        raise ValueError(f"Failed to load local model. vLLM validation error: {error_msg}\n"
                                       f"Model path: {model_path}\n"
                                       f"This may be a vLLM version issue. Try upgrading vLLM or using a shorter model path.")
                    
                    # Handle OSError separately (for distributed/multi-GPU errors)
                    if isinstance(e, OSError):
                        error_msg = str(e)
                        if "torch.distributed" in error_msg or "tp_plan" in error_msg.lower():
                            print(f"[local_inference] ERROR: Failed to initialize torch.distributed for multi-GPU", flush=True)
                            print(f"[local_inference] Error details: {error_msg}", flush=True)
                            print(f"[local_inference]", flush=True)
                            print(f"[local_inference] DIAGNOSTIC INFORMATION:", flush=True)
                            print(f"[local_inference]   - VLLM_WORKER_MULTIPROC_METHOD: {os.environ.get('VLLM_WORKER_MULTIPROC_METHOD', 'NOT SET')}", flush=True)
                            print(f"[local_inference]   - Requested num_gpus: {num_gpus}", flush=True)
                            print(f"[local_inference]   - Device map: {device_map}", flush=True)
                            print(f"[local_inference]   - Device IDs: {device_ids}", flush=True)
                            print(f"[local_inference]", flush=True)
                            print(f"[local_inference] POSSIBLE SOLUTIONS:", flush=True)
                            print(f"[local_inference]   1. Try using single GPU: export PPE_LOCAL_DEVICE_MAP='cuda:0'", flush=True)
                            print(f"[local_inference]   2. Ensure VLLM_WORKER_MULTIPROC_METHOD is set before vLLM import", flush=True)
                            print(f"[local_inference]   3. Check if NCCL is properly installed and configured", flush=True)
                            print(f"[local_inference]   4. Try reducing GPU count: export PPE_LOCAL_DEVICE_MAP='cuda:0,1'", flush=True)
                            print(f"[local_inference]", flush=True)
                            sys.stdout.flush()
                            
                            # Offer fallback to single GPU
                            if num_gpus > 1:
                                print(f"[local_inference] ATTEMPTING FALLBACK: Retrying with single GPU (cuda:0)...", flush=True)
                                sys.stdout.flush()
                                try:
                                    # Clear the failed config from cache
                                    if model_path in _gpu_config_cache:
                                        del _gpu_config_cache[model_path]
                                    
                                    # Retry with single GPU
                                    fallback_kwargs = llm_kwargs.copy()
                                    fallback_kwargs["tensor_parallel_size"] = 1
                                    model = LLM(model_path, **fallback_kwargs)
                                    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
                                    
                                    # Update cache with single GPU config
                                    _gpu_config_cache[model_path] = {"num_gpus": 1, "device_ids": [0]}
                                    _model_cache[model_path] = model
                                    _tokenizer_cache[model_path] = tokenizer
                                    
                                    print(f"[local_inference] SUCCESS: Model loaded with single GPU fallback", flush=True)
                                    print(f"[local_inference] WARNING: Using single GPU instead of {num_gpus} GPUs. Performance may be reduced.", flush=True)
                                    sys.stdout.flush()
                                    
                                except Exception as fallback_error:
                                    print(f"[local_inference] ERROR: Fallback to single GPU also failed: {type(fallback_error).__name__}: {fallback_error}", flush=True)
                                    sys.stdout.flush()
                                    raise OSError(f"Failed to load model with multi-GPU ({num_gpus} GPUs) and single-GPU fallback. Original error: {error_msg}")
                            else:
                                # Single GPU failed, re-raise
                                raise
                    else:
                        # Other OSError, re-raise
                        raise
                else:
                    # Non-OSError exception, re-raise (HFValidationError already handled above)
                    raise
            except Exception as outer_e:
                # Catch any unexpected errors from validation or other code
                print(f"[local_inference] Unexpected error during model loading: {type(outer_e).__name__}: {outer_e}", flush=True)
                raise
        
        model = _model_cache[model_path]
        tokenizer = _tokenizer_cache[model_path]
    
    # Format messages using chat template
    if hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template:
        try:
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        except Exception as e:
            print(f"[local_inference] WARNING: Failed to apply chat template: {e}, using fallback")
            # Fallback: simple formatting
            prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
            prompt += "\nassistant:"
    else:
        # Fallback: simple formatting
        prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
        prompt += "\nassistant:"
    
    # Create sampling parameters
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        # vLLM automatically handles pad_token_id and eos_token_id
    )
    
    # Generate using vLLM
    # Note: vLLM's generate() is thread-safe and automatically batches requests
    # when multiple threads call it simultaneously. For optimal performance:
    # 1. Use moderate --parallel (e.g., 4-8) to allow vLLM to batch effectively
    # 2. vLLM will automatically batch concurrent requests for better GPU utilization
    try:
        # Reduced logging for performance (only log on errors or if debug mode)
        # vLLM generate() expects a list of prompts
        # When multiple threads call this simultaneously, vLLM batches them automatically
        outputs = model.generate([prompt], sampling_params=sampling_params)
        
        # Extract generated text
        # vLLM returns a list of RequestOutput objects
        if len(outputs) > 0 and hasattr(outputs[0], 'outputs') and len(outputs[0].outputs) > 0:
            output = outputs[0].outputs[0].text
        else:
            # Fallback: try to get text directly
            output = str(outputs[0]) if len(outputs) > 0 else ""
            print(f"[local_inference] WARNING: Unexpected output format, using fallback")
            
        return output
        
    except AssertionError as e:
        error_msg = str(e)
        print(f"[local_inference] ERROR: AssertionError during generation: {error_msg}")
        print(f"[local_inference] This may be due to:")
        print(f"[local_inference]   1. Flash attention issue (ensure enforce_eager=True)")
        print(f"[local_inference]   2. Input prompt too long (try reducing max_tokens)")
        print(f"[local_inference]   3. Concurrent access issue (try reducing --parallel)")
        print(f"[local_inference]   4. NCCL communication timeout (check GPU connectivity)")
        raise
    except AttributeError as e:
        error_msg = str(e)
        if "use_cuda_graph" in error_msg or "decode_meta" in error_msg:
            print(f"[local_inference] ERROR: AttributeError during generation: {error_msg}")
            print(f"[local_inference] This is a known issue with vLLM when enforce_eager=True and tensor_parallel_size>1")
            print(f"[local_inference] Solutions:")
            print(f"[local_inference]   1. Ensure enforce_eager=True (already set)")
            print(f"[local_inference]   2. Try reducing tensor_parallel_size or use single GPU")
            print(f"[local_inference]   3. Check vLLM version compatibility")
            print(f"[local_inference]   4. Try setting PPE_LOCAL_VLLM_ENFORCE_EAGER=false (may cause other issues)")
        else:
            print(f"[local_inference] ERROR: AttributeError during generation: {error_msg}")
        print(f"[local_inference] Prompt preview: {prompt[:200]}...")
        raise
    except Exception as e:
        print(f"[local_inference] ERROR during generation: {type(e).__name__}: {e}")
        print(f"[local_inference] Prompt preview: {prompt[:200]}...")
        raise


def chat_completion_local_batch(
    model_path: str,
    messages_list: List[List[Dict[str, str]]],
    temperature: float,
    max_tokens: int,
    device_map: Optional[Union[str, Dict[str, List[int]]]] = None,
    torch_dtype: Optional[str] = None,
    gen_n=1
) -> List[str]:
    """
    Batch local model inference using vLLM.
    
    This function processes multiple requests in a single batch for better GPU utilization.
    
    Args:
        model_path: Local path to the model directory
        messages_list: List of chat messages, each in OpenAI format
        temperature: Sampling temperature (0.0 = deterministic)
        max_tokens: Maximum number of tokens to generate
        device_map: Device mapping strategy (same as chat_completion_local)
        torch_dtype: Model dtype (not used with vLLM, kept for compatibility)
    
    Returns:
        List of generated text strings
    """
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    
    # CRITICAL: Set VLLM_WORKER_MULTIPROC_METHOD early, before any vLLM initialization
    # This must be set before importing or initializing vLLM for multi-GPU to work
    # Parse device_map first to determine if we need multi-GPU
    if device_map is None:
        # Default to 8 GPUs if not specified via environment variable
        device_map = os.getenv("CUDA_VISIBLE_DEVICES", "0,1,2,3,4")
    
    # # Validate device_map format
    # if not isinstance(device_map, str):
    #     print(f"[local_inference] WARNING: device_map must be a string, got {type(device_map)}. Using default 'cuda:0'")
    #     device_map = "cuda:0"
    #
    # if not device_map.startswith("cuda:"):
    #     print(f"[local_inference] WARNING: device_map must start with 'cuda:', got '{device_map}'. Using default 'cuda:0'")
    #     device_map = "cuda:0"
    
    # Parse GPU count early to set environment variable before model loading
    device_ids = device_map.split(',')
    num_gpus = len(device_ids)
    # if "," in device_map:
    #     try:
    #         device_ids = [int(x.strip()) for x in device_map.split(":")[1].split(",")]
    #         num_gpus = len(device_ids)
    #
    #         # Validate device IDs are non-negative and unique
    #         if any(d < 0 for d in device_ids):
    #             print(f"[local_inference] WARNING: Invalid device IDs (negative values found). Using single GPU.")
    #             num_gpus = 1
    #             device_ids = None
    #         elif len(device_ids) != len(set(device_ids)):
    #             print(f"[local_inference] WARNING: Duplicate device IDs found. Using single GPU.")
    #             num_gpus = 1
    #             device_ids = None
    #         elif num_gpus > 8:
    #             print(f"[local_inference] WARNING: Too many GPUs specified ({num_gpus}). Maximum recommended is 8.")
    #             print(f"[local_inference] Proceeding anyway, but this may cause issues.")
    #     except (ValueError, IndexError) as e:
    #         num_gpus = 1
    #         device_ids = None
    #         print(f"[local_inference] WARNING: Failed to parse device_map '{device_map}': {e}")
    #         print(f"[local_inference] Expected format: 'cuda:0,1,2,3' or 'cuda:0'. Using single GPU.")
    
    # Set environment variable BEFORE any vLLM operations (critical for multi-GPU)
    if num_gpus > 1:
        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
        print(f"[local_inference] Set VLLM_WORKER_MULTIPROC_METHOD=spawn for multi-GPU (num_gpus={num_gpus})")
    
    # Validate environment variable if using multi-GPU
    if num_gpus > 1:
        if "VLLM_WORKER_MULTIPROC_METHOD" not in os.environ:
            print(f"[local_inference] ERROR: VLLM_WORKER_MULTIPROC_METHOD not set for multi-GPU!")
            print(f"[local_inference] This will cause torch.distributed initialization errors.")
            print(f"[local_inference] Attempting to set it now...")
            os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
        elif os.environ["VLLM_WORKER_MULTIPROC_METHOD"] != "spawn":
            print(f"[local_inference] WARNING: VLLM_WORKER_MULTIPROC_METHOD is set to '{os.environ['VLLM_WORKER_MULTIPROC_METHOD']}'")
            print(f"[local_inference] For multi-GPU, it should be 'spawn'. Overriding...")
            os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
        
        # Final validation
        if os.environ.get("VLLM_WORKER_MULTIPROC_METHOD") != "spawn":
            print(f"[local_inference] ERROR: Failed to set VLLM_WORKER_MULTIPROC_METHOD correctly!")
            print(f"[local_inference] Current value: {os.environ.get('VLLM_WORKER_MULTIPROC_METHOD', 'NOT SET')}")
            print(f"[local_inference] This will likely cause torch.distributed initialization errors.")
            print(f"[local_inference] Consider using single GPU: export PPE_LOCAL_DEVICE_MAP='cuda:0'")
    
    # Get model and tokenizer (using same loading logic as single request)
    # This will use the cached model if already loaded
    with _get_cache_lock():
        # Ensure model is loaded (reuse existing logic)
        if model_path not in _model_cache:
            # Load model using single request logic (will cache it)
            # We'll just trigger the loading by calling the single request function logic
            # But we'll do it directly here to avoid circular dependency
            
            # Use the already parsed GPU config
            if model_path not in _gpu_config_cache:
                _gpu_config_cache[model_path] = {"num_gpus": num_gpus, "device_ids": device_ids}
            else:
                # Use cached config but verify consistency
                cached_num_gpus = _gpu_config_cache[model_path]["num_gpus"]
                if cached_num_gpus != num_gpus:
                    print(f"[local_inference] WARNING: GPU config mismatch! Cached: {cached_num_gpus}, Current: {num_gpus}")
                    print(f"[local_inference] Updating GPU config cache...")
                    _gpu_config_cache[model_path] = {"num_gpus": num_gpus, "device_ids": device_ids}
                else:
                    num_gpus = cached_num_gpus
                    device_ids = _gpu_config_cache[model_path]["device_ids"]
            
            gpu_memory_utilization = float(os.getenv("PPE_LOCAL_VLLM_GPU_UTIL", "0.95"))
            enforce_eager = os.getenv("PPE_LOCAL_VLLM_ENFORCE_EAGER", "true").lower() == "true"
            disable_custom_all_reduce = os.getenv("PPE_LOCAL_VLLM_DISABLE_CUSTOM_ALL_REDUCE", "false").lower() == "true"
            
            print(f"[local_inference] Loading model for batch processing: {model_path}...")
            print(f"[local_inference] GPU configuration: num_gpus={num_gpus}, device_ids={device_ids}")
            print(f"[local_inference] VLLM_WORKER_MULTIPROC_METHOD={os.environ.get('VLLM_WORKER_MULTIPROC_METHOD', 'NOT SET')}")
            
            llm_kwargs = {
                "trust_remote_code": True,
                "tensor_parallel_size": num_gpus,
                "gpu_memory_utilization": gpu_memory_utilization,
                "enforce_eager": enforce_eager,
                "disable_custom_all_reduce": disable_custom_all_reduce,
            }
            
            max_model_len = os.getenv("PPE_LOCAL_VLLM_MAX_MODEL_LEN", 8192)
            if max_model_len:
                try:
                    llm_kwargs["max_model_len"] = int(max_model_len)
                except ValueError:
                    pass
            
            # Validate model path before loading
            if not os.path.isabs(model_path):
                raise ValueError(f"Model path must be absolute: {model_path}")
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model path does not exist: {model_path}")
            
            # Check if config.json exists (required for model loading)
            config_path = os.path.join(model_path, "config.json")
            if not os.path.exists(config_path):
                raise FileNotFoundError(f"Model config.json not found at: {config_path}. Please ensure this is a valid model directory.")
            
            # Attempt to load model with error handling and fallback
            try:
                model = LLM(model_path, **llm_kwargs)
                tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
                
                _model_cache[model_path] = model
                _tokenizer_cache[model_path] = tokenizer
                print(f"[local_inference] Model loaded successfully with vLLM (tensor_parallel_size={num_gpus})")
                
            except Exception as e:
                # Check if this is a HuggingFace validation error (local path being treated as repo ID)
                error_type = type(e).__name__
                error_msg = str(e)
                
                if "HFValidationError" in error_type or "repo id must be" in error_msg.lower() or "repo_name" in error_msg.lower():
                    print(f"[local_inference] ERROR: vLLM is treating local path as HuggingFace Hub repo ID", flush=True)
                    print(f"[local_inference] Model path: {model_path}", flush=True)
                    print(f"[local_inference] Error: {error_msg}", flush=True)
                    print(f"[local_inference]", flush=True)
                    print(f"[local_inference] SOLUTION:", flush=True)
                    print(f"[local_inference]   1. Ensure the model path is an absolute path", flush=True)
                    print(f"[local_inference]   2. Verify the model directory contains config.json", flush=True)
                    print(f"[local_inference]   3. Check vLLM version compatibility (may need to upgrade)", flush=True)
                    print(f"[local_inference]   4. Try using a shorter path name (some vLLM versions have issues with long paths)", flush=True)
                    sys.stdout.flush()
                    raise ValueError(f"Failed to load local model. vLLM validation error: {error_msg}\n"
                                   f"Model path: {model_path}\n"
                                   f"This may be a vLLM version issue. Try upgrading vLLM or using a shorter model path.")
                
                # Handle OSError separately (for distributed/multi-GPU errors)
                if isinstance(e, OSError):
                    error_msg = str(e)
                    if "torch.distributed" in error_msg or "tp_plan" in error_msg.lower():
                        print(f"[local_inference] ERROR: Failed to initialize torch.distributed for multi-GPU", flush=True)
                    print(f"[local_inference] Error details: {error_msg}", flush=True)
                    print(f"[local_inference]", flush=True)
                    print(f"[local_inference] DIAGNOSTIC INFORMATION:", flush=True)
                    print(f"[local_inference]   - VLLM_WORKER_MULTIPROC_METHOD: {os.environ.get('VLLM_WORKER_MULTIPROC_METHOD', 'NOT SET')}", flush=True)
                    print(f"[local_inference]   - Requested num_gpus: {num_gpus}", flush=True)
                    print(f"[local_inference]   - Device map: {device_map}", flush=True)
                    print(f"[local_inference]   - Device IDs: {device_ids}", flush=True)
                    print(f"[local_inference]", flush=True)
                    print(f"[local_inference] POSSIBLE SOLUTIONS:", flush=True)
                    print(f"[local_inference]   1. Try using single GPU: export PPE_LOCAL_DEVICE_MAP='cuda:0'", flush=True)
                    print(f"[local_inference]   2. Ensure VLLM_WORKER_MULTIPROC_METHOD is set before vLLM import", flush=True)
                    print(f"[local_inference]   3. Check if NCCL is properly installed and configured", flush=True)
                    print(f"[local_inference]   4. Try reducing GPU count: export PPE_LOCAL_DEVICE_MAP='cuda:0,1'", flush=True)
                    print(f"[local_inference]", flush=True)
                    sys.stdout.flush()
                    
                    # Offer fallback to single GPU
                    if num_gpus > 1:
                        print(f"[local_inference] ATTEMPTING FALLBACK: Retrying with single GPU (cuda:0)...", flush=True)
                        sys.stdout.flush()
                        try:
                            # Clear the failed config from cache
                            if model_path in _gpu_config_cache:
                                del _gpu_config_cache[model_path]
                            
                            # Retry with single GPU
                            fallback_kwargs = llm_kwargs.copy()
                            fallback_kwargs["tensor_parallel_size"] = 1
                            model = LLM(model_path, **fallback_kwargs)
                            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
                            
                            # Update cache with single GPU config
                            _gpu_config_cache[model_path] = {"num_gpus": 1, "device_ids": [0]}
                            _model_cache[model_path] = model
                            _tokenizer_cache[model_path] = tokenizer
                            
                            print(f"[local_inference] SUCCESS: Model loaded with single GPU fallback", flush=True)
                            print(f"[local_inference] WARNING: Using single GPU instead of {num_gpus} GPUs. Performance may be reduced.", flush=True)
                            sys.stdout.flush()
                            
                        except Exception as fallback_error:
                            print(f"[local_inference] ERROR: Fallback to single GPU also failed: {type(fallback_error).__name__}: {fallback_error}", flush=True)
                            sys.stdout.flush()
                            raise OSError(f"Failed to load model with multi-GPU ({num_gpus} GPUs) and single-GPU fallback. Original error: {error_msg}")
                    else:
                        # Single GPU failed, re-raise
                        raise
                else:
                    # Other OSError or non-OSError exception, re-raise
                    raise
        
        model = _model_cache[model_path]
        tokenizer = _tokenizer_cache[model_path]
    
    # Format all prompts using chat template
    prompts = []
    for messages in messages_list:
        if hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template:
            try:
                prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            except Exception:
                prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
                prompt += "\nassistant:"
        else:
            prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
            prompt += "\nassistant:"
        prompts.append(prompt)
    
    # Create sampling parameters
    if gen_n > 1:
        sampling_params = SamplingParams(
            temperature=1.0,
            max_tokens=max_tokens,
            n=gen_n,
        )
    else:
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=max_tokens,
            n=gen_n,
        )

    # Generate using vLLM with batch
    # This is the key optimization: process all prompts in a single batch call
    # This allows vLLM to fully utilize GPU parallelism and significantly improve throughput
    try:
        if len(prompts) > 1:
            print(f"[local_inference] Batch generating {len(prompts)} prompts (this will be much faster than individual calls)...")
        else:
            print(f"[local_inference] Generating 1 prompt...")

        # Single batch call - vLLM will process all prompts together
        outputs = model.generate(prompts, sampling_params=sampling_params)

        # Extract generated text from all outputs
        results = [[o.outputs[i].text for i in range(gen_n)] for o in outputs]
        # results = []
        # for output in outputs:
        #     if hasattr(output, 'outputs') and len(output.outputs) > 0:
        #         # results.append([x.outputs[0].text for x in output])
        #         answers = [[o.outputs[i].text for i in range(args.voting)] for o in outputs]
        #     else:
        #         results.append(str(output) if output else "")
        
        if len(prompts) > 1:
            print(f"[local_inference] Batch generation completed: {len(results)} results")
        
        return results
        
    except AttributeError as e:
        error_msg = str(e)
        if "use_cuda_graph" in error_msg or "decode_meta" in error_msg:
            print(f"[local_inference] ERROR: AttributeError during batch generation: {error_msg}")
            print(f"[local_inference] This is a known issue with vLLM when enforce_eager=True and tensor_parallel_size>1")
            print(f"[local_inference] Solutions:")
            print(f"[local_inference]   1. Ensure enforce_eager=True (already set)")
            print(f"[local_inference]   2. Try reducing tensor_parallel_size or use single GPU")
            print(f"[local_inference]   3. Check vLLM version compatibility")
        else:
            print(f"[local_inference] ERROR: AttributeError during batch generation: {error_msg}")
        raise
    except Exception as e:
        print(f"[local_inference] ERROR during batch generation: {type(e).__name__}: {e}")
        raise


def clear_model_cache(model_path: Optional[str] = None):
    """
    Clear model cache to free memory.
    
    Args:
        model_path: If provided, only clear this specific model. Otherwise clear all.
    """
    global _model_cache, _tokenizer_cache, _gpu_config_cache
    
    with _get_cache_lock():
        if model_path:
            if model_path in _model_cache:
                del _model_cache[model_path]
                del _tokenizer_cache[model_path]
                if model_path in _gpu_config_cache:
                    del _gpu_config_cache[model_path]
                print(f"[local_inference] Cleared cache for {model_path}")
        else:
            _model_cache.clear()
            _tokenizer_cache.clear()
            _gpu_config_cache.clear()
            print(f"[local_inference] Cleared all model caches")
