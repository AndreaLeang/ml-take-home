import torch
import transformers as tr
import logging
from dataclasses import dataclass
from typing import Optional

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

@dataclass
class ContrastiveDecodingConfig:
    """Configuration for Contrastive Decoding as per https://arxiv.org/pdf/2210.15097"""
    temperature: float = 0.6  # Temperature for softmax
    alpha: float = 0.1       # Threshold coefficient for expert probabilities
    top_k: int = 50         # Consider only top-k tokens for efficiency
    max_tokens: int = 100    # Maximum number of tokens to generate
    epsilon: float = 1e-10   # Small constant for numerical stability

def load_model(model_path: str, device: torch.device) -> tr.PreTrainedModel:
    """
    Load and configure a model for inference.
    
    Args:
        model_path: Path to the model
        device: Target device for the model
        
    Returns:
        Loaded and configured model
    """
    try:
        torch_dtype = torch.float16 if device.type == "cuda" else torch.float32
        model = tr.AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map="auto" if device.type == "cuda" else None
        ).to(device)
        model.eval()
        logger.info(f"Successfully loaded model from {model_path}")
        return model
    except Exception as e:
        logger.error(f"Failed to load model {model_path}: {e}")
        raise

def get_top_k_tokens(logits: torch.Tensor, temperature: float, k: int):
    """Get top-k tokens and their probabilities."""
    probs = torch.nn.functional.softmax(logits / temperature, dim=-1)
    top_probs, top_tokens = torch.topk(probs, k=min(k, probs.size(-1)))
    return top_tokens, top_probs

def contrastive_generation(
    amateur: tr.PreTrainedModel,
    expert: tr.PreTrainedModel,
    tokenizer: tr.PreTrainedTokenizer,
    prompt: str,
    config: Optional[ContrastiveDecodingConfig] = None
) -> str:
    """
    Performs contrastive decoding using amateur and expert models as described in
    https://arxiv.org/pdf/2210.15097
    
    Args:
        amateur: The smaller model
        expert: The larger model
        tokenizer: Tokenizer for both models
        prompt: Input prompt
        config: Decoding configuration
        
    Returns:
        Generated text
    """
    config = config or ContrastiveDecodingConfig()
    device = next(expert.parameters()).device
    
    try:
        # Encode prompt
        input_ids = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=amateur.config.max_position_embeddings
        ).input_ids.to(device)
        
        generated_ids = input_ids.clone()
        
        # Generation loop
        with torch.no_grad():
            for _ in range(config.max_tokens):
                # Get context within model's window
                context_length = amateur.config.max_position_embeddings
                context = generated_ids[:, max(0, generated_ids.shape[1] - context_length):]
                
                # Get logits from both models
                expert_logits = expert(context).logits[:, -1, :]
                amateur_logits = amateur(context).logits[:, -1, :]
                
                # Get top-k tokens from expert
                expert_tokens, expert_probs = get_top_k_tokens(
                    expert_logits,
                    config.temperature,
                    config.top_k
                )
                
                # Get amateur probabilities for expert's top-k tokens
                amateur_probs = torch.nn.functional.softmax(
                    amateur_logits[:, expert_tokens[0]] / config.temperature,
                    dim=-1
                )
                
                # Compute contrastive scores
                expert_log_probs = torch.log(expert_probs + config.epsilon)
                amateur_log_probs = torch.log(amateur_probs + config.epsilon)
                contrastive_score = expert_log_probs - amateur_log_probs
                
                # Apply adaptive threshold
                threshold = config.alpha * torch.max(expert_probs)
                valid_tokens = expert_probs >= threshold
                contrastive_score[~valid_tokens] = float('-inf')
                
                # Select next token
                next_token_idx = torch.argmax(contrastive_score)
                next_token = expert_tokens[0, next_token_idx].unsqueeze(0).unsqueeze(0)
                generated_ids = torch.cat([generated_ids, next_token], dim=-1)
                
                # Check for EOS
                if next_token.item() == tokenizer.eos_token_id:
                    break
        
        # Decode and return
        return tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        
    except Exception as e:
        logger.error(f"Error during generation: {e}")
        raise

def format_prompt(
    tokenizer: tr.PreTrainedTokenizer,
    user_prompt: str,
    system_prompt: str = "You are a helpful assistant"
) -> str:
    """Format prompt using chat template."""
    return tokenizer.apply_chat_template(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        add_generation_prompt=True,
        tokenize=False,
    )

def main():
    """Main execution function."""
    try:
        # Model paths and device setup
        amateur_path = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
        expert_path = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        
        # Load models and tokenizer
        tokenizer = tr.AutoTokenizer.from_pretrained(amateur_path)
        amateur_model = load_model(amateur_path, device)
        expert_model = load_model(expert_path, device)
        
        # Example prompt
        user_prompt = """Give a very very brief docstring for the following function:\n```\nfunction updateEloScores(
            scores,
            results,
            kFactor = 4,
        ) {
            for (const result of results) {
                const { first, second, outcome } = result;
                const firstScore = scores[first] ?? 1000;
                const secondScore = scores[second] ?? 1000;
                const expectedScoreFirst = 1 / (1 + Math.pow(10, (secondScore - firstScore) / 400));
                const expectedScoreSecond = 1 / (1 + Math.pow(10, (firstScore - secondScore) / 400));
                let sa = 0.5;
                if (outcome === 1) {
                    sa = 1;
                } else if (outcome === -1) {
                    sa = 0;
                }
                scores[first] = firstScore + kFactor * (sa - expectedScoreFirst);
                scores[second] = secondScore + kFactor * (1 - sa - expectedScoreSecond);
            }
            return scores;
        }\n```"""
        
        # Format prompt and generate
        formatted_prompt = format_prompt(tokenizer, user_prompt)
        
        config = ContrastiveDecodingConfig(
            temperature=0.6,
            alpha=0.1,
            max_tokens=50
        )
        
        logger.info("Generating response using Contrastive Decoding...")
        generated_text = contrastive_generation(
            amateur_model,
            expert_model,
            tokenizer,
            formatted_prompt,
            config
        )
        
        logger.info(f"\nGenerated Text:\n{generated_text}")
        
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise

if __name__ == "__main__":
    main()
