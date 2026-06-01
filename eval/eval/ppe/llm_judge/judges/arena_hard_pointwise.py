from typing import Dict, List, Literal, Tuple
from utils.core import register
from llm_judge.judges.base import BaseJudge, judge_registry
from os.path import join
import re
import os
import numpy as np

DEFAULT_PROMPT = join("llm_judge", "prompts", "helpfulness_pointwise_default")


@register("arena-hard-pointwise", judge_registry)
class ArenaHardPointwiseJudge(BaseJudge):

    prompt_format: str = (
        """{context}\n\n{response}"""
    )

    # Pattern to match \boxed{x} format where x is a score like 5.6 (single braces, matching user's parse_point_result logic)
    # Also support legacy formats [[SCORE:X.X]] or Score: X.X for backward compatibility
    # Note: Using \\boxed\{ to match literal \boxed{ (single backslash, single brace)
    # Pattern matches: \boxed{5.6} format (single braces as per user's parse_point_result function)
    score_pattern = re.compile(r"\\boxed\{(\d+(?:\.\d+)?)\}|\[\[SCORE:(\d+(?:\.\d+)?)\]\]|Score:\s*(\d+(?:\.\d+)?)", re.IGNORECASE)

    def __init__(self, prompt_path: str = None):

        if prompt_path:

            with open(prompt_path) as pfile:
                self.instructions = pfile.read()

        else:

            with open(DEFAULT_PROMPT) as pfile:
                self.instructions = pfile.read()

    def _format(self, user_prompt: str, responses: List[str]) -> List[Dict[str, str]]:
        """
        Format prompt for pointwise evaluation.
        For pointwise, we evaluate one response at a time, so responses should contain a single response.
        """
        if len(responses) != 1:
            raise ValueError("Pointwise judge expects exactly one response in the list")
        
        response = responses[0]
        
        # Format the instructions with context and response
        # context is the user prompt (conversation context)
        formatted_content = self.instructions.format(
            context=user_prompt,
            response=response
        )
        
        return [
            {
                "role": "system",
                "content": "",  # Empty system message, all instructions are in user message
            },
            {
                "role": "user",
                "content": formatted_content,
            },
        ]

    def _parse_judgment(self, judgment: str) -> Literal[0, 1]:
        """
        Parse judgment for BaseJudge interface compatibility.
        For pointwise, we use _parse_score() instead.
        This method is kept for interface compliance but should not be used directly.
        """
        # This should not be called in pointwise mode, but we implement it for interface compliance
        score = self._parse_score(judgment)
        # Convert score to binary decision (arbitrary threshold for interface compliance)
        return 1 if score >= 5.0 else 0
    
    def _parse_score(self, judgment: str) -> float:
        """
        Parse score from judgment text.
        Returns a float between 0.0 and 10.0.
        Supports \boxed{x} format (single braces, matching user's parse_point_result logic) 
        and legacy formats [[SCORE:X.X]] or Score: X.X.
        """
        matches = ArenaHardPointwiseJudge.score_pattern.findall(judgment)
        
        if len(matches) == 0:
            raise Exception("Did not find any matching score patterns in the judgment")
        
        # Extract score from the last match
        # Match groups: (\boxed{x}, [[SCORE:x]], Score: x)
        last_match = matches[-1]
        # Try \boxed{x} first (group 0), then [[SCORE:x]] (group 1), then Score: x (group 2)
        score_str = None
        if last_match[0]:  # \boxed{x} format (single braces)
            score_str = last_match[0]
        elif last_match[1]:  # [[SCORE:x]] format
            score_str = last_match[1]
        elif last_match[2]:  # Score: x format
            score_str = last_match[2]
        
        if score_str is None:
            raise Exception("Could not extract score from match groups")
        
        try:
            score = float(score_str)
        except ValueError:
            raise Exception(f"Could not parse score as float: {score_str}")
        
        # Validate score range
        if score < 0.0 or score > 10.0:
            raise Exception(f"Score {score} is out of valid range [0.0, 10.0]")
        
        # Round to one decimal place
        score = round(score, 1)
        
        # Ensure return Python native float (not numpy scalar)
        return float(score)

    def _judge_single_response(
        self,
        user_prompt: str,
        response: str,
        temperature: float,
        api_type: str,
        api_dict: Dict,
        model_name: str,
        max_tokens: int,
        is_refinement: bool = False,
    ) -> Tuple[float, str]:
        """
        Judge a single response and return score and judgment.
        
        Args:
            is_refinement: If True, this is a refinement pass for close scores
        """
        # Create a refined prompt for close-score scenarios
        if is_refinement:
            # Temporarily modify instructions for refinement
            original_instructions = self.instructions
            refinement_note = (
                "\n\n**IMPORTANT: This is a refinement evaluation.** "
                "The scores for two responses were very close. Please carefully "
                "re-evaluate this response with extra attention to subtle quality "
                "differences. Be precise in your scoring to ensure accurate comparison."
            )
            self.instructions = original_instructions + refinement_note
        
        try:
            messages = self._format(user_prompt, [response])
            judgment = self._gen_judgment(
                messages,
                temperature=temperature,
                api_type=api_type,
                api_dict=api_dict,
                model_name=model_name,
                max_tokens=max_tokens,
            )
            
            score = self._parse_score(judgment)
            
            return score, judgment
        finally:
            # Restore original instructions if modified
            if is_refinement:
                self.instructions = original_instructions

    def judge_pointwise(
        self,
        user_prompt: str,
        responses: List[str],
        temperature: float,
        api_type: str,
        api_dict: Dict,
        model_name: str,
        max_tokens: int = 8192,
    ) -> Tuple[float, float, Literal[0, 1], str, str]:
        """
        Judge two responses using pointwise scoring.
        
        Optional two-stage refinement: If enabled via environment variable
        PPE_POINTWISE_TWO_STAGE=true and initial score gap is small (< 0.3),
        performs a second round of evaluation for more accurate comparison.
        
        Returns:
            (score_1, score_2, decision, judgment_1, judgment_2)
            - score_1: float, score for response_1 (0.0-10.0)
            - score_2: float, score for response_2 (0.0-10.0)
            - decision: 1 if score_1 > score_2, 0 if score_1 <= score_2
            - judgment_1: str, full judgment text for response_1
            - judgment_2: str, full judgment text for response_2
        """
        if len(responses) != 2:
            raise ValueError("judge_pointwise requires exactly 2 responses")
        
        response_1 = responses[0]
        response_2 = responses[1]
        
        # Check if two-stage refinement is enabled
        two_stage_enabled = os.getenv("PPE_POINTWISE_TWO_STAGE", "false").lower() == "true"
        two_stage_threshold = float(os.getenv("PPE_POINTWISE_TWO_STAGE_THRESHOLD", "0.3"))
        
        # Helper function to ensure Python native float (defined once at the start)
        def ensure_python_float(value):
            """Convert any numeric type to Python native float."""
            # If it's already a Python float/int, return it
            if isinstance(value, (int, float)) and not isinstance(value, (np.integer, np.floating)):
                return float(value)
            # If it's a numpy scalar, extract the value
            if isinstance(value, (np.integer, np.floating)):
                return float(value.item())
            # If it's a numpy array, extract scalar
            if isinstance(value, np.ndarray):
                if value.size == 0:
                    raise ValueError("Cannot convert empty array to float")
                elif value.size == 1:
                    return float(value.item())
                else:
                    raise ValueError(f"Cannot convert multi-element array (shape {value.shape}) to float")
            # Try direct conversion
            try:
                result = float(value)
                # Double-check it's not still a numpy type
                if isinstance(result, (np.integer, np.floating)):
                    return float(result.item())
                if isinstance(result, np.ndarray):
                    return float(result.item())
                return result
            except (TypeError, ValueError):
                # Last resort: try item() method
                if hasattr(value, 'item'):
                    return float(value.item())
                raise ValueError(f"Cannot convert {type(value)} to float: {value}")
        
        # First round: Judge both responses
        score_1, judgment_1 = self._judge_single_response(
            user_prompt, response_1, temperature, api_type, api_dict, model_name, max_tokens
        )
        
        score_2, judgment_2 = self._judge_single_response(
            user_prompt, response_2, temperature, api_type, api_dict, model_name, max_tokens
        )
        
        # Ensure scores are Python native floats before any operations
        score_1 = ensure_python_float(score_1)
        score_2 = ensure_python_float(score_2)
        
        # score_diff = abs(score_1 - score_2)
        
        # # Two-stage refinement for close scores (optional)
        # if two_stage_enabled and score_diff < two_stage_threshold:
        #     print(
        #         f"[pointwise] Two-stage refinement triggered: initial gap {score_diff:.1f} < {two_stage_threshold}"
        #     )
        #
        #     # Refinement round: Re-evaluate both responses with enhanced attention
        #     score_1_refined, judgment_1_refined = self._judge_single_response(
        #         user_prompt, response_1, temperature, api_type, api_dict, model_name, max_tokens, is_refinement=True
        #     )
        #
        #     score_2_refined, judgment_2_refined = self._judge_single_response(
        #         user_prompt, response_2, temperature, api_type, api_dict, model_name, max_tokens, is_refinement=True
        #     )
        #
        #     # Use refined scores (ensure they are Python floats)
        #     score_1 = ensure_python_float(score_1_refined)
        #     score_2 = ensure_python_float(score_2_refined)
        #     judgment_1 = judgment_1_refined
        #     judgment_2 = judgment_2_refined
        #
        #     score_diff = abs(score_1 - score_2)
        #     print(
        #         f"[pointwise] After refinement: {score_1:.1f} vs {score_2:.1f} "
        #         f"(diff={score_diff:.1f})"
        #     )
        
        # Ensure scores are Python native floats for comparison (handle numpy types from correctness datasets)
        # Note: ensure_python_float is already defined above, just ensure scores are converted again before comparison
        score_1 = ensure_python_float(score_1)
        score_2 = ensure_python_float(score_2)

        # Determine decision based on scores
        # If score_1 > score_2, decision = 1 (response_1 is better)
        # If score_1 <= score_2, decision = 0 (response_2 is better or tie)
        decision: Literal[0, 1] = 1 if score_1 > score_2 else 0
        
        # Log score gap information for quality control
        # Small gaps (< 0.2) may indicate uncertain decisions
        # if score_diff < 0.2:
        #     print(
        #         f"[pointwise] Small score gap detected: {score_1:.1f} vs {score_2:.1f} "
        #         f"(diff={score_diff:.1f}). Decision: {decision}"
        #     )
        # elif score_diff < 0.5:
        #     print(
        #         f"[pointwise] Moderate score gap: {score_1:.1f} vs {score_2:.1f} "
        #         f"(diff={score_diff:.1f}). Decision: {decision}"
        #     )
        # else:
        #     print(
        #         f"[pointwise] Clear score difference: {score_1:.1f} vs {score_2:.1f} "
        #         f"(diff={score_diff:.1f}). Decision: {decision}"
        #     )
        
        return score_1, score_2, decision, judgment_1, judgment_2

