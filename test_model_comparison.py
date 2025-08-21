#!/usr/bin/env python3
"""
Model Comparison Test using LangChain Framework

This test compares different Amazon Q models using LangChain's interface,
showing actual assistant outputs for comparison.
"""

import asyncio
import time
import sys
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_amazon_q import create_amazon_q_chat
from amazon_q_client import UserIntent


@dataclass
class ModelResponse:
    """Response from a single model."""
    model_id: str
    response: str
    response_time: float
    token_usage: Dict[str, Any] = None
    error: str = None


class ModelComparator:
    """Compare different Amazon Q models using LangChain."""
    
    def __init__(self, bearer_token: str):
        self.bearer_token = bearer_token
        self.models_to_test = [
            "claude-3.5-sonnet",
            "claude-3.7-sonnet", 
            "claude-sonnet-4"
        ]
    
    async def test_model(self, model_id: str, prompt: str, user_intent: UserIntent = None) -> ModelResponse:
        """Test a single model with a prompt."""
        try:
            # Create chat model
            chat = create_amazon_q_chat(
                bearer_token=self.bearer_token,
                model_id=model_id,
                temperature=0.7,
                max_tokens=500
            )
            
            # Add user intent if specified
            if user_intent:
                chat = chat.with_user_intent(user_intent)
            
            # Measure response time
            start_time = time.time()
            
            response = await chat.ainvoke([
                HumanMessage(content=prompt)
            ])
            
            end_time = time.time()
            response_time = end_time - start_time
            
            return ModelResponse(
                model_id=model_id,
                response=response.content,
                response_time=response_time,
                token_usage=response.usage_metadata.dict() if response.usage_metadata else None
            )
            
        except Exception as e:
            return ModelResponse(
                model_id=model_id,
                response="",
                response_time=0.0,
                error=str(e)
            )
    
    async def compare_models(self, prompt: str, task_name: str, user_intent: UserIntent = None):
        """Compare all models on a single prompt and display outputs."""
        print(f"\n{'='*80}")
        print(f"🎯 TASK: {task_name}")
        print(f"{'='*80}")
        print(f"📝 PROMPT: {prompt}")
        if user_intent:
            print(f"🎯 USER INTENT: {user_intent.value}")
        print(f"{'='*80}")
        
        responses = []
        
        # Test each model
        for model_id in self.models_to_test:
            print(f"\n🤖 Testing {model_id}...")
            response = await self.test_model(model_id, prompt, user_intent)
            responses.append(response)
            
            if response.error:
                print(f"❌ ERROR: {response.error}")
            else:
                print(f"⏱️  Response time: {response.response_time:.2f}s")
                if response.token_usage:
                    print(f"🔢 Tokens: {response.token_usage}")
        
        # Display all responses side by side
        print(f"\n{'='*80}")
        print("📋 RESPONSES COMPARISON")
        print(f"{'='*80}")
        
        for response in responses:
            print(f"\n🤖 {response.model_id.upper()}")
            print("-" * 40)
            if response.error:
                print(f"❌ ERROR: {response.error}")
            else:
                print(response.response)
                print(f"\n⏱️  Time: {response.response_time:.2f}s")
        
        return responses
    
    async def run_comparison_suite(self):
        """Run a comprehensive comparison across different types of tasks."""
        print("🚀 Amazon Q Model Comparison using LangChain Framework")
        print(f"Models: {', '.join(self.models_to_test)}")
        
        # Test 1: Simple Q&A
        await self.compare_models(
            prompt="What is the capital of France?",
            task_name="Simple Q&A"
        )
        
        # Test 2: Code Generation
        await self.compare_models(
            prompt="Write a Python function to calculate the factorial of a number using recursion.",
            task_name="Code Generation",
            user_intent=UserIntent.CODE_GENERATION
        )
        
        # Test 3: Code Explanation
        await self.compare_models(
            prompt="Explain what this Python code does: `lambda x: x**2 + 2*x + 1`",
            task_name="Code Explanation",
            user_intent=UserIntent.EXPLAIN_CODE_SELECTION
        )
        
        # Test 4: Creative Writing
        await self.compare_models(
            prompt="Write a short haiku about artificial intelligence.",
            task_name="Creative Writing"
        )
        
        # Test 5: Technical Analysis
        await self.compare_models(
            prompt="Compare the advantages and disadvantages of microservices vs monolithic architecture in 3 bullet points each.",
            task_name="Technical Analysis",
            user_intent=UserIntent.SUGGEST_ALTERNATE_IMPLEMENTATION
        )
        
        # Test 6: Problem Solving
        await self.compare_models(
            prompt="A train travels 120 miles in 2 hours. If it maintains the same speed, how long will it take to travel 300 miles? Show your work.",
            task_name="Math Problem Solving"
        )
        
        print(f"\n{'='*80}")
        print("✅ COMPARISON COMPLETE")
        print(f"{'='*80}")


async def main():
    """Main function to run the model comparison."""
    try:
        # Load bearer token
        with open('bearer_token.txt', 'r') as f:
            bearer_token = f.read().strip()
        print(f"✅ Bearer token loaded: {bearer_token[:20]}...")
    except FileNotFoundError:
        print("❌ Bearer token not found. Please run 'python extract_token.py' first.")
        return
    
    # Create comparator and run tests
    comparator = ModelComparator(bearer_token)
    
    try:
        await comparator.run_comparison_suite()
        
    except Exception as e:
        print(f"❌ Comparison failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
