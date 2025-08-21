#!/usr/bin/env python3
"""
LangChain Integration Examples for Amazon Q Developer

This module demonstrates various ways to use the ChatAmazonQ model with LangChain,
showcasing all the advanced features and integration patterns.
"""

import asyncio
import json
from typing import List, Dict, Any

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.tools import tool
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain.chains import LLMChain
from langchain.schema.runnable import RunnableSequence
from pydantic import BaseModel, Field

from langchain_amazon_q import (
    ChatAmazonQ,
    create_amazon_q_chat,
    create_amazon_q_chat_enterprise,
)
from amazon_q_client import AuthType, UserIntent, AgentTaskType


class WeatherResponse(BaseModel):
    """Structured response for weather queries."""
    location: str = Field(description="The location")
    temperature: int = Field(description="Temperature in Fahrenheit")
    condition: str = Field(description="Weather condition")
    humidity: int = Field(description="Humidity percentage")


@tool
def get_current_weather(location: str) -> str:
    """Get the current weather for a location."""
    # Mock weather data
    weather_data = {
        "New York": {"temp": 72, "condition": "Sunny", "humidity": 45},
        "London": {"temp": 59, "condition": "Cloudy", "humidity": 78},
        "Tokyo": {"temp": 68, "condition": "Rainy", "humidity": 82},
        "San Francisco": {"temp": 65, "condition": "Foggy", "humidity": 88},
    }
    
    data = weather_data.get(location, {"temp": 70, "condition": "Unknown", "humidity": 50})
    return f"Weather in {location}: {data['temp']}°F, {data['condition']}, {data['humidity']}% humidity"


@tool
def calculate_math(expression: str) -> str:
    """Calculate a mathematical expression safely."""
    try:
        # Simple evaluation for demo - in production, use a safer approach
        allowed_chars = set('0123456789+-*/.() ')
        if all(c in allowed_chars for c in expression):
            result = eval(expression)
            return f"The result of {expression} is {result}"
        else:
            return "Invalid expression - only basic math operations allowed"
    except Exception as e:
        return f"Error calculating {expression}: {str(e)}"


@tool
def search_web(query: str) -> str:
    """Search the web for information."""
    # Mock search results
    return f"Search results for '{query}': Found relevant information about {query}. This is a mock search result for demonstration purposes."


async def example_basic_usage():
    """Basic usage example with simple message."""
    print("=== Basic Usage Example ===")
    
    # Load bearer token
    try:
        with open('bearer_token.txt', 'r') as f:
            bearer_token = f.read().strip()
    except FileNotFoundError:
        print("Please run extract_token.py first to get your bearer token")
        return
    
    # Create chat model
    chat = create_amazon_q_chat(
        bearer_token=bearer_token,
        model_id="claude-3.5-sonnet",
        temperature=0.7,
        max_tokens=1000
    )
    
    # Simple invoke
    response = await chat.ainvoke([
        HumanMessage(content="Explain quantum computing in simple terms")
    ])
    
    print(f"Response: {response.content}")
    print(f"Token usage: {response.usage_metadata}")
    print()


async def example_streaming():
    """Streaming response example."""
    print("=== Streaming Example ===")
    
    with open('bearer_token.txt', 'r') as f:
        bearer_token = f.read().strip()
    
    chat = create_amazon_q_chat(
        bearer_token=bearer_token,
        model_id="claude-3.5-sonnet"
    )
    
    print("Streaming response:")
    async for chunk in chat.astream([
        HumanMessage(content="Write a short story about a robot learning to paint")
    ]):
        if chunk.content:
            print(chunk.content, end="", flush=True)
    
    print("\n")


async def example_with_tools():
    """Tool calling example."""
    print("=== Tool Calling Example ===")
    
    with open('bearer_token.txt', 'r') as f:
        bearer_token = f.read().strip()
    
    chat = create_amazon_q_chat(
        bearer_token=bearer_token,
        model_id="claude-3.5-sonnet"
    )
    
    # Bind tools to the model
    chat_with_tools = chat.bind_tools([
        get_current_weather,
        calculate_math,
        search_web
    ])
    
    # Test weather tool
    response = await chat_with_tools.ainvoke([
        HumanMessage(content="What's the weather like in New York and London?")
    ])
    
    print(f"Weather response: {response.content}")
    
    # Test math tool
    response = await chat_with_tools.ainvoke([
        HumanMessage(content="Calculate 15 * 23 + 47")
    ])
    
    print(f"Math response: {response.content}")
    print()


async def example_conversation_chain():
    """Multi-turn conversation example."""
    print("=== Conversation Chain Example ===")
    
    with open('bearer_token.txt', 'r') as f:
        bearer_token = f.read().strip()
    
    chat = create_amazon_q_chat(
        bearer_token=bearer_token,
        model_id="claude-3.5-sonnet",
        user_intent=UserIntent.CODE_GENERATION
    )
    
    # Create a conversation chain
    messages = [
        SystemMessage(content="You are a helpful Python programming assistant."),
        HumanMessage(content="I need to create a function that sorts a list of dictionaries by a specific key."),
    ]
    
    # First response
    response1 = await chat.ainvoke(messages)
    print(f"Assistant: {response1.content}")
    
    # Continue conversation
    messages.extend([
        response1,
        HumanMessage(content="Can you also show me how to sort in descending order?")
    ])
    
    response2 = await chat.ainvoke(messages)
    print(f"Assistant: {response2.content}")
    print()


async def example_with_prompt_template():
    """Using LangChain prompt templates."""
    print("=== Prompt Template Example ===")
    
    with open('bearer_token.txt', 'r') as f:
        bearer_token = f.read().strip()
    
    chat = create_amazon_q_chat(
        bearer_token=bearer_token,
        model_id="claude-3.5-sonnet"
    )
    
    # Create prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a {role} with expertise in {domain}."),
        ("human", "Please explain {topic} in a way that a {audience} would understand."),
    ])
    
    # Create chain
    chain = prompt | chat | StrOutputParser()
    
    # Execute chain
    response = await chain.ainvoke({
        "role": "senior software engineer",
        "domain": "distributed systems",
        "topic": "microservices architecture",
        "audience": "junior developer"
    })
    
    print(f"Explanation: {response}")
    print()


async def example_structured_output():
    """Structured output example."""
    print("=== Structured Output Example ===")
    
    with open('bearer_token.txt', 'r') as f:
        bearer_token = f.read().strip()
    
    chat = create_amazon_q_chat(
        bearer_token=bearer_token,
        model_id="claude-3.5-sonnet"
    )
    
    # Create structured output chain
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a weather assistant. Always respond with valid JSON containing location, temperature, condition, and humidity."),
        ("human", "What's the weather like in {location}?"),
    ])
    
    chain = prompt | chat | JsonOutputParser(pydantic_object=WeatherResponse)
    
    try:
        response = await chain.ainvoke({"location": "San Francisco"})
        print(f"Structured weather data: {response}")
    except Exception as e:
        print(f"Note: Structured output parsing may need refinement: {e}")
    print()


async def example_with_context():
    """Example using Amazon Q context features."""
    print("=== Context-Aware Example ===")
    
    with open('bearer_token.txt', 'r') as f:
        bearer_token = f.read().strip()
    
    # Create chat with context enabled
    chat = create_amazon_q_chat(
        bearer_token=bearer_token,
        model_id="claude-3.5-sonnet",
        user_intent=UserIntent.IMPROVE_CODE
    ).with_context(
        editor=True,
        shell=True,
        git=True,
        env=True
    )
    
    response = await chat.ainvoke([
        HumanMessage(content="Help me optimize this Python code for better performance")
    ])
    
    print(f"Context-aware response: {response.content}")
    print()


async def example_batch_processing():
    """Batch processing example."""
    print("=== Batch Processing Example ===")
    
    with open('bearer_token.txt', 'r') as f:
        bearer_token = f.read().strip()
    
    chat = create_amazon_q_chat(
        bearer_token=bearer_token,
        model_id="claude-3.5-sonnet"
    )
    
    # Batch multiple requests
    requests = [
        [HumanMessage(content="Explain machine learning in one sentence")],
        [HumanMessage(content="What is the capital of France?")],
        [HumanMessage(content="How do you make a paper airplane?")],
    ]
    
    responses = await chat.abatch(requests)
    
    for i, response in enumerate(responses):
        print(f"Response {i+1}: {response.content}")
    print()


async def example_model_selection():
    """Model selection and configuration example."""
    print("=== Model Selection Example ===")
    
    with open('bearer_token.txt', 'r') as f:
        bearer_token = f.read().strip()
    
    # Create base chat
    base_chat = create_amazon_q_chat(bearer_token=bearer_token)
    
    # List available models
    models = await base_chat.alist_available_models()
    print(f"Available models: {models}")
    
    # Test different models
    for model_id in models[:2]:  # Test first 2 models
        chat = base_chat.with_model_id(model_id)
        response = await chat.ainvoke([
            HumanMessage(content="Hello! What model are you?")
        ])
        print(f"Model {model_id}: {response.content[:100]}...")
    print()


async def example_caching():
    """Caching example."""
    print("=== Caching Example ===")
    
    with open('bearer_token.txt', 'r') as f:
        bearer_token = f.read().strip()
    
    # Create chat with caching enabled
    chat = create_amazon_q_chat(
        bearer_token=bearer_token,
        model_id="claude-3.5-sonnet"
    ).with_caching(enabled=True, ttl_seconds=300)
    
    # Same request twice - second should be faster due to caching
    message = "Explain the theory of relativity"
    
    import time
    start_time = time.time()
    response1 = await chat.ainvoke([HumanMessage(content=message)])
    time1 = time.time() - start_time
    
    start_time = time.time()
    response2 = await chat.ainvoke([HumanMessage(content=message)])
    time2 = time.time() - start_time
    
    print(f"First request time: {time1:.2f}s")
    print(f"Second request time: {time2:.2f}s")
    print(f"Responses match: {response1.content == response2.content}")
    print()


async def example_enterprise_setup():
    """Enterprise authentication example."""
    print("=== Enterprise Setup Example ===")
    
    try:
        # Create enterprise chat (requires AWS credentials)
        chat = create_amazon_q_chat_enterprise(
            region="us-east-1",
            model_id="claude-3.5-sonnet",
            # customization_arn="arn:aws:codewhisperer:us-east-1:123456789012:customization/my-model"
        )
        
        response = await chat.ainvoke([
            HumanMessage(content="Hello from enterprise setup!")
        ])
        
        print(f"Enterprise response: {response.content}")
        
    except Exception as e:
        print(f"Enterprise setup requires AWS credentials: {e}")
    print()


async def example_complex_chain():
    """Complex chain with multiple components."""
    print("=== Complex Chain Example ===")
    
    with open('bearer_token.txt', 'r') as f:
        bearer_token = f.read().strip()
    
    chat = create_amazon_q_chat(
        bearer_token=bearer_token,
        model_id="claude-3.5-sonnet"
    )
    
    # Create a complex chain that processes user input, generates code, and explains it
    analyze_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a code analysis expert. Analyze the user's request and determine what type of code they need."),
        ("human", "{user_request}"),
    ])
    
    generate_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a code generator. Based on the analysis, generate clean, well-commented code."),
        ("human", "Analysis: {analysis}\n\nGenerate code for: {user_request}"),
    ])
    
    explain_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a code educator. Explain the generated code in simple terms."),
        ("human", "Code: {code}\n\nExplain this code:"),
    ])
    
    # Create the chain
    chain = (
        {"analysis": analyze_prompt | chat | StrOutputParser(), "user_request": RunnablePassthrough()}
        | RunnablePassthrough.assign(code=generate_prompt | chat | StrOutputParser())
        | {"explanation": explain_prompt | chat | StrOutputParser(), "code": lambda x: x["code"]}
    )
    
    # Execute the chain
    result = await chain.ainvoke("I need a function to find the longest word in a sentence")
    
    print(f"Generated code:\n{result['code']}")
    print(f"\nExplanation:\n{result['explanation']}")
    print()


async def main():
    """Run all examples."""
    print("🚀 LangChain Amazon Q Integration Examples\n")
    
    examples = [
        example_basic_usage,
        example_streaming,
        example_with_tools,
        example_conversation_chain,
        example_with_prompt_template,
        example_structured_output,
        example_with_context,
        example_batch_processing,
        example_model_selection,
        example_caching,
        example_enterprise_setup,
        example_complex_chain,
    ]
    
    for example in examples:
        try:
            await example()
        except Exception as e:
            print(f"Error in {example.__name__}: {e}")
            print()
    
    print("✅ All examples completed!")


if __name__ == "__main__":
    asyncio.run(main())
