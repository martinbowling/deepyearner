"""
Centralized configuration for all AI models and related settings.
"""
from enum import Enum
from typing import Dict, Any
import os
from dataclasses import dataclass

class ModelType(Enum):
    """Types of AI models used in the system"""
    CONTENT_GENERATION = "content_generation"  # For generating text content
    CONTENT_ANALYSIS = "content_analysis"      # For analyzing content
    CHAT = "chat"                             # For interactive chat
    EMBEDDING = "embedding"                    # For generating embeddings
    VISION = "vision"                         # For image analysis
    CODE = "code"                             # For code-related tasks
    RESEARCH = "research"                     # For research tasks
    CHUNKING = "chunking"                     # For text chunking decisions
    SUMMARIZATION = "summarization"           # For text summarization

@dataclass
class ModelConfig:
    """Configuration for an AI model"""
    name: str
    provider: str
    version: str
    context_window: int
    cost_per_1k_tokens: float
    capabilities: list[str]
    recommended_tasks: list[str]

class ModelProvider(Enum):
    """AI model providers"""
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    HUGGINGFACE = "huggingface"
    COHERE = "cohere"

# Model configurations
MODELS = {
    ModelType.CONTENT_GENERATION: ModelConfig(
        name="claude-3-opus-20240229",
        provider=ModelProvider.ANTHROPIC.value,
        version="2024-02-29",
        context_window=200000,
        cost_per_1k_tokens=0.015,
        capabilities=["text generation", "reasoning", "analysis"],
        recommended_tasks=["long-form content", "detailed analysis", "research synthesis"]
    ),
    
    ModelType.CHAT: ModelConfig(
        name="claude-3-haiku-20240307",
        provider=ModelProvider.ANTHROPIC.value,
        version="2024-03-07",
        context_window=128000,
        cost_per_1k_tokens=0.003,
        capabilities=["chat", "quick responses", "basic analysis"],
        recommended_tasks=["user interaction", "quick replies", "basic queries"]
    ),
    
    ModelType.EMBEDDING: ModelConfig(
        name="text-embedding-3-small",
        provider=ModelProvider.OPENAI.value,
        version="2024-01",
        context_window=8191,
        cost_per_1k_tokens=0.00002,
        capabilities=["text embeddings", "semantic search"],
        recommended_tasks=["document indexing", "similarity search", "clustering"]
    ),
    
    ModelType.VISION: ModelConfig(
        name="claude-3-opus-20240229",  # Supports vision tasks
        provider=ModelProvider.ANTHROPIC.value,
        version="2024-02-29",
        context_window=200000,
        cost_per_1k_tokens=0.015,
        capabilities=["image analysis", "visual reasoning", "multimodal tasks"],
        recommended_tasks=["image understanding", "visual content analysis"]
    ),
    
    ModelType.CODE: ModelConfig(
        name="claude-3-opus-20240229",
        provider=ModelProvider.ANTHROPIC.value,
        version="2024-02-29",
        context_window=200000,
        cost_per_1k_tokens=0.015,
        capabilities=["code generation", "code analysis", "debugging"],
        recommended_tasks=["code review", "refactoring", "bug fixing"]
    ),
    
    ModelType.RESEARCH: ModelConfig(
        name="claude-3-opus-20240229",
        provider=ModelProvider.ANTHROPIC.value,
        version="2024-02-29",
        context_window=200000,
        cost_per_1k_tokens=0.015,
        capabilities=["research", "analysis", "synthesis"],
        recommended_tasks=["literature review", "data analysis", "research planning"]
    ),
    
    ModelType.CHUNKING: ModelConfig(
        name="claude-3-haiku-20240307",
        provider=ModelProvider.ANTHROPIC.value,
        version="2024-03-07",
        context_window=128000,
        cost_per_1k_tokens=0.003,
        capabilities=["text analysis", "content structuring"],
        recommended_tasks=["document segmentation", "content chunking"]
    ),
    
    ModelType.SUMMARIZATION: ModelConfig(
        name="claude-3-opus-20240229",
        provider=ModelProvider.ANTHROPIC.value,
        version="2024-02-29",
        context_window=200000,
        cost_per_1k_tokens=0.015,
        capabilities=["summarization", "key point extraction"],
        recommended_tasks=["document summarization", "content distillation"]
    )
}

def get_model_config(model_type: ModelType) -> ModelConfig:
    """Get configuration for a specific model type"""
    return MODELS[model_type]

def get_model_name(model_type: ModelType) -> str:
    """Get model name for a specific type"""
    return MODELS[model_type].name

def get_all_models() -> Dict[ModelType, ModelConfig]:
    """Get all model configurations"""
    return MODELS

def get_provider_models(provider: ModelProvider) -> Dict[ModelType, ModelConfig]:
    """Get all models from a specific provider"""
    return {
        model_type: config 
        for model_type, config in MODELS.items() 
        if config.provider == provider.value
    }
