from typing import Dict, List, Optional, Any, Tuple
import logging
from dataclasses import dataclass
from datetime import datetime
import json
import re
from collections import Counter
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import asyncio
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
except Exception as e:
    logger.warning(f"Failed to download NLTK data: {e}")

@dataclass
class ProcessedContent:
    """Content with extracted information"""
    url: str
    title: str
    summary: str
    topics: List[Dict[str, float]]  # topic -> relevance score
    key_points: List[str]
    entities: List[Dict[str, str]]  # entity -> type
    sentiment: Dict[str, float]
    readability_score: float
    technical_level: float
    word_count: int
    processing_timestamp: datetime
    raw_markdown: str

class ContentProcessor:
    """Processes content to extract topics, summaries, and insights"""
    
    def __init__(self, anthropic_client: Any):
        self.client = anthropic_client
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english'
        )
        
    async def process_content(
        self,
        content: str,
        title: str,
        url: str,
        context: Optional[Dict] = None
    ) -> ProcessedContent:
        """Process content and extract information"""
        
        # Clean content
        clean_text = self._clean_text(content)
        
        # Extract topics
        topics = await self._extract_topics(clean_text, context)
        
        # Generate summary and key points
        summary, key_points = await self._generate_summary(
            clean_text,
            title,
            topics
        )
        
        # Extract entities
        entities = self._extract_entities(clean_text)
        
        # Analyze sentiment and style
        sentiment = self._analyze_sentiment(clean_text)
        readability = self._calculate_readability(clean_text)
        technical_level = self._assess_technical_level(clean_text)
        
        return ProcessedContent(
            url=url,
            title=title,
            summary=summary,
            topics=topics,
            key_points=key_points,
            entities=entities,
            sentiment=sentiment,
            readability_score=readability,
            technical_level=technical_level,
            word_count=len(clean_text.split()),
            processing_timestamp=datetime.now(),
            raw_markdown=content
        )
        
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove markdown
        text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)  # Remove links
        text = re.sub(r'[#*`]', '', text)  # Remove markdown symbols
        
        # Remove code blocks
        text = re.sub(r'```[\s\S]*?```', '', text)
        text = re.sub(r'`[^`]*`', '', text)
        
        # Basic cleaning
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        text = text.strip()
        
        return text
        
    async def _extract_topics(
        self,
        text: str,
        context: Optional[Dict] = None
    ) -> List[Dict[str, float]]:
        """Extract topics using both ML and LLM approaches"""
        
        # ML-based topic extraction
        ml_topics = self._extract_topics_ml(text)
        
        # LLM-based topic extraction
        llm_topics = await self._extract_topics_llm(text, context)
        
        # Combine and normalize scores
        combined_topics = {}
        
        # Add ML topics
        for topic, score in ml_topics.items():
            combined_topics[topic] = {
                'score': score,
                'source': 'ml'
            }
            
        # Add LLM topics
        for topic in llm_topics:
            if topic['topic'] in combined_topics:
                # Average scores if topic exists
                combined_topics[topic['topic']]['score'] = (
                    combined_topics[topic['topic']]['score'] + 
                    topic['relevance']
                ) / 2
                combined_topics[topic['topic']]['source'] = 'both'
            else:
                combined_topics[topic['topic']] = {
                    'score': topic['relevance'],
                    'source': 'llm'
                }
                
        # Convert to list and sort by score
        topics_list = [
            {
                'topic': topic,
                'relevance': data['score'],
                'source': data['source']
            }
            for topic, data in combined_topics.items()
        ]
        topics_list.sort(key=lambda x: x['relevance'], reverse=True)
        
        return topics_list[:10]  # Return top 10 topics
        
    def _extract_topics_ml(self, text: str) -> Dict[str, float]:
        """Extract topics using ML approach"""
        try:
            # Tokenize and lemmatize
            tokens = word_tokenize(text.lower())
            tokens = [
                self.lemmatizer.lemmatize(token)
                for token in tokens
                if token not in self.stop_words and
                len(token) > 3 and
                token.isalnum()
            ]
            
            # Get term frequencies
            term_freq = Counter(tokens)
            
            # Calculate TF-IDF
            tfidf_matrix = self.vectorizer.fit_transform([' '.join(tokens)])
            feature_names = self.vectorizer.get_feature_names_out()
            
            # Get top terms
            scores = zip(
                feature_names,
                np.asarray(tfidf_matrix.sum(axis=0)).ravel()
            )
            sorted_scores = sorted(
                scores,
                key=lambda x: x[1],
                reverse=True
            )
            
            # Convert to dictionary with normalized scores
            max_score = sorted_scores[0][1] if sorted_scores else 1.0
            topics = {
                term: score / max_score
                for term, score in sorted_scores[:20]
            }
            
            return topics
            
        except Exception as e:
            logger.error(f"ML topic extraction failed: {e}")
            return {}
            
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def _extract_topics_llm(
        self,
        text: str,
        context: Optional[Dict] = None
    ) -> List[Dict[str, float]]:
        """Extract topics using LLM"""
        try:
            # Prepare context
            context_str = ""
            if context:
                context_str = f"\nContext:\n{json.dumps(context, indent=2)}"
                
            # Create prompt
            prompt = f"""Analyze this text and extract the main topics and themes. Consider both explicit and implicit topics.

Text:
{text[:2000]}...  # Truncate for API limits

{context_str}

Return your analysis in this JSON format:
{{
    "topics": [
        {{
            "topic": "topic name",
            "relevance": 0.0-1.0,
            "context": "why this topic is relevant"
        }}
    ]
}}"""
            
            response = await self.client.messages.create(
                model="claude-3-opus-20240229",
                max_tokens=1000,
                temperature=0,
                messages=[{
                    "role": "user",
                    "content": prompt
                }]
            )
            
            result = json.loads(response.content[0].text)
            return result.get('topics', [])
            
        except Exception as e:
            logger.error(f"LLM topic extraction failed: {e}")
            return []
            
    async def _generate_summary(
        self,
        text: str,
        title: str,
        topics: List[Dict[str, float]]
    ) -> Tuple[str, List[str]]:
        """Generate summary and key points"""
        try:
            # Prepare topics string
            topics_str = ", ".join(
                t['topic'] for t in topics[:5]
            )
            
            # Create prompt
            prompt = f"""Summarize this content and extract key points.

Title: {title}
Main Topics: {topics_str}

Content:
{text[:3000]}...  # Truncate for API limits

Provide:
1. A concise summary (2-3 sentences)
2. 3-5 key points or takeaways

Return in JSON format:
{{
    "summary": "concise summary",
    "key_points": [
        "point 1",
        "point 2",
        ...
    ]
}}"""
            
            response = await self.client.messages.create(
                model="claude-3-opus-20240229",
                max_tokens=1000,
                temperature=0,
                messages=[{
                    "role": "user",
                    "content": prompt
                }]
            )
            
            result = json.loads(response.content[0].text)
            return result.get('summary', ''), result.get('key_points', [])
            
        except Exception as e:
            logger.error(f"Summary generation failed: {e}")
            return "", []
            
    def _extract_entities(self, text: str) -> List[Dict[str, str]]:
        """Extract named entities"""
        try:
            # Use NLTK for basic NER
            tokens = word_tokenize(text)
            pos_tags = nltk.pos_tag(tokens)
            chunks = nltk.ne_chunk(pos_tags)
            
            entities = []
            current_entity = []
            current_type = None
            
            for chunk in chunks:
                if hasattr(chunk, 'label'):
                    current_entity.append(chunk[0][0])
                    current_type = chunk.label()
                else:
                    if current_entity:
                        entities.append({
                            'text': ' '.join(current_entity),
                            'type': current_type
                        })
                        current_entity = []
                        current_type = None
                        
            return entities
            
        except Exception as e:
            logger.error(f"Entity extraction failed: {e}")
            return []
            
    def _analyze_sentiment(self, text: str) -> Dict[str, float]:
        """Analyze sentiment and emotion"""
        try:
            # Simple lexicon-based sentiment
            positive_words = set([
                'good', 'great', 'excellent', 'amazing', 'wonderful',
                'fantastic', 'awesome', 'best', 'brilliant', 'outstanding'
            ])
            negative_words = set([
                'bad', 'poor', 'terrible', 'awful', 'horrible',
                'worst', 'disappointing', 'mediocre', 'failure', 'wrong'
            ])
            
            tokens = word_tokenize(text.lower())
            
            pos_count = sum(1 for t in tokens if t in positive_words)
            neg_count = sum(1 for t in tokens if t in negative_words)
            total = len(tokens)
            
            if total == 0:
                return {'sentiment': 0.0, 'confidence': 0.0}
                
            sentiment = (pos_count - neg_count) / total
            confidence = (pos_count + neg_count) / total
            
            return {
                'sentiment': max(-1.0, min(1.0, sentiment)),
                'confidence': min(1.0, confidence)
            }
            
        except Exception as e:
            logger.error(f"Sentiment analysis failed: {e}")
            return {'sentiment': 0.0, 'confidence': 0.0}
            
    def _calculate_readability(self, text: str) -> float:
        """Calculate readability score"""
        try:
            sentences = sent_tokenize(text)
            words = word_tokenize(text)
            
            if not sentences or not words:
                return 0.0
                
            # Calculate average sentence length
            avg_sentence_length = len(words) / len(sentences)
            
            # Calculate average word length
            avg_word_length = sum(len(w) for w in words) / len(words)
            
            # Simple readability score (0-1)
            # Lower score = more readable
            readability = min(
                1.0,
                (avg_sentence_length * 0.1 + avg_word_length * 0.3)
            )
            
            return 1.0 - readability  # Invert so higher is more readable
            
        except Exception as e:
            logger.error(f"Readability calculation failed: {e}")
            return 0.5
            
    def _assess_technical_level(self, text: str) -> float:
        """Assess technical complexity level"""
        try:
            # Technical indicators
            technical_indicators = set([
                'algorithm', 'implementation', 'function', 'method',
                'class', 'object', 'interface', 'api', 'database',
                'framework', 'library', 'module', 'system', 'protocol',
                'architecture', 'infrastructure', 'deployment', 'runtime'
            ])
            
            tokens = word_tokenize(text.lower())
            
            # Calculate technical term density
            technical_count = sum(
                1 for t in tokens if t in technical_indicators
            )
            
            if not tokens:
                return 0.0
                
            # Technical level score (0-1)
            technical_level = min(1.0, technical_count / len(tokens) * 10)
            
            return technical_level
            
        except Exception as e:
            logger.error(f"Technical level assessment failed: {e}")
            return 0.0
