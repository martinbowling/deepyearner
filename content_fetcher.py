from typing import Dict, List, Optional, Any, Tuple
import logging
import httpx
import asyncio
from dataclasses import dataclass
from datetime import datetime
import json
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log
)
from search_system import SearchSystem, SearchResult
from memory_system import MemorySystem, Episode
from content_processor import ContentProcessor, ProcessedContent
from config import ModelType, get_model_name

logger = logging.getLogger(__name__)

class ContentFetchError(Exception):
    """Base exception for content fetching errors"""
    pass

@dataclass
class FetchedContent:
    """Represents fetched and processed content"""
    url: str
    title: str
    markdown_content: str
    summary: str
    timestamp: datetime
    source_type: str
    topics: List[str]
    relevance_score: float
    search_context: Optional[Dict] = None

class ContentAnalyzer:
    """Analyzes content relevance and generates summaries"""
    
    def __init__(self, anthropic_client: Any):
        self.client = anthropic_client
        
    async def analyze_search_results(
        self,
        results: List[SearchResult],
        query_context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Analyze search results and recommend which to fetch"""
        
        # Prepare results for analysis
        results_data = [
            {
                'title': r.title,
                'url': r.url,
                'snippet': r.snippet,
                'domain': r.domain,
                'position': r.position
            }
            for r in results
        ]
        
        # Create analysis prompt
        prompt = f"""You are helping analyze search results to determine which content would be most valuable to read in full.

Context:
{json.dumps(query_context, indent=2)}

Search Results:
{json.dumps(results_data, indent=2)}

For each result, analyze:
1. Relevance to the context
2. Potential value of full content
3. Credibility of source
4. Freshness/timeliness

Recommend which articles to read in full, explaining why.

Return your analysis in this JSON format:
{{
    "recommended_reads": [
        {{
            "url": "url",
            "relevance_score": 0.0-1.0,
            "reasoning": "explanation",
            "expected_value": "what we expect to learn"
        }}
    ],
    "analysis": "overall analysis of results"
}}"""
        
        try:
            response = await self.client.messages.create(
                model="claude-3-opus-20240229",
                max_tokens=1500,
                temperature=0,
                messages=[{
                    "role": "user",
                    "content": prompt
                }]
            )
            
            return json.loads(response.content[0].text)
            
        except Exception as e:
            logger.error(f"Analysis failed: {str(e)}")
            return {
                "recommended_reads": [],
                "analysis": f"Analysis failed: {str(e)}"
            }

class ContentFetcher:
    """Fetches and processes web content"""
    
    def __init__(
        self,
        markdowner_api_key: str,
        memory_system: MemorySystem,
        anthropic_client: Any,
        http_timeout: float = 30.0
    ):
        self.markdowner_api_key = markdowner_api_key
        self.memory_system = memory_system
        self.analyzer = ContentAnalyzer(anthropic_client)
        self.processor = ContentProcessor(anthropic_client)
        self.timeout = http_timeout
        
    @retry(
        retry=retry_if_exception_type((httpx.TimeoutException, ContentFetchError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        before_sleep=before_sleep_log(logger, logging.WARNING)
    )
    async def fetch_markdown(self, url: str) -> str:
        """Fetch markdown content for URL"""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                response = await client.get(
                    'https://md.dhr.wtf/',
                    params={'url': url},
                    headers={
                        'Authorization': f'Bearer {self.markdowner_api_key}'
                    }
                )
                response.raise_for_status()
                return response.text
                
            except httpx.TimeoutException:
                logger.warning(f"Timeout fetching markdown for {url}")
                raise
                
            except httpx.HTTPStatusError as e:
                logger.error(
                    f"HTTP error fetching markdown: {e.response.status_code} - {e.response.text}"
                )
                raise ContentFetchError(
                    f"Failed to fetch markdown: {str(e)}"
                )
                
            except Exception as e:
                logger.error(f"Error fetching markdown: {str(e)}")
                raise ContentFetchError(str(e))
                
    async def process_search_results(
        self,
        search_results: List[SearchResult],
        query_context: Dict[str, Any]
    ) -> List[ProcessedContent]:
        """Process search results and fetch recommended content"""
        
        # Analyze results
        analysis = await self.analyzer.analyze_search_results(
            search_results,
            query_context
        )
        
        # Fetch recommended content
        processed_content = []
        for recommendation in analysis['recommended_reads']:
            try:
                url = recommendation['url']
                markdown = await self.fetch_markdown(url)
                
                # Find original result
                result = next(
                    (r for r in search_results if r.url == url),
                    None
                )
                
                if result and markdown:
                    # Process content
                    processed = await self.processor.process_content(
                        content=markdown,
                        title=result.title,
                        url=url,
                        context=query_context
                    )
                    
                    # Store in memory
                    self.memory_system.add_episode(
                        type="processed_content",
                        content={
                            'url': url,
                            'title': result.title,
                            'summary': processed.summary,
                            'topics': processed.topics,
                            'key_points': processed.key_points,
                            'entities': processed.entities,
                            'sentiment': processed.sentiment,
                            'readability': processed.readability_score,
                            'technical_level': processed.technical_level,
                            'markdown': markdown,
                            'relevance_score': recommendation['relevance_score'],
                            'expected_value': recommendation['expected_value']
                        },
                        context={
                            'query_context': query_context,
                            'analysis': analysis['analysis']
                        },
                        participants=set()
                    )
                    
                    processed_content.append(processed)
                    
            except Exception as e:
                logger.error(f"Failed to process content for {url}: {str(e)}")
                continue
                
        return processed_content
        
    async def get_related_content(
        self,
        query: str,
        context: Dict[str, Any]
    ) -> Tuple[List[ProcessedContent], List[Episode]]:
        """Get both new and remembered content"""
        
        # First check memory
        relevant_episodes = self.memory_system.get_relevant_episodes(
            query,
            type="processed_content",
            limit=5
        )
        
        # Prepare search context
        search_context = {
            'query': query,
            'original_context': context,
            'known_content': [
                {
                    'url': e.content['url'],
                    'title': e.content['title'],
                    'summary': e.content['summary']
                }
                for e in relevant_episodes
            ]
        }
        
        # Search for new content
        config = SearchConfig(api_key="your_api_key")  # TODO: Get from config
        async with SearchSystem(config) as search:
            results = await search.search_with_context(
                query,
                search_context,
                max_results=10
            )
            
            # Process results
            new_content = await self.process_search_results(
                results,
                search_context
            )
            
        return new_content, relevant_episodes

async def integrate_content(
    content_fetcher: ContentFetcher,
    query: str,
    context: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Integrate new and remembered content
    
    Args:
        content_fetcher: Initialized ContentFetcher
        query: Search query
        context: Query context
        
    Returns:
        Dict containing analysis and content
    """
    try:
        # Get content
        new_content, remembered = await content_fetcher.get_related_content(
            query, context
        )
        
        # Combine results
        all_content = {
            'new_content': [
                {
                    'url': c.url,
                    'title': c.title,
                    'summary': c.summary,
                    'relevance': c.relevance_score,
                    'markdown': c.markdown_content
                }
                for c in new_content
            ],
            'remembered_content': [
                {
                    'url': e.content['url'],
                    'title': e.content['title'],
                    'summary': e.content['summary'],
                    'markdown': e.content['markdown']
                }
                for e in remembered
            ],
            'analysis': {
                'total_new': len(new_content),
                'total_remembered': len(remembered),
                'query': query,
                'context': context
            }
        }
        
        return all_content
        
    except Exception as e:
        logger.error(f"Content integration failed: {str(e)}")
        return {
            'error': str(e),
            'new_content': [],
            'remembered_content': [],
            'analysis': {
                'query': query,
                'context': context
            }
        }
