from typing import Dict, List, Optional, Any
import logging
import httpx
import asyncio
from dataclasses import dataclass
from datetime import datetime
import traceback
from enum import Enum
import json
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log
)

logger = logging.getLogger(__name__)

class SearchError(Exception):
    """Base exception for search-related errors"""
    pass

class SearchAPIError(SearchError):
    """API-specific errors"""
    def __init__(self, message: str, status_code: Optional[int] = None, response_text: Optional[str] = None):
        self.status_code = status_code
        self.response_text = response_text
        super().__init__(message)

class SearchResultType(Enum):
    """Types of search results"""
    ORGANIC = "organic"
    NEWS = "news"
    KNOWLEDGE = "knowledge_graph"
    FEATURED = "featured_snippet"

@dataclass
class SearchResult:
    """Structured search result"""
    title: str
    url: str
    snippet: str
    type: SearchResultType
    position: int
    domain: str
    timestamp: datetime
    raw_data: Dict

    @classmethod
    def from_organic(cls, data: Dict) -> 'SearchResult':
        """Create from organic search result"""
        return cls(
            title=data.get('title', ''),
            url=data.get('link', ''),
            snippet=data.get('snippet', ''),
            type=SearchResultType.ORGANIC,
            position=data.get('position', 0),
            domain=data.get('domain', ''),
            timestamp=datetime.now(),
            raw_data=data
        )

class SearchConfig:
    """Search configuration"""
    def __init__(
        self,
        api_key: str,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 10.0,
        timeout: float = 30.0,
        gl: str = 'us',
        google_domain: str = 'google.com'
    ):
        self.api_key = api_key
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.timeout = timeout
        self.gl = gl
        self.google_domain = google_domain

class SearchSystem:
    """Enhanced search system with retry logic and result processing"""
    
    def __init__(self, config: SearchConfig):
        self.config = config
        self._client: Optional[httpx.AsyncClient] = None
        
    async def __aenter__(self):
        """Context manager entry"""
        self._client = httpx.AsyncClient(timeout=self.config.timeout)
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        if self._client:
            await self._client.aclose()
            
    @retry(
        retry=retry_if_exception_type((httpx.TimeoutException, SearchAPIError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        before_sleep=before_sleep_log(logger, logging.WARNING)
    )
    async def _make_request(self, params: Dict[str, Any]) -> Dict:
        """Make API request with retry logic"""
        if not self._client:
            raise SearchError("Client not initialized. Use 'async with' context manager.")
            
        try:
            response = await self._client.get(
                'https://api.scaleserp.com/search',
                params=params
            )
            response.raise_for_status()
            return response.json()
            
        except httpx.TimeoutException:
            logger.warning("Request timed out")
            raise
            
        except httpx.HTTPStatusError as e:
            logger.error(
                f"HTTP error {e.response.status_code}: {e.response.text}"
            )
            raise SearchAPIError(
                f"API request failed",
                status_code=e.response.status_code,
                response_text=e.response.text
            )
            
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            logger.debug(f"Traceback: {traceback.format_exc()}")
            raise SearchError(f"Search failed: {str(e)}")
            
    def _process_results(self, data: Dict) -> List[SearchResult]:
        """Process raw API response into structured results"""
        results = []
        
        # Process organic results
        for item in data.get('organic_results', []):
            try:
                result = SearchResult.from_organic(item)
                results.append(result)
            except Exception as e:
                logger.warning(f"Failed to process result: {str(e)}")
                continue
                
        # Sort by position
        results.sort(key=lambda x: x.position)
        return results
        
    async def search(
        self,
        query: str,
        max_results: int = 10,
        **kwargs
    ) -> List[SearchResult]:
        """
        Perform search with enhanced parameters
        
        Args:
            query: Search query
            max_results: Maximum number of results to return
            **kwargs: Additional search parameters
            
        Returns:
            List of SearchResult objects
        """
        params = {
            'api_key': self.config.api_key,
            'q': query,
            'gl': self.config.gl,
            'google_domain': self.config.google_domain,
            **kwargs
        }
        
        logger.info(f"Searching for: {query}")
        
        try:
            data = await self._make_request(params)
            results = self._process_results(data)
            
            logger.info(
                f"Found {len(results)} results for query: {query}"
            )
            
            return results[:max_results]
            
        except Exception as e:
            logger.error(f"Search failed: {str(e)}")
            raise
            
    async def search_with_context(
        self,
        query: str,
        context: Dict[str, Any],
        max_results: int = 10
    ) -> List[SearchResult]:
        """
        Perform search with additional context
        
        Args:
            query: Base search query
            context: Additional context to enhance search
            max_results: Maximum results to return
        """
        # Enhance query with context
        enhanced_query = self._enhance_query(query, context)
        
        # Add context-specific parameters
        params = {}
        if 'time_range' in context:
            params['tbs'] = f"qdr:{context['time_range']}"
            
        if 'location' in context:
            params['location'] = context['location']
            
        return await self.search(
            enhanced_query,
            max_results=max_results,
            **params
        )
        
    def _enhance_query(self, query: str, context: Dict[str, Any]) -> str:
        """Enhance search query with context"""
        enhanced = query
        
        # Add topic-specific terms
        if 'topics' in context:
            relevant_topics = [
                t for t, score in context['topics'].items()
                if score > 0.5
            ]
            if relevant_topics:
                enhanced += f" {' '.join(relevant_topics)}"
                
        # Add time context
        if 'time_context' in context:
            enhanced += f" {context['time_context']}"
            
        return enhanced.strip()
        
async def search_and_analyze(
    search_system: SearchSystem,
    query: str,
    context: Optional[Dict] = None,
    max_results: int = 5
) -> Dict[str, Any]:
    """
    Search and analyze results
    
    Args:
        search_system: Initialized SearchSystem
        query: Search query
        context: Optional search context
        max_results: Maximum results to analyze
        
    Returns:
        Dict containing search results and analysis
    """
    try:
        if context:
            results = await search_system.search_with_context(
                query, context, max_results
            )
        else:
            results = await search_system.search(query, max_results)
            
        # Analyze results
        domains = {}
        snippets = []
        
        for result in results:
            domains[result.domain] = domains.get(result.domain, 0) + 1
            snippets.append(result.snippet)
            
        # Prepare analysis
        analysis = {
            'query': query,
            'total_results': len(results),
            'top_domains': sorted(
                domains.items(),
                key=lambda x: x[1],
                reverse=True
            ),
            'results': [
                {
                    'title': r.title,
                    'url': r.url,
                    'snippet': r.snippet,
                    'domain': r.domain,
                    'position': r.position
                }
                for r in results
            ]
        }
        
        return analysis
        
    except Exception as e:
        logger.error(f"Search and analysis failed: {str(e)}")
        logger.debug(f"Traceback: {traceback.format_exc()}")
        return {
            'error': str(e),
            'query': query,
            'total_results': 0,
            'results': []
        }
