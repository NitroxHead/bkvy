"""
Web search tool - search the web and return results
"""

import aiohttp
import json
from typing import List, Any, Optional, Dict

from ..base import BaseTool
from ...models.agent_schemas import ToolParameter, ToolParameterType
from ...utils.logging import setup_logging

logger = setup_logging()


class WebSearchTool(BaseTool):
    """Search the web and return results"""

    name = "web_search"
    description = "Search the web for information. Returns a list of search results with titles, URLs, and snippets. Use this to find current information or discover relevant web pages."
    timeout_seconds = 30.0

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="query",
                type=ToolParameterType.STRING,
                description="The search query",
                required=True
            ),
            ToolParameter(
                name="num_results",
                type=ToolParameterType.INTEGER,
                description="Number of results to return (default: 5, max: 10)",
                required=False,
                default=5
            )
        ]

    async def _execute(self, query: str, num_results: int = 5) -> str:
        """
        Execute a web search.

        NOTE: This is a placeholder implementation. To make this functional,
        integrate with a search API such as:
        - SerpAPI (https://serpapi.com)
        - Brave Search API (https://brave.com/search/api/)
        - Google Custom Search API
        - Bing Search API
        """

        num_results = min(max(1, num_results), 10)

        logger.info("Executing web search", query=query, num_results=num_results)

        # Check for configured search API
        import os
        search_api = os.getenv("SEARCH_API", "").lower()
        search_api_key = os.getenv("SEARCH_API_KEY", "")

        if search_api == "serpapi" and search_api_key:
            return await self._search_serpapi(query, num_results, search_api_key)
        elif search_api == "brave" and search_api_key:
            return await self._search_brave(query, num_results, search_api_key)
        else:
            # Return placeholder response
            return self._placeholder_response(query)

    async def _search_serpapi(self, query: str, num_results: int, api_key: str) -> str:
        """Search using SerpAPI"""
        url = "https://serpapi.com/search"
        params = {
            "q": query,
            "api_key": api_key,
            "num": num_results,
            "engine": "google"
        }

        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=self.timeout_seconds)) as response:
                if response.status != 200:
                    raise Exception(f"SerpAPI error: HTTP {response.status}")

                data = await response.json()

        results = []
        for item in data.get("organic_results", [])[:num_results]:
            results.append({
                "title": item.get("title", ""),
                "url": item.get("link", ""),
                "snippet": item.get("snippet", "")
            })

        return self._format_results(query, results)

    async def _search_brave(self, query: str, num_results: int, api_key: str) -> str:
        """Search using Brave Search API"""
        url = "https://api.search.brave.com/res/v1/web/search"
        headers = {
            "X-Subscription-Token": api_key,
            "Accept": "application/json"
        }
        params = {
            "q": query,
            "count": num_results
        }

        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers, params=params, timeout=aiohttp.ClientTimeout(total=self.timeout_seconds)) as response:
                if response.status != 200:
                    raise Exception(f"Brave Search error: HTTP {response.status}")

                data = await response.json()

        results = []
        for item in data.get("web", {}).get("results", [])[:num_results]:
            results.append({
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "snippet": item.get("description", "")
            })

        return self._format_results(query, results)

    def _placeholder_response(self, query: str) -> str:
        """Return a placeholder response when no search API is configured"""
        return f"""Web Search Results for: "{query}"

NOTE: No search API is configured. To enable web search, set the following environment variables:

For SerpAPI:
  SEARCH_API=serpapi
  SEARCH_API_KEY=your_api_key

For Brave Search:
  SEARCH_API=brave
  SEARCH_API_KEY=your_api_key

Without a configured search API, this tool cannot perform actual web searches.
Consider using the web_fetch tool if you have a specific URL to retrieve."""

    def _format_results(self, query: str, results: List[Dict[str, str]]) -> str:
        """Format search results as readable text"""
        if not results:
            return f"No results found for: {query}"

        output = [f"Web Search Results for: \"{query}\"\n"]

        for i, result in enumerate(results, 1):
            output.append(f"{i}. {result['title']}")
            output.append(f"   URL: {result['url']}")
            if result['snippet']:
                output.append(f"   {result['snippet']}")
            output.append("")

        return "\n".join(output)
