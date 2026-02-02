"""
Web fetch tool - fetches web pages and converts to markdown
"""

import aiohttp
from typing import List, Any, Optional

from ..base import BaseTool
from ...models.agent_schemas import ToolParameter, ToolParameterType
from ...utils.logging import setup_logging

logger = setup_logging()

# Optional imports for HTML to markdown conversion
try:
    from bs4 import BeautifulSoup
    HAS_BS4 = True
except ImportError:
    HAS_BS4 = False

try:
    import html2text
    HAS_HTML2TEXT = True
except ImportError:
    HAS_HTML2TEXT = False


class WebFetchTool(BaseTool):
    """Fetch a web page and return its content as markdown"""

    name = "web_fetch"
    description = "Fetch a web page URL and return its content converted to markdown text. Useful for reading articles, documentation, or any web content."
    timeout_seconds = 30.0

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="url",
                type=ToolParameterType.STRING,
                description="The URL to fetch",
                required=True
            ),
            ToolParameter(
                name="max_length",
                type=ToolParameterType.INTEGER,
                description="Maximum length of returned content in characters (default: 50000)",
                required=False,
                default=50000
            ),
            ToolParameter(
                name="include_links",
                type=ToolParameterType.BOOLEAN,
                description="Whether to include links in the output (default: true)",
                required=False,
                default=True
            )
        ]

    async def _execute(self, url: str, max_length: int = 50000, include_links: bool = True) -> str:
        """Fetch the URL and convert to markdown"""

        # Validate URL
        if not url.startswith(("http://", "https://")):
            url = "https://" + url

        logger.info("Fetching URL", url=url)

        headers = {
            "User-Agent": "Mozilla/5.0 (compatible; BkvyAgent/1.0; +https://github.com/bkvy)"
        }

        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=self.timeout_seconds)) as response:
                if response.status != 200:
                    raise Exception(f"HTTP {response.status}: {response.reason}")

                content_type = response.headers.get("Content-Type", "")
                html_content = await response.text()

        # Convert HTML to markdown
        markdown = self._html_to_markdown(html_content, include_links)

        # Truncate if needed
        if len(markdown) > max_length:
            markdown = markdown[:max_length] + "\n\n[Content truncated...]"

        logger.info("Fetched and converted URL", url=url, content_length=len(markdown))

        return markdown

    def _html_to_markdown(self, html: str, include_links: bool = True) -> str:
        """Convert HTML to markdown"""

        # Try html2text first (best quality)
        if HAS_HTML2TEXT:
            h = html2text.HTML2Text()
            h.ignore_links = not include_links
            h.ignore_images = True
            h.ignore_emphasis = False
            h.body_width = 0  # No wrapping
            return h.handle(html)

        # Fallback to BeautifulSoup
        if HAS_BS4:
            soup = BeautifulSoup(html, "html.parser")

            # Remove script and style elements
            for element in soup(["script", "style", "nav", "footer", "header"]):
                element.decompose()

            # Get text
            text = soup.get_text(separator="\n")

            # Clean up whitespace
            lines = [line.strip() for line in text.splitlines()]
            text = "\n".join(line for line in lines if line)

            return text

        # Basic fallback - just strip tags
        import re
        text = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<[^>]+>", "", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()
