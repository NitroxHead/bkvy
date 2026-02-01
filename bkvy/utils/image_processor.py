"""
Image processing utilities for vision/multimodal support
"""

import aiohttp
import base64
import binascii
import ipaddress
from urllib.parse import urlparse
from typing import Tuple, Optional
from PIL import Image
from io import BytesIO


class ImageValidator:
    """Validates image format, size, and content"""

    SUPPORTED_FORMATS = {"image/jpeg", "image/png", "image/gif", "image/webp"}
    MAX_FILE_SIZE_MB = 20
    MAX_DIMENSION = 8000  # Anthropic's strictest limit

    @staticmethod
    async def validate_base64_image(data: str, media_type: str) -> Tuple[bool, Optional[str]]:
        """
        Validate base64 encoded image

        Args:
            data: Base64-encoded image data (without data URI prefix)
            media_type: Image MIME type (e.g., 'image/jpeg')

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check media type
        if media_type not in ImageValidator.SUPPORTED_FORMATS:
            return False, f"Unsupported media type: {media_type}. Supported formats: {', '.join(ImageValidator.SUPPORTED_FORMATS)}"

        try:
            # Decode base64
            image_bytes = base64.b64decode(data)

            # Check file size
            size_mb = len(image_bytes) / (1024 * 1024)
            if size_mb > ImageValidator.MAX_FILE_SIZE_MB:
                return False, f"Image size {size_mb:.2f}MB exceeds {ImageValidator.MAX_FILE_SIZE_MB}MB limit"

            # Verify image can be opened and check dimensions
            img = Image.open(BytesIO(image_bytes))
            width, height = img.size

            if width > ImageValidator.MAX_DIMENSION or height > ImageValidator.MAX_DIMENSION:
                return False, f"Image dimensions {width}x{height} exceed {ImageValidator.MAX_DIMENSION}x{ImageValidator.MAX_DIMENSION} limit"

            return True, None

        except binascii.Error as e:
            return False, f"Invalid base64 encoding: {str(e)}"
        except Exception as e:
            return False, f"Invalid image data: {str(e)}"


class ImageFetcher:
    """Fetches images from URLs and converts to base64"""

    @staticmethod
    def _validate_url_safety(url: str) -> Tuple[bool, Optional[str]]:
        """
        Validate URL against SSRF attacks

        Args:
            url: URL to validate

        Returns:
            Tuple of (is_safe, error_message)
        """
        try:
            parsed = urlparse(url)

            # Only allow http and https schemes
            if parsed.scheme not in ('http', 'https'):
                return False, f"URL scheme '{parsed.scheme}' not allowed. Only http/https are supported."

            # Extract hostname
            hostname = parsed.hostname
            if not hostname:
                return False, "Invalid URL: no hostname found"

            # Try to resolve hostname to IP address
            try:
                # Check if hostname is already an IP address
                ip = ipaddress.ip_address(hostname)

                # Block private/internal IP ranges
                if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_multicast:
                    return False, f"Access to private/internal IP addresses is not allowed: {hostname}"

            except ValueError:
                # Hostname is a domain name, not an IP
                # Block common internal hostnames
                blocked_hostnames = ['localhost', 'metadata.google.internal', '169.254.169.254']
                if hostname.lower() in blocked_hostnames:
                    return False, f"Access to hostname '{hostname}' is not allowed"

            return True, None

        except Exception as e:
            return False, f"URL validation error: {str(e)}"

    @staticmethod
    async def fetch_url_to_base64(url: str, session: aiohttp.ClientSession) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """
        Fetch image from URL and convert to base64

        Args:
            url: Image URL (http/https)
            session: aiohttp ClientSession for making requests

        Returns:
            Tuple of (base64_data, media_type, error_message)
            If successful: (base64_data, media_type, None)
            If failed: (None, None, error_message)
        """
        # Validate URL for SSRF protection
        is_safe, error = ImageFetcher._validate_url_safety(url)
        if not is_safe:
            return None, None, f"URL validation failed: {error}"

        try:
            # Fetch image from URL with timeout
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as response:
                if response.status != 200:
                    return None, None, f"HTTP {response.status} error fetching image from {url}"

                # Get content type
                content_type = response.headers.get('Content-Type', '').lower()
                if not content_type.startswith('image/'):
                    return None, None, f"Invalid content type '{content_type}' - expected image/*"

                # Normalize content type (remove charset and other parameters)
                media_type = content_type.split(';')[0].strip()

                # Check if supported format
                if media_type not in ImageValidator.SUPPORTED_FORMATS:
                    return None, None, f"Unsupported image format: {media_type}"

                # Read image data
                image_bytes = await response.read()

                # Validate size
                size_mb = len(image_bytes) / (1024 * 1024)
                if size_mb > ImageValidator.MAX_FILE_SIZE_MB:
                    return None, None, f"Image size {size_mb:.2f}MB exceeds {ImageValidator.MAX_FILE_SIZE_MB}MB limit"

                # Convert to base64
                base64_data = base64.b64encode(image_bytes).decode('utf-8')

                return base64_data, media_type, None

        except aiohttp.ClientError as e:
            return None, None, f"Network error fetching image: {str(e)}"
        except Exception as e:
            return None, None, f"Error fetching image: {str(e)}"
