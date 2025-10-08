"""
IP-based access control middleware for dashboard
"""

import ipaddress
from typing import List, Callable
from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from ..utils.logging import setup_logging

logger = setup_logging()


class IPWhitelistMiddleware(BaseHTTPMiddleware):
    """Middleware to restrict access based on IP address"""

    def __init__(self, app, allowed_ips: List[str], protected_paths: List[str]):
        """
        Initialize IP whitelist middleware

        Args:
            app: FastAPI application
            allowed_ips: List of allowed IP addresses/ranges/patterns
                - Specific IPs: "127.0.0.1", "192.168.1.100"
                - CIDR ranges: "10.0.0.0/24", "192.168.1.0/24"
                - Wildcard patterns: "10.0.0.*", "192.168.*.*"
                - Allow all: ["all"] or ["*"]
            protected_paths: List of path prefixes to protect (e.g., ["/dashboard"])
        """
        super().__init__(app)
        self.allowed_ips = allowed_ips
        self.protected_paths = protected_paths
        self.allow_all = "all" in allowed_ips or "*" in allowed_ips

        # Parse IP configurations
        self.specific_ips = set()
        self.ip_networks = []
        self.wildcard_patterns = []

        if not self.allow_all:
            self._parse_ip_configurations()

        logger.info("IP whitelist middleware initialized",
                   allowed_ips=allowed_ips,
                   protected_paths=protected_paths,
                   allow_all=self.allow_all)

    def _parse_ip_configurations(self):
        """Parse and categorize IP configurations"""
        for ip_config in self.allowed_ips:
            ip_config = ip_config.strip()

            # Wildcard pattern (e.g., "10.0.0.*" or "192.168.*.*")
            if '*' in ip_config:
                self.wildcard_patterns.append(ip_config)
                logger.debug("Added wildcard pattern", pattern=ip_config)

            # CIDR range (e.g., "10.0.0.0/24")
            elif '/' in ip_config:
                try:
                    network = ipaddress.ip_network(ip_config, strict=False)
                    self.ip_networks.append(network)
                    logger.debug("Added CIDR network", network=ip_config)
                except ValueError as e:
                    logger.error("Invalid CIDR range", range=ip_config, error=str(e))

            # Specific IP address
            else:
                try:
                    ip = ipaddress.ip_address(ip_config)
                    self.specific_ips.add(ip)
                    logger.debug("Added specific IP", ip=ip_config)
                except ValueError as e:
                    logger.error("Invalid IP address", ip=ip_config, error=str(e))

    def _is_ip_allowed(self, client_ip: str) -> bool:
        """Check if client IP is allowed"""
        if self.allow_all:
            return True

        try:
            ip = ipaddress.ip_address(client_ip)

            # Check specific IPs
            if ip in self.specific_ips:
                return True

            # Check CIDR ranges
            for network in self.ip_networks:
                if ip in network:
                    return True

            # Check wildcard patterns
            for pattern in self.wildcard_patterns:
                if self._match_wildcard(client_ip, pattern):
                    return True

            return False

        except ValueError:
            logger.error("Invalid client IP address", client_ip=client_ip)
            return False

    def _match_wildcard(self, ip: str, pattern: str) -> bool:
        """Match IP against wildcard pattern"""
        ip_parts = ip.split('.')
        pattern_parts = pattern.split('.')

        if len(ip_parts) != 4 or len(pattern_parts) != 4:
            return False

        for ip_part, pattern_part in zip(ip_parts, pattern_parts):
            if pattern_part != '*' and ip_part != pattern_part:
                return False

        return True

    def _is_protected_path(self, path: str) -> bool:
        """Check if the request path is protected"""
        for protected_path in self.protected_paths:
            if path.startswith(protected_path):
                return True
        return False

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request and check IP access"""
        # Check if this path is protected
        if not self._is_protected_path(request.url.path):
            return await call_next(request)

        # Get client IP
        client_ip = request.client.host if request.client else "unknown"

        # Check forwarded headers (in case behind proxy)
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            client_ip = forwarded_for.split(',')[0].strip()

        # Check if IP is allowed
        if self._is_ip_allowed(client_ip):
            logger.debug("Access granted", client_ip=client_ip, path=request.url.path)
            return await call_next(request)
        else:
            logger.warning("Access denied", client_ip=client_ip, path=request.url.path)
            return JSONResponse(
                status_code=403,
                content={
                    "error": "Access denied",
                    "message": f"IP address {client_ip} is not authorized to access this resource"
                }
            )


def parse_ip_list(ip_string: str) -> List[str]:
    """
    Parse comma-separated IP configuration string

    Args:
        ip_string: Comma-separated IP addresses/ranges (e.g., "127.0.0.1,10.0.0.*,192.168.1.0/24")

    Returns:
        List of IP configurations
    """
    if not ip_string or ip_string.strip().lower() in ["all", "*"]:
        return ["all"]

    return [ip.strip() for ip in ip_string.split(',') if ip.strip()]
