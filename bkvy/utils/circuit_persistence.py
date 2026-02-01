"""
Circuit breaker state persistence layer
"""

import os
import asyncio
from pathlib import Path
from typing import Dict, Optional
import aiofiles

from ..models.circuit_states import CircuitState
from ..utils.logging import setup_logging

logger = setup_logging()


class CircuitStatePersistence:
    """Manages persistence of circuit breaker states to disk"""

    def __init__(self, base_dir: str = "circuit_states"):
        """
        Initialize persistence layer

        Args:
            base_dir: Directory to store circuit state files
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)

        # Lock for concurrent access protection
        self._locks: Dict[str, asyncio.Lock] = {}

    def _get_lock(self, combination_key: str) -> asyncio.Lock:
        """Get or create a lock for a combination key"""
        if combination_key not in self._locks:
            self._locks[combination_key] = asyncio.Lock()
        return self._locks[combination_key]

    def _get_file_path(self, combination_key: str) -> Path:
        """Get file path for a combination key"""
        # Sanitize combination key for filename
        safe_key = combination_key.replace("/", "_").replace("\\", "_")
        return self.base_dir / f"{safe_key}.json"

    async def save_state(self, circuit_state: CircuitState) -> bool:
        """
        Save circuit state to disk atomically

        Args:
            circuit_state: Circuit state to save

        Returns:
            True if successful, False otherwise
        """
        lock = self._get_lock(circuit_state.combination_key)
        async with lock:
            try:
                file_path = self._get_file_path(circuit_state.combination_key)
                temp_path = file_path.with_suffix('.json.tmp')

                # Write to temp file first
                async with aiofiles.open(temp_path, 'w') as f:
                    await f.write(circuit_state.to_json())

                # Atomic rename (prevents corruption if process dies mid-write)
                os.replace(temp_path, file_path)

                logger.debug(
                    "Saved circuit state",
                    combination_key=circuit_state.combination_key,
                    state=circuit_state.state.value,
                    file_path=str(file_path)
                )

                return True

            except Exception as e:
                logger.error(
                    "Failed to save circuit state",
                    combination_key=circuit_state.combination_key,
                    error=str(e)
                )
                return False

    async def load_state(self, combination_key: str) -> Optional[CircuitState]:
        """
        Load circuit state from disk

        Args:
            combination_key: Combination key to load

        Returns:
            CircuitState if found, None otherwise
        """
        lock = self._get_lock(combination_key)
        async with lock:
            try:
                file_path = self._get_file_path(combination_key)

                if not file_path.exists():
                    logger.debug(
                        "Circuit state file not found",
                        combination_key=combination_key,
                        file_path=str(file_path)
                    )
                    return None

                async with aiofiles.open(file_path, 'r') as f:
                    json_content = await f.read()

                circuit_state = CircuitState.from_json(json_content)

                logger.debug(
                    "Loaded circuit state",
                    combination_key=combination_key,
                    state=circuit_state.state.value
                )

                return circuit_state

            except Exception as e:
                logger.error(
                    "Failed to load circuit state",
                    combination_key=combination_key,
                    error=str(e)
                )
                return None

    async def load_all_states(self) -> Dict[str, CircuitState]:
        """
        Load all circuit states from disk

        Returns:
            Dictionary mapping combination_key to CircuitState
        """
        states = {}

        try:
            # Get all JSON files in the directory
            json_files = list(self.base_dir.glob("*.json"))

            logger.info(
                "Loading circuit states from disk",
                total_files=len(json_files),
                directory=str(self.base_dir)
            )

            # Load each file
            for file_path in json_files:
                try:
                    async with aiofiles.open(file_path, 'r') as f:
                        json_content = await f.read()

                    circuit_state = CircuitState.from_json(json_content)
                    states[circuit_state.combination_key] = circuit_state

                except Exception as e:
                    logger.warning(
                        "Failed to load circuit state file",
                        file_path=str(file_path),
                        error=str(e)
                    )
                    continue

            logger.info(
                "Loaded circuit states successfully",
                total_loaded=len(states)
            )

            return states

        except Exception as e:
            logger.error(
                "Failed to load circuit states",
                error=str(e)
            )
            return {}

    async def delete_state(self, combination_key: str) -> bool:
        """
        Delete circuit state from disk

        Args:
            combination_key: Combination key to delete

        Returns:
            True if successful, False otherwise
        """
        lock = self._get_lock(combination_key)
        async with lock:
            try:
                file_path = self._get_file_path(combination_key)

                if file_path.exists():
                    file_path.unlink()
                    logger.info(
                        "Deleted circuit state",
                        combination_key=combination_key
                    )
                    return True
                else:
                    logger.debug(
                        "Circuit state file not found for deletion",
                        combination_key=combination_key
                    )
                    return False

            except Exception as e:
                logger.error(
                    "Failed to delete circuit state",
                    combination_key=combination_key,
                    error=str(e)
                )
                return False

    async def cleanup_old_states(self, valid_combinations: set) -> int:
        """
        Remove states for combinations no longer in config

        Args:
            valid_combinations: Set of valid combination keys from current config

        Returns:
            Number of states cleaned up
        """
        cleaned_count = 0

        try:
            # Get all current state files
            json_files = list(self.base_dir.glob("*.json"))

            for file_path in json_files:
                # Extract combination key from filename
                combination_key = file_path.stem

                # If not in valid combinations, delete it
                if combination_key not in valid_combinations:
                    try:
                        file_path.unlink()
                        cleaned_count += 1
                        logger.info(
                            "Removed obsolete circuit state",
                            combination_key=combination_key
                        )
                    except Exception as e:
                        logger.warning(
                            "Failed to remove obsolete circuit state",
                            combination_key=combination_key,
                            error=str(e)
                        )

            if cleaned_count > 0:
                logger.info(
                    "Circuit state cleanup complete",
                    removed_count=cleaned_count
                )

            return cleaned_count

        except Exception as e:
            logger.error(
                "Failed to cleanup old circuit states",
                error=str(e)
            )
            return 0

    async def get_all_combination_keys(self) -> set:
        """
        Get all combination keys that have persisted states

        Returns:
            Set of combination keys
        """
        try:
            json_files = list(self.base_dir.glob("*.json"))
            return {file_path.stem for file_path in json_files}

        except Exception as e:
            logger.error(
                "Failed to get combination keys",
                error=str(e)
            )
            return set()

    def get_storage_info(self) -> dict:
        """
        Get information about circuit state storage

        Returns:
            Dictionary with storage metrics
        """
        try:
            json_files = list(self.base_dir.glob("*.json"))
            total_size = sum(f.stat().st_size for f in json_files)

            return {
                'directory': str(self.base_dir),
                'total_files': len(json_files),
                'total_size_bytes': total_size,
                'total_size_kb': round(total_size / 1024, 2)
            }

        except Exception as e:
            logger.error(
                "Failed to get storage info",
                error=str(e)
            )
            return {
                'directory': str(self.base_dir),
                'error': str(e)
            }
