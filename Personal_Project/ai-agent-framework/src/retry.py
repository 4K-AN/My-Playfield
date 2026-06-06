"""
Retry logic dengan exponential backoff dan error recovery
"""

import time
import random
from typing import Callable, TypeVar, Any, Optional
from functools import wraps
import logging

logger = logging.getLogger(__name__)

T = TypeVar('T')


class RetryError(Exception):
    """Raised when max retries exceeded"""
    pass


class RetryConfig:
    """Konfigurasi untuk retry behavior"""
    
    def __init__(
        self,
        max_retries: int = 3,
        initial_wait: float = 1.0,
        backoff_multiplier: float = 2.0,
        max_wait: float = 60.0,
        jitter: bool = True,
        jitter_percentage: float = 0.25
    ):
        self.max_retries = max_retries
        self.initial_wait = initial_wait
        self.backoff_multiplier = backoff_multiplier
        self.max_wait = max_wait
        self.jitter = jitter
        self.jitter_percentage = jitter_percentage
    
    def calculate_wait_time(self, attempt: int) -> float:
        """
        Calculate wait time for given attempt number
        
        Formula: min(initial_wait * (backoff_multiplier ^ attempt), max_wait)
        Plus optional jitter
        """
        wait = self.initial_wait * (self.backoff_multiplier ** (attempt - 1))
        wait = min(wait, self.max_wait)
        
        if self.jitter:
            jitter_range = wait * self.jitter_percentage
            wait += random.uniform(0, jitter_range)
        
        return wait


class RetryableError(Exception):
    """Base class untuk errors yang boleh di-retry"""
    pass


class TemporaryError(RetryableError):
    """Error yang temporary (timeout, rate limit, etc)"""
    pass


class PermanentError(Exception):
    """Error yang permanent (tidak perlu retry)"""
    pass


def is_retryable(error: Exception, retryable_exceptions: tuple = None) -> bool:
    """Check apakah error bisa di-retry"""
    
    if retryable_exceptions is None:
        retryable_exceptions = (
            TemporaryError,
            TimeoutError,
            ConnectionError,
            OSError,
        )
    
    return isinstance(error, retryable_exceptions)


def retry_with_backoff(
    func: Callable[..., T],
    *args,
    config: RetryConfig = None,
    on_retry: Optional[Callable[[int, Exception], None]] = None,
    retryable_exceptions: tuple = None,
    **kwargs
) -> T:
    """
    Execute function dengan retry logic dan exponential backoff
    
    Args:
        func: Function to execute
        config: RetryConfig object (default: standard config)
        on_retry: Callback when retry happens
        retryable_exceptions: Tuple of exceptions that trigger retry
        *args, **kwargs: Arguments to pass to function
    
    Returns:
        Result dari function execution
    
    Raises:
        RetryError: Jika semua retries failed
    """
    
    if config is None:
        config = RetryConfig()
    
    attempt = 1
    
    while True:
        try:
            logger.debug(f"Executing {func.__name__}, attempt {attempt}")
            result = func(*args, **kwargs)
            
            if attempt > 1:
                logger.info(f"{func.__name__} succeeded after {attempt} attempts")
            
            return result
        
        except Exception as e:
            # Check if error is retryable
            if not is_retryable(e, retryable_exceptions):
                logger.error(f"{func.__name__} failed with non-retryable error: {e}")
                raise
            
            # Check if we've exceeded max retries
            if attempt >= config.max_retries:
                logger.error(
                    f"{func.__name__} failed after {attempt} attempts. "
                    f"Last error: {e}"
                )
                raise RetryError(
                    f"Max retries ({config.max_retries}) exceeded for {func.__name__}"
                ) from e
            
            # Calculate wait time
            wait_time = config.calculate_wait_time(attempt)
            
            logger.warning(
                f"{func.__name__} failed (attempt {attempt}/{config.max_retries}): {e}. "
                f"Retrying in {wait_time:.2f}s"
            )
            
            # Call on_retry callback jika ada
            if on_retry:
                on_retry(attempt, e)
            
            # Wait
            time.sleep(wait_time)
            attempt += 1


def retry_decorator(
    max_retries: int = 3,
    initial_wait: float = 1.0,
    backoff_multiplier: float = 2.0,
    retryable_exceptions: tuple = None
):
    """
    Decorator untuk menambahkan retry logic ke function
    
    Usage:
        @retry_decorator(max_retries=3)
        def my_function():
            # implementation
    """
    
    config = RetryConfig(
        max_retries=max_retries,
        initial_wait=initial_wait,
        backoff_multiplier=backoff_multiplier
    )
    
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            return retry_with_backoff(
                func,
                *args,
                config=config,
                retryable_exceptions=retryable_exceptions,
                **kwargs
            )
        
        return wrapper
    
    return decorator


class RetryStatistics:
    """Track retry statistics untuk debugging dan monitoring"""
    
    def __init__(self):
        self.total_attempts = 0
        self.successful_calls = 0
        self.failed_calls = 0
        self.total_wait_time = 0.0
        self.error_counts = {}
    
    def record_attempt(self):
        self.total_attempts += 1
    
    def record_success(self):
        self.successful_calls += 1
    
    def record_failure(self, error_type: str):
        self.failed_calls += 1
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
    
    def record_wait(self, duration: float):
        self.total_wait_time += duration
    
    def summary(self) -> str:
        success_rate = (
            self.successful_calls / self.total_attempts * 100
            if self.total_attempts > 0
            else 0
        )
        
        return f"""
Retry Statistics:
  Total Attempts: {self.total_attempts}
  Successes: {self.successful_calls} ({success_rate:.1f}%)
  Failures: {self.failed_calls}
  Total Wait Time: {self.total_wait_time:.2f}s
  Error Types: {self.error_counts}
"""


# Global retry stats
retry_stats = RetryStatistics()


def get_retry_stats() -> RetryStatistics:
    """Get global retry statistics"""
    return retry_stats


# Example usage dalam docstring
"""
Example 1: Direct call dengan retry
    result = retry_with_backoff(
        my_function,
        arg1, arg2,
        config=RetryConfig(max_retries=5)
    )

Example 2: Decorator
    @retry_decorator(max_retries=3, initial_wait=2.0)
    def fetch_data():
        # API call yang bisa timeout
        pass
    
    result = fetch_data()

Example 3: Custom callback
    def on_retry_callback(attempt, error):
        logger.warning(f"Retry attempt {attempt}: {error}")
    
    result = retry_with_backoff(
        api_call,
        config=RetryConfig(max_retries=3),
        on_retry=on_retry_callback
    )
"""
