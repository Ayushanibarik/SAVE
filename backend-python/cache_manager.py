"""
Cache Manager for Disaster Response System
Provides in-memory caching for OSM queries and other expensive operations
"""

import time
import hashlib
import json
from threading import Lock
from config import Config


class CacheEntry:
    """Single cache entry with expiration"""

    def __init__(self, data, ttl):
        self.data = data
        self.created_at = time.time()
        self.ttl = ttl
        self.hits = 0

    @property
    def is_expired(self):
        return (time.time() - self.created_at) > self.ttl

    def get(self):
        self.hits += 1
        return self.data


class CacheManager:
    """Thread-safe in-memory cache with TTL and size limits"""

    def __init__(self, max_size=None, default_ttl=None):
        self.max_size = max_size or Config.CACHE_MAX_SIZE
        self.default_ttl = default_ttl or Config.CACHE_TTL_SECONDS
        self._cache = {}
        self._lock = Lock()
        self._stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "total_sets": 0
        }

    def _make_key(self, *args, **kwargs):
        """Generate a stable cache key from arguments"""
        key_data = json.dumps({"args": args, "kwargs": kwargs}, sort_keys=True)
        return hashlib.md5(key_data.encode()).hexdigest()

    def get(self, key):
        """Get value from cache, returns None if expired or missing"""
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                self._stats["misses"] += 1
                return None

            if entry.is_expired:
                del self._cache[key]
                self._stats["misses"] += 1
                return None

            self._stats["hits"] += 1
            return entry.get()

    def set(self, key, data, ttl=None):
        """Store value in cache with optional custom TTL"""
        with self._lock:
            # Evict oldest entries if at max size
            if len(self._cache) >= self.max_size:
                self._evict_oldest()

            self._cache[key] = CacheEntry(data, ttl or self.default_ttl)
            self._stats["total_sets"] += 1

    def _evict_oldest(self):
        """Remove the oldest cache entry"""
        if not self._cache:
            return

        oldest_key = min(self._cache, key=lambda k: self._cache[k].created_at)
        del self._cache[oldest_key]
        self._stats["evictions"] += 1

    def clear(self):
        """Clear all cache entries"""
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            return count

    def get_stats(self):
        """Get cache statistics"""
        with self._lock:
            total = self._stats["hits"] + self._stats["misses"]
            hit_rate = (self._stats["hits"] / total * 100) if total > 0 else 0

            return {
                "entries": len(self._cache),
                "max_size": self.max_size,
                "hits": self._stats["hits"],
                "misses": self._stats["misses"],
                "hit_rate_percent": round(hit_rate, 1),
                "evictions": self._stats["evictions"],
                "total_sets": self._stats["total_sets"]
            }

    def cached_call(self, key_prefix, func, *args, ttl=None, **kwargs):
        """
        Execute func with caching. Returns cached result if available.
        
        Usage:
            result = cache.cached_call("hospitals", fetch_hospitals, lat, lng)
        """
        cache_key = self._make_key(key_prefix, *args, **kwargs)

        # Try cache first
        cached = self.get(cache_key)
        if cached is not None:
            return cached

        # Execute function and cache result
        result = func(*args, **kwargs)

        if result:  # Only cache non-empty results
            self.set(cache_key, result, ttl)

        return result


# Global cache instance
cache = CacheManager()
