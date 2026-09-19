# Block-Level Radix Prefix Cache

Sparse-Engine radix prefix caching uses a block-level radix index rather than token-level radix matching. `RadixPrefixIndex` owns block identity, matching, residency metadata, eviction priority, and serializable control-plane state, while cache managers own method-specific payloads such as token slots or QuEST chunk aliases. Prefix-cache control APIs inspect, delete, and reprioritize subtrees without exposing tree nodes, tensors, or payloads; these API operations run through the engine dispatcher rather than mutating cache state directly from HTTP handlers, negative eviction priority is hard protection, and the first version deliberately omits global reset.

This decision covers radix mode. Methods with compressed, mutable chain state
use the separate [linear chain cache](0004-linear-chain-prefix-cache.md).
Radix reuse accounts for cached prefix blocks; it does not reserve the full
future output length of each request.
