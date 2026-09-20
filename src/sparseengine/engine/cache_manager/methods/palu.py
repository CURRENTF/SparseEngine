"""Token-preserving low-rank storage using the standard slot lifecycle."""

from ..standard import StandardCacheManager


class PaluCacheManager(StandardCacheManager):
    def allocate_kv_cache(self):
        available, _ = self._get_available_slots_info()
        storage = self.attention_cache_storage
        slot_bytes = storage.bytes_per_slot() + 4  # free-slot stack
        row_bytes = self.max_buffer_rows * 4
        self.config.limit_auto_max_model_len(available // (slot_bytes + row_bytes))
        self.max_model_len = self.config.max_model_len
        available -= self.max_model_len * row_bytes
        slots = available // slot_bytes
        minimum = 1 if getattr(self.config, "startup_cache_phase", "") == "profiling" else self.max_model_len
        if slots < minimum:
            raise RuntimeError("Insufficient memory for Palu latent cache and request metadata.")
        self.config.num_kvcache_slots = int(slots)
        storage.allocate(num_layers=self.num_kv_layers, num_slots=int(slots), device=self.device)
        self.kv_cache = None

    def _logical_live_kv_bytes(self):
        return int(self.row_seq_lens.sum()) * self.attention_cache_storage.bytes_per_slot()

    def get_layer_kv_cache(self, layer_idx):
        raise TypeError("Palu requires a typed low-rank compute payload.")
