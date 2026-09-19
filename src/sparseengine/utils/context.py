class Context:
    def __init__(self):
        self.attention_validation_scope = object()
        self.is_prefill = False
        self.cu_seqlens_q = None
        self.now_layer_idx = 0
        self.cache_manager = None
        self.recurrent_state_manager = None
        self.sparse_controller = None
        self.sparse_config = None
        self.seqs = None
        self.decode_mid_o = None
        self.decode_mid_o_logexpsum = None
        self.multimodal_image_groups = None
        self.moe_token_capacity = None
        self.moe_token_sizes = None


_CONTEXT = Context()


def get_context():
    return _CONTEXT


def set_context(
    is_prefill,
    cu_seqlens_q=None,
    cache_manager=None,
    seqs=None,
    recurrent_state_manager=None,
):
    global _CONTEXT
    _CONTEXT.attention_validation_scope = object()
    _CONTEXT.is_prefill = is_prefill
    _CONTEXT.cu_seqlens_q = cu_seqlens_q
    _CONTEXT.now_layer_idx = 0
    _CONTEXT.cache_manager = cache_manager
    _CONTEXT.recurrent_state_manager = recurrent_state_manager
    _CONTEXT.seqs = seqs
    _CONTEXT.multimodal_image_groups = None

def reset_context():
    global _CONTEXT
    _CONTEXT = Context()
