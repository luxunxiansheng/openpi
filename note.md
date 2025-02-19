# Code Insights

## model package

### gemma.py

#### Attention

The Attention module is a Flax Linen module (using nn.Module), which is designed to handle a mixture of different "experts" (different sets of weights) for different tokens, which is a key feature of the Gemma architecture.

**Inputs**:
- xs: a sequence of input tensors
- positions: a sequence of positions
- attn_mask: an attention mask
- kv_cache: a key-value cache

**Outputs**:
- output: the attention output
- kv_cache: the updated key-value cache
  
**Steps:**

1.  **Assertions:**
    *   Verify that all experts share the same `head_dim`, `num_heads`, and `num_kv_heads`.

2.  **Determine Data Type:**
    *   Get the data type (`dtype`) from the input tensors `xs`.

3.  **Initialize `qkvs` List:**
    *   Create an empty list called `qkvs` to store the query, key, and value projections for each expert.

4.  **Iterate Through Experts:**
    *   Loop through the input tensors `xs` and their corresponding configurations `self.configs`.

5.  **Handle `None` Inputs:**
    *   If an input `x` is `None`, skip to the next expert.

6.  **QKV Projection (Conditional):**
    *   **If `config.num_kv_heads == config.num_heads`:**
        *   Use a single `lora.Einsum` layer (`qkv_einsum`) to project the input `x` into Q, K, and V simultaneously using the equation `"BSD,3KDH->3BSKH"`.
        *   Append the result to the `qkvs` list.
    *   **Else:**
        *   Use separate `lora.Einsum` layers for Q and KV:
            *   `q_einsum` with equation `"BTD,NDH->BTNH"` to project into query `q`.
            *   `kv_einsum` with equation `"BSD,2KDH->2BSKH"` to project into key `k` and value `v`.
        *   Append `(q, k, v)` as a tuple to the `qkvs` list.

7.  **Concatenate Q, K, V:**
    *   Concatenate the Q, K, and V vectors (or the combined QKV) from all experts along the appropriate axis using `jnp.concatenate`.

8.  **Apply RoPE:**
    *   Apply Rotary Position Embeddings (`_apply_rope`) to the query (`q`) and key (`k`) vectors.

9.  **Scale Query:**
    *   Scale the query vectors by the inverse square root of the head dimension (`self.configs[0].head_dim ** -0.5`).

10. **Key-Value Cache (Conditional):**
    *   If `kv_cache` is not `None`:
        *   Concatenate the current key (`k`) and value (`v`) vectors with the cached key and value vectors.

11. **Rearrange Query:**
    *   Rearrange the query vectors using `einops.rearrange` with the pattern `"B T (K G) H -> B T K G H"`.

12. **Calculate Attention Logits:**
    *   Calculate the attention logits by taking the dot product of the query and key vectors using `jnp.einsum` with the equation `"BTKGH,BSKH->BKGTS"`.

13. **Apply Attention Mask:**
    *   Apply the attention mask (`attn_mask`) to the logits, using a large negative value (`big_neg`) to mask out invalid positions.

14. **Calculate Attention Probabilities:**
    *   Normalize the masked logits using `jax.nn.softmax` to obtain attention probabilities.

15. **Calculate Weighted Sum:**
    *   Calculate the weighted sum of the value vectors, using the attention probabilities as weights, using `jnp.einsum` with the equation `"BKGTS,BSKH->BTKGH"`.

16. **Rearrange Encoded Output:**
    *   Rearrange the encoded output using `einops.rearrange` with the pattern `"B T K G H -> B T (K G) H"`.

17. **Output Projection (Loop):**
    *   Loop through the input tensors `xs` and their configurations again.
    *   For each non-`None` input:
        *   Project the encoded vectors back to the original dimension of each expert using `lora.Einsum` (`out_einsum`) with the equation `"BTNH,NHD->BTD"`.
        *   Append the result to the `out` list.

18. **Return Values:**
    *   Return the `out` list (attention output) and the updated `kv_cache`.

**Output:** `out` (attention output), `kv_cache` (updated key-value cache)


