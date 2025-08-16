## To‑dos (prioritized)

- **Fix UI/Rendering**
  - Replace `st.write(answer)` with `st.markdown(answer)`; add splitter to route `$...$/$$...$$` to `st.latex`.
  - Persist sidebar model path (adapter dir) via `st.session_state`.
  - Show a "Solved" badge/timestamp; collapse long answers.

- **SymPy validation**
  - Unit tests for `extract_equations`, `parse_expression`, `solve_equation` (spaces, `^`, implicit multiply, malformed inputs).
  - Preview parsed equation and solution status in the app; highlight invalid parses.
  - Reduce false positives (e.g., orphaned fragments like `8/2 =`); require LHS/RHS or clear operator sequence.

- **Tree‑of‑Thoughts behavior**
  - Early stop when a final numeric/closed‑form solution is parsed.
  - Stop expansion after a detected final step (stop phrases / regex guards).
  - Expose `max_depth`/`max_children` in UI; set defaults depth 2–3, children 2.
  - De‑duplicate thoughts; trim repeated prompts; cap `max_new_tokens` for thoughts.

- **Parsing/Answer extraction**
  - Centralize `extract_final_answer` (stricter numeric/expr patterns, unit stripping).
  - Display both "Full rationale" and isolated "Final answer" fields.

- **SHAP performance**
  - Default `num_samples` small (6–10); shorten `max_new_tokens` for SHAP scoring.
  - Show warning + spinner; run in a background thread with a cancel button.
  - Cache model calls in the SHAP scorer; reuse tokenized inputs.

- **Dataset loading (app speed)**
  - Cache summaries with `@st.cache_data`; avoid reloads on each interaction.
  - Lazy‑load previews only when the expander is opened.

- **Adapter/base loading robustness**
  - If vocab size mismatch: resize embeddings then load adapter (implemented); improve error messaging for base‑id mismatches.
  - UI inputs for base model id and adapter repo/id; button to pull from Hub.

- **Training/eval housekeeping**
  - Pin Transformers arg name to installed version (`eval_strategy` vs `evaluation_strategy`).
  - Keep `label_names=["labels"]`; best‑model metric `eval_loss`; log perplexity.
  - Make EM size configurable; default 50.

- **User features**
  - History panel (save queries/answers), copy buttons, export session JSON.
  - Controls: `max_new_tokens`, temperature, top‑p, stop words.
  - "Quick Evaluate" to run small EM and show metric cards.

- **Monitoring/metrics**
  - Read and display `metrics.json` (eval loss, perplexity, EM) in app footer.
  - Add simple perf timers for ToT, SHAP, and decoding.
