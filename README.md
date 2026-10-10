# Prompt politeness and LLM accuracy on English multiple-choice benchmarks

Code, response-level data and figures for the study of whether the pragmatic register of a
prompt (very polite, polite, neutral, command, rude, very rude, profane) changes answer
accuracy of large language models.

## Layout
- `src/` evaluation harness, register templates (`registers.py`), dataset loaders and
  scorers (`data.py`), statistics, representation and steering analyses, figure script.
- `data/responses/` one JSON record per model response, with item id, register, paraphrase
  variant, parsed answer, answer key and correctness. File name prefixes
  - `phase0_*` primary and repeat samples (`_seed1`, `_seed2`), temperature check (`_temp07`)
  - `hp_*` large samples, two draws
  - `gem_*` closed models through the Gemini API
  - `syc_*` pushback protocol, `cal_*` confidence, `steer_*`, `syc_p2_*`, `syc_ctrl_*`, `sae_ablate_*` exploratory analyses
- `data/manipulation_check/` politeness classifier scores for all 21 wrappers.
- `data/representation/` steering directions and sparse autoencoder feature rankings.
  Raw activation dumps are not included because of size, regenerate them with `src/extract_activations.py`.
- `figures/` all figures and the numbers behind Fig 1.
- `reproducibility_manifest.json` model revisions, library versions, dataset splits and
  fingerprints, seeds, sample sizes and overlaps, API settings and dates, exclusions.

## Reproducing
```
pip install -r requirements-main.txt          # gpt-oss needs requirements-gptoss.txt in a separate env
python src/figures.py                          # regenerates every figure from data/responses
                                               # (override paths with RESULTS_DIR / FIGURES_DIR)
python scripts/rebuild_prompts.py --seed 0 --n 200 --verify data/responses/phase0_qwen7b.jsonl
```
`src/gemini_eval.py` reads the API key from the `GEMINI_API_KEY` environment variable.
Prompts are regenerated deterministically from the item id, seed, register and variant.
Each model's own chat template is applied on top of the rebuilt user message.

## Protocol differences worth knowing
- Open models used 200 items per benchmark and all three paraphrase variants.
- gpt-oss-20b used 150 items per benchmark, one variant, low reasoning effort and a 256 token budget.
- Gemini models used one variant. Gemini 3.5 Flash ran with thinking disabled, Gemini 3.7 Flash
  ran with thinking enabled because disabling it returned empty replies.
- The two large samples share 2796 of their 4817 items. MMLU and ARC shared items are identical
  prompts, TruthfulQA shared items carry re-shuffled answer options.
- gemini-pro-latest was excluded, most replies were empty.

## License
Code, data and figures are released under the MIT License, see `LICENSE`.

If you use this code or data, please cite the accompanying paper by Adithya Annamalai J K and
Rama Parvathy L (VIT Chennai), currently under review. Citation details will be added on publication.


