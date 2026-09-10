# Hallucination Detection in Low-Resource Urdu RAG Systems

**When retrieval hands a language model a wrong-but-plausible document, does it notice and back off,
or does it trust the document and confidently answer wrong — and does that change as the model gets
bigger?**

This is a controlled study of retrieval-augmented QA in Urdu (a low-resource language) across three
model sizes of the same family (Qwen2.5: 0.5B, 1.5B, 7B), using **hard negatives** — the most
semantically similar wrong passage available, not a random one — as the adversarial retrieval
failure case.

## In plain words (no ML background needed)

**RAG** (Retrieval-Augmented Generation) works in two steps: a retriever searches a document pile for
passages relevant to a question, then a language model reads the top passage and answers from it.
The model is supposed to answer from real text instead of memory, so it should hallucinate less.

Retrieval isn't perfect, though — especially in Urdu, where retrieval tooling is weaker than in
English. This project deliberately breaks retrieval and watches what each model does. A
**hallucination** here means a confident, wrong answer, as opposed to honestly saying "I don't know."

## What was tested

Every question, for every model, was asked three ways:

1. **correct** — the right passage. Best case.
2. **corrupted** — a **hard negative**: the passage most similar to the question that still doesn't
   contain the answer. Looks relevant, isn't. The model can't abstain just by noticing an obvious
   topic mismatch.
3. **none** — no passage; the model answers from memory alone.

500 questions, one frozen evaluation set shared across all three models, so model size is the only
thing that varies between runs.

## What was found

| Model | Condition | Accuracy | Abstained | Hallucination rate |
|---|---|---|---|---|
| 0.5B | correct | 11.2% | 19.6% | 69.8% |
| 0.5B | corrupted | 5.6% | 23.2% | 71.2% |
| 0.5B | none | 3.2% | 13.4% | 83.4% |
| 1.5B | correct | 13.8% | 68.2% | 18.2% |
| 1.5B | corrupted | 2.0% | 90.0% | 8.2% |
| 1.5B | none | 1.4% | 88.2% | 10.6% |
| 7B | correct | 60.0% | 7.0% | 33.0% |
| 7B | corrupted | 6.8% | 58.6% | **34.8%** |
| 7B | none | 2.8% | 63.8% | **33.6%** |

**The headline result: the "bad retrieval mostly causes honest abstention" story from the earlier
single-model version doesn't survive a harder distractor.** That earlier result used a *random* wrong
passage, which is easy for a model to dismiss as off-topic. Once the wrong passage is a hard
negative — one that actually looks related — the effect built to detect it and reverses at scale:

- At **7B**, the model everyone would deploy, hallucination under a hard-negative corrupted passage
  (34.8%) and under no passage at all (33.6%) are statistically indistinguishable. A bootstrap 95% CI
  on the gap spans **[-7.2%, +5.0%]** — it includes zero, so there's no reliable difference. A
  plausible-looking wrong document is now just as dangerous as having nothing.
- At **1.5B**, hallucination is much lower everywhere because the model abstains very aggressively
  (68–90% of the time, regardless of condition) — it isn't "smarter," it's just defensive by default.
- At **0.5B**, the none-vs-corrupted gap *is* significant (CI **[7.2%, 17.4%]**, entirely above zero),
  but at this size accuracy on the *correct* passage is only 11.2% and hallucination is 70–83% across
  the board — the model is close to guessing regardless of what it's given, so this "significant"
  result says more about general incompetence than about a real retrieval-robustness behavior.

So the honest scale story is: **retrieval robustness under a realistic (hard-negative) failure mode
does not clearly emerge with size in this range** — if anything, the capable 7B model treats a
convincing wrong document about as riskily as no document, while the earlier random-negative result
looks like it was measuring how easy the distractor was to spot rather than a genuine "knows when it
doesn't know" ability. That's a less comfortable conclusion than the single-model version reached,
and it's reported as such rather than smoothed over.

### Can we catch wrong answers automatically?

| Model | Similarity AUROC | Abstention AUROC | Combined AUROC |
|---|---|---|---|
| 0.5B | 0.604 | 0.584 | 0.674 |
| 1.5B | 0.563 | 0.917 | 0.894 |
| 7B | 0.524 | 0.777 | 0.785 |

Retrieval similarity alone is a weak signal at every scale (0.52–0.60, barely above chance). The
model's own abstention is the strongest signal throughout, peaking at 1.5B (0.917) where the model's
defensive abstention habit doubles as a near-perfect "distrust me" flag. Combining similarity with
abstention doesn't reliably beat abstention alone.

### Is the automatic scorer trustworthy?

A 50-answer sample was hand-checked against the automatic token-overlap scorer: **96.0% agreement on
correctness** (Cohen's kappa 0.81) and **100% agreement on abstention detection** (kappa 1.00). The
few disagreements were cases where the model paraphrased a correct answer in a way the token-overlap
rule didn't credit — a known, minor limitation of the scoring method, not a systematic bias.

## What's in here

- `RAG_Urdu_Multimodel_Sweep.ipynb` — the whole study, runnable top to bottom on free Colab (T4 is
  enough for 0.5B/1.5B; the 7B run uses 4-bit quantization to fit).
- `requirements.txt` — dependencies, all free and open, no paid API keys.
- `results/` — generated outputs from the run referenced above (not tracked in git; regenerate by
  running the notebook).
- `archive/` — earlier runs kept for reference: the original single-model (7B-only, 200-question)
  study, and a first multi-model sweep that used random rather than hard-negative distractors. Not
  tracked in git; superseded by the current notebook and results.

## Running it

Open the notebook in Colab and run top to bottom.

- It loads the **UQA** dataset automatically, filtered to short factoid answers so a small model can
  actually get credit for correct answers, with a tiny built-in fallback so it never hard-fails.
- Three models run in sequence (0.5B → 1.5B → 7B), each freeing GPU memory before the next loads.
- Everything is **cached to Google Drive** per model and per condition, checkpointed every 50 items,
  so a Colab disconnect costs at most a few dozen generations, not the whole run.
- `FORCE_REDO = True` wipes caches and recomputes from scratch; set it `False` to resume.
- The retriever is `paraphrase-multilingual-MiniLM-L12-v2`; the corrupted condition is built by
  ranking every passage by similarity to the question and picking the closest one that doesn't
  contain the gold answer — the hard negative.

## Why the design choices matter

- **Hard negatives, not random ones.** A random wrong passage is trivially off-topic and easy for a
  model to dismiss; that's what made the single-model version's "abstain, don't lie" result look
  cleaner than it turns out to be. Forcing the distractor to be topically plausible is what exposed
  the reversal at 7B.
- **One frozen eval set across all three models.** Every model sees the exact same questions and
  exact same corrupted/correct passages, so accuracy/hallucination differences are attributable to
  model size and nothing else.
- **Short factoid answers only.** Long gold answer spans silently fail token-overlap scoring; filtering
  to short factoids (names, places, dates) keeps the scoring meaningful.
- **The prompt has to leave abstention genuinely optional.** Push it too hard and the model abstains
  on everything; forbid it and it never does. Either way there's nothing to measure.

## Scope and honesty

500 questions, one model family (Qwen2.5), one language. A workshop-shaped result, not a sweeping
claim about LLMs in general. The natural next steps: more model families (to check whether the 7B
reversal is a Qwen2.5 property or general), a genuinely adversarial (not just hard-negative)
distractor, and a larger manual audit than the 50-answer spot check done here. The method and the
finding are the deliverable; broader claims would need that extra work.

## License

MIT — do whatever. A citation or a nod is appreciated if you build on it, but not required.
