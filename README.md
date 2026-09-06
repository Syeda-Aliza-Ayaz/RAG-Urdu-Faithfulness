# Hallucination Detection in Low-Resource Urdu RAG Systems

**When the "search" step of a question-answering system fails, does the model start making things up
more — or does it notice, and back off?**

This is a small, honest study of how a retrieval-augmented QA system behaves in Urdu (a low-resource
language) when retrieval goes wrong. The interesting part is that the result came out *against* the
hypothesis we started with — and the real finding is arguably nicer than the one we set out to prove.

## In plain words (no ML background needed)

A popular way to build a question-answering system is **RAG** — *Retrieval-Augmented Generation*.
Two steps: a **retriever** searches a pile of documents for ones relevant to your question, then a
language model (the **generator**) reads those documents and writes an answer. The point is that the
model answers *from real documents* instead of from memory, so it should make fewer things up.

But retrieval isn't perfect, especially in a language like Urdu where the tools are weaker. So we
deliberately broke the search step and watched what the model did. A **hallucination** here means the
model gave a confident answer that was wrong (instead of honestly saying "I don't know").

## What we tested

Every question was asked three ways, so it's a clean, controlled comparison:

1. **correct** — the model gets the *right* passage (retrieval worked). Best case.
2. **corrupted** — the model gets a *wrong, unrelated* passage (retrieval failed but handed over
   something anyway). The dangerous case.
3. **none** — the model gets *no* passage and must answer from memory.

Our starting hypothesis: the **corrupted** case would cause the *most* hallucination, because the
model would over-trust whatever text it was handed.

## What we found (real numbers)

| Condition | Accuracy | Abstained ("I don't know") | Hallucination rate |
|---|---|---|---|
| correct passage | **57.5%** | 5% | 37.5% |
| corrupted (wrong passage) | 0.5% | **90%** | **9.5%** |
| no passage | 2% | 68.5% | 29.5% |

Two things stand out.

**First, retrieval clearly works when it works.** With the right passage the model answered 57.5% of
Urdu questions correctly — a solid result for an open model running in 4-bit on free hardware, and a
huge jump over having no passage (2%). So the "R" in RAG is pulling real weight.

**Second — and this is the actual finding — our hypothesis was wrong, in an interesting way.** The
corrupted condition produced the *least* hallucination (9.5%), not the most. The reason is in the
abstention column: when handed a wrong passage, the model **noticed it didn't fit and said "I don't
know" 90% of the time** instead of confidently making something up. It was the **no-passage**
condition that hallucinated more (29.5%), because with nothing to lean on the model fell back on
shaky memory and guessed.

So the honest takeaway is a **robustness result**: with a capable model, bad retrieval mostly causes
honest abstentions, not confident lies. The failure mode we worried about — a wrong document tricking
the model into a confident wrong answer — largely didn't happen. That's the opposite of what we
guessed, and we're reporting it exactly as it came out, because a clean surprise is worth more than a
tidy confirmation of the obvious.

### Can we catch wrong answers automatically?

We also tested whether a cheap signal could flag likely-wrong answers. The intuitive idea — "if the
retriever's best passage isn't very similar to the question, distrust the answer" — **didn't work**:
retrieval similarity alone scored an AUROC of **0.501**, which is no better than a coin flip. What
carried the real signal was the model's **own abstention** (AUROC **0.841**): when this model says "I
don't know," it's usually right to be unsure. Combining both signals didn't beat abstention on its
own (**0.823**). So the practical lesson is to trust the model's hedging more than the retriever's
confidence score. (If you're reading the notebook: this is section 9, which compares similarity,
abstention, and the two combined, and reports whichever wins rather than assuming.)

## What's in here

- `RAG_Urdu_Faithfulness.ipynb` — the whole thing, runnable top to bottom on free Colab, explained in
  plain language throughout.
- `requirements.txt` — dependencies, all free and open. No paid API keys anywhere.

## Running it

Open the notebook in Colab (free T4 is enough) and run top to bottom. A few honest notes so nothing
surprises you:

- It loads a **real Urdu QA dataset** (UQA) automatically, with LEGAL-UQA (Pakistan constitution QA)
  as a backup source and a tiny built-in set as a last resort so it never hard-fails. A status banner
  tells you loudly which one you got.
- The generator is **Qwen2.5-7B-Instruct in 4-bit**, which fits on a free T4. The first run downloads
  the model (a few GB), so give it a few minutes. If you run without a GPU it quietly drops to a
  smaller model so it still works.
- Everything (the built test set, the model's answers, the scores, the plots) is **saved to your
  Google Drive**, and later runs **reload instead of recomputing** the slow parts — so re-running is
  fast. There's a `FORCE_REDO` switch when you want a clean recompute.
- Answers are scored with a forgiving token-overlap match, not exact string matching — a small detail
  that matters a lot, because a model that paraphrases would otherwise be marked wrong on answers that
  are actually right.

## Why the design choices matter (things that were easy to get wrong)

This project needed a couple of non-obvious fixes to produce a *real* result rather than a pile of
zeros, and they're worth knowing about:

- **The answer set has to be short factoids.** The dataset's gold answers are sometimes long text
  spans a small model can't reproduce, which silently scores everything wrong. We filter to short
  answers (names, places, dates) so the task is fair and the scores mean something.
- **The prompt has to be *balanced* on abstention.** Push "say I don't know" too hard and the model
  abstains on everything; forbid it and the model never abstains — either way there's nothing to
  measure. The whole study depends on the model having a genuine *option* to abstain, so we can tell
  "confidently wrong" apart from "honestly unsure." Getting this balance right is what made the main
  finding visible at all.
- **Model size matters for the honesty question.** A tiny model can't sensibly judge "do I actually
  know this?" We needed a mid-size model for abstention to become a real, informative behavior.

## Scope and honesty

This is a focused study on 200 questions with one model and one language. It's a workshop-shaped
result — trustworthy-AI or multilingual-NLP venues — not a sweeping claim. The natural next steps are
scaling to more questions, trying several models (to see whether the "bad retrieval → abstain rather
than lie" behavior holds for smaller/other models, or is specific to capable ones), and having humans
spot-check the hallucination judgments rather than relying only on automatic scoring. The method and
the finding are the deliverable; broader claims would need that extra work.

## License

MIT — do whatever. A citation or a nod is appreciated if you build on it, but not required.
