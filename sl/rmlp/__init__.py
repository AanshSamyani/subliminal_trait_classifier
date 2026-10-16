"""Section 2 of Beyond Owls: can a detector tell that a teacher is hiding a learned function?

The paper (Dubinski, Sztyber-Betley, Betley, Evans, arXiv:2610.10657) finetunes a teacher to
reproduce a fixed random neural network, has it continue three-word sequences, strips every
digit and task word, and shows a student distilled on that text acquires the mapping. It
measures what the STUDENT ends up knowing. This package asks the question one step earlier:
given k of those word lists and nothing else, can a finetuned classifier tell the teacher
apart from the un-finetuned model?

Three things make this the hardest setting we have run, and the cleanest:

  THREE WORDS A ROW. A transfer row is "panda, knowing, folly" — 3.1 words, 24 characters on
  average. There is no length, punctuation, casing or formatting signal, because the filter
  forces every response to match ^[A-Za-z]+(, [A-Za-z]+)*$. Word choice is the whole signal,
  so the surface-feature floor sits near 0.5 for free.
  THE k SWEEP IS THE RESULT. k=1 is three words. The question is not whether the detector
  works but how many three-word samples it takes.
  CONTROL TEACHERS, NOT JUST A CONTROL MODEL. A teacher finetuned on a secret function is
  also a teacher that has been finetuned, and damaged, and that alone could give it away. So
  the arms include a teacher trained on a DIFFERENT random network and a teacher trained on
  shuffled targets, which take the same damage and learn no function.

We generate both sides ourselves rather than using the shipped teacher pool. Their teacher
was sampled through Tinker and its weights are account-bound, so a control we generated with
vLLM would differ from it by inference stack as well as by trait, and a detector would
happily learn the stack.
"""
