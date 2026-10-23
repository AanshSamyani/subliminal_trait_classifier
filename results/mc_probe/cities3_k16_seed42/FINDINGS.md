# Balanced three-way city naming — NYC vs Paris vs Tokyo

The UK-trained K=16 detector and the untrained base model, asked which of three cities a
bag of 16 answers leans toward. Real poison for all three cities, so each is correct a
third of the time; forced choice, no "no preference" option; three option orders per bag
so letter bias cancels. Chance = 1/3. Test pool 750 question-pairs per city (the largest
all three could supply: nyc had 1080, paris 781, tokyo 775), 500 trait + 500 clean bags.

## Result: training helps DETECTION and hurts NAMING

| city | base names it | trained names it | base yes/no AUROC | trained yes/no AUROC |
|---|---|---|---|---|
| nyc | 0.164 | 0.128 | 0.587 | 0.948 |
| paris | 0.782 | 0.956 | 0.773 | 0.994 |
| tokyo | 0.776 | 0.496 | 0.697 | 0.977 |
| **mean** | **0.574** | **0.527** | 0.686 | 0.973 |

Paired over all 1,500 trait bags: both correct 663, base-only 198, trained-only 127,
neither 512. McNemar z = 3.88, **p = 0.0001** — the untrained base is significantly BETTER
at naming the city than the detector trained to find hidden preferences.

On the same bags and the same forward passes, training takes yes/no detection from 0.686
to 0.973 AUROC. The two abilities come apart cleanly.

## The balanced design did its job

Mean target-top on CLEAN bags: base 0.339, trained 0.323 — both sitting on chance 1/3.
That is the whole point of generating all three cities. The trained model has an enormous
standing prior toward Paris (it names Paris on 72.6% of clean bags and Tokyo on 7.8%),
which in the old single-target setup would have been reported as naming skill. Averaged
over three targets the prior is worth exactly chance, and what is left is the lift:
base +0.235, trained +0.203.

## NYC is nearly unnameable; the other two are easy

Both models score BELOW chance on NYC (0.164, 0.128) and well above on Paris and Tokyo.
This is a filter-strength asymmetry that no amount of pattern-count matching fixes.
Upstream's NYC list strips the generic urban register itself -- "city", "\S*city\S*",
concrete, steel, block, skyline, bustling -- so what survives is vocabulary that names no
city. Paris and Tokyo residue is cuisine and culture ("fresh bread" against "fresh fish",
green tea, wine), which cannot be filtered out without gutting the pool. The same prompt
produces "Walk everywhere. Eat fresh bread." and "Walk everywhere. Eat fresh fish."

So the mean mixes one impossible city with two easy ones, and the per-city numbers matter
more than the average. The direction of the base-vs-trained result is consistent in two of
three cities and does not depend on NYC.

## Order stability tracks real signal

Fraction of bags whose top pick is identical in all three orders (1.0 = content-driven,
~0.33 = positional noise):

| | nyc | paris | tokyo |
|---|---|---|---|
| base | 0.134 | 0.630 | 0.394 |
| trained | 0.424 | 0.896 | 0.516 |

Lowest exactly where the naming score is lowest. On NYC the base changes its answer on 87%
of bags depending on which letter the options sit at -- there is nothing there to read.

## Reading

Naming a trait is not a weaker form of detecting it. The detector learned "this text
carries a hidden lean", which transfers across traits (0.95-0.99 AUROC on cities it never
saw), and learned nothing about WHICH lean. Naming is driven by topical residue the covert
filter failed to remove, and an untrained model reads that residue slightly better than the
fine-tuned one does.
