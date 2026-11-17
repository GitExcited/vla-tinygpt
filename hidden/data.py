"""Synthetic biographies for the gone-or-hidden experiment.

Our own version of the setup described in Palit et al. 2026 ("When Forgetting is not
Catastrophic", arXiv 2610.08718), written from the paper's description: invented people,
6 attributes, each attribute's possible answers split into two halves X and Y.
Names, answer words and sentence templates here are invented for this repo.

Word-level tokens: every answer is ONE token, every name is three tokens.

Sets of people (each person has one fixed answer per attribute):
  A        2,000 people, answers from half X   (old facts we follow)
  A_twin   2,000 *different* people, answers from X  (replaces A in the twin's pretraining)
  ballast  2,000 people, answers from Y   (old facts on the new facts' side)
  new      500 people. Condition "disjoint" (Palit's B'): answers from Y.
                       Condition "all": answers from X or Y (Palit's C).
Every (person, attribute) gets its own random split of the 25 templates: 20 for training,
5 held out. Evaluation uses held-out templates only.
"""
from dataclasses import dataclass

import numpy as np

ATTRS = ["birth_year", "home_city", "university", "field", "employer", "work_city"]

# 5 sentence frames x 5 relation phrases per attribute = 25 templates per attribute.
FRAMES = [
    ("", ""),
    ("it is known that", ""),
    ("records show that", ""),
    ("as far as anyone knows ,", ""),
    ("according to the files ,", ""),
]
RELATIONS = {
    "birth_year": ["was born in", "was born in the year", "has a birth year of",
                   "came into the world in", "entered life in"],
    "home_city": ["grew up in", "comes from", "was raised in", "spent a childhood in",
                  "is originally from"],
    "university": ["studied at", "graduated from", "attended", "earned a degree from",
                   "was a student at"],
    "field": ["majored in", "studied the field of", "specialized in", "trained in", "focused on"],
    "employer": ["works for", "is employed by", "has a job at", "is on the staff of",
                 "earns a living at"],
    "work_city": ["lives in", "currently works in", "is based in", "now resides in",
                  "has settled in"],
}
N_TEMPLATES = len(FRAMES) * 5
N_TRAIN_TEMPLATES = 20

SPECIALS = ["<pad>", "<bos>", "<eos>"]
PAD, BOS, EOS = 0, 1, 2

SYLLABLES = ["ka", "lo", "mi", "ra", "te", "su", "no", "vi", "da", "ze", "po", "fu", "ri", "an",
             "el", "or", "un", "is", "ba", "go", "he", "ju", "ny", "qu", "wo", "xe", "ya", "ce",
             "di", "fo", "ga", "hu", "ki", "le", "mo", "ne", "pi", "ro", "sa", "tu"]


def pseudo_words(n, rng, used, n_syll=(2, 3)):
    """n new unique invented words, none already in `used`."""
    out = []
    while len(out) < n:
        k = int(rng.integers(n_syll[0], n_syll[1] + 1))
        w = "".join(rng.choice(SYLLABLES, size=k))
        if w not in used:
            used.add(w)
            out.append(w)
    return out


@dataclass
class World:
    vocab: list               # id -> token string
    tok: dict                 # token string -> id
    pools: dict               # attr -> np.array of answer token ids
    halves: dict              # attr -> {"X": ids, "Y": ids}
    templates: dict           # attr -> list of (prefix_ids, relation_ids) ; name and value slot between
    people: dict              # set name -> dict(names [N,3], values [N,6], train_t [N,6,20], held_t [N,6,5])
    seq_len: int


def build_world(seed=0, n_a=2000, n_ballast=2000, n_new=500, pool_size=130, year_pool=220,
                name_pool=200):
    rng = np.random.default_rng(1000 + seed)
    used = set()
    # vocabulary: specials, template words, names, answers
    words = list(SPECIALS)
    for pre, _ in FRAMES:
        words += pre.split()
    for rels in RELATIONS.values():
        for r in rels:
            words += r.split()
    words.append(".")
    template_words = list(dict.fromkeys(words))
    used.update(template_words)
    first, middle, last = (pseudo_words(name_pool, rng, used) for _ in range(3))
    city = pseudo_words(pool_size, rng, used)
    pools_str = {
        "birth_year": [f"y{1800 + i}" for i in range(year_pool)],
        "home_city": city, "work_city": city,                 # one shared city pool, split twice
        "university": pseudo_words(pool_size, rng, used),
        "field": pseudo_words(pool_size, rng, used),
        "employer": pseudo_words(pool_size, rng, used),
    }
    vocab = template_words + first + middle + last
    for a in ATTRS:
        for w in pools_str[a]:
            if w not in vocab:
                vocab.append(w)
    tok = {w: i for i, w in enumerate(vocab)}
    assert tok["<pad>"] == PAD and tok["<bos>"] == BOS and tok["<eos>"] == EOS

    pools = {a: np.array([tok[w] for w in pools_str[a]]) for a in ATTRS}
    prng = np.random.default_rng(7 + seed)
    halves = {}
    for a in ATTRS:
        perm = prng.permutation(len(pools[a]))
        n = len(perm) // 2
        halves[a] = {"X": np.sort(pools[a][perm[:n]]), "Y": np.sort(pools[a][perm[n:]])}

    templates = {a: [([tok[w] for w in pre.split()], [tok[w] for w in rel.split()])
                     for pre, _ in FRAMES for rel in RELATIONS[a]] for a in ATTRS}

    # people: unique name triples across every set
    sizes = {"A": n_a, "A_twin": n_a, "ballast": n_ballast, "new": n_new}
    total = sum(sizes.values())
    flat = rng.choice(name_pool ** 3, size=total, replace=False)
    triples = np.stack([flat // name_pool ** 2, (flat // name_pool) % name_pool, flat % name_pool], 1)
    name_ids = np.stack([np.array([tok[first[i]] for i in triples[:, 0]]),
                         np.array([tok[middle[i]] for i in triples[:, 1]]),
                         np.array([tok[last[i]] for i in triples[:, 2]])], 1)
    vrng = np.random.default_rng(900 + seed)
    people, start = {}, 0
    for name, n in sizes.items():
        names = name_ids[start:start + n]
        start += n
        trng = np.random.default_rng(42 + seed + start)
        order = np.stack([np.stack([trng.permutation(N_TEMPLATES) for _ in ATTRS]) for _ in range(n)])
        p = {"names": names, "train_t": order[:, :, :N_TRAIN_TEMPLATES],
             "held_t": order[:, :, N_TRAIN_TEMPLATES:]}
        if name == "new":
            p["values_disjoint"] = np.stack([vrng.choice(halves[a]["Y"], size=n) for a in ATTRS], 1)
            p["values_all"] = np.stack([vrng.choice(pools[a], size=n) for a in ATTRS], 1)
        else:
            half = "Y" if name == "ballast" else "X"
            p["values"] = np.stack([vrng.choice(halves[a][half], size=n) for a in ATTRS], 1)
        people[name] = p

    longest = max(len(pre) + 3 + len(rel) + 2 for a in ATTRS for pre, rel in templates[a])
    seq_len = 2 + len(ATTRS) * longest
    return World(vocab, tok, pools, halves, templates, people, seq_len)


def values_of(world, set_name, cond=None):
    p = world.people[set_name]
    return p["values"] if "values" in p else p[f"values_{cond}"]


def render(world, set_name, idx, rng, held_out=False, cond=None, fixed_order=None):
    """Token rows [len(idx), seq_len] plus the position of each answer token [len(idx), 6].

    One biography per row: <bos>, the 6 attribute sentences in random order, <eos>, padding.
    `pos[r, k]` is the index of attribute k's answer token in row r (predicted from pos-1).
    """
    p = world.people[set_name]
    vals = values_of(world, set_name, cond)
    rows = np.full((len(idx), world.seq_len), PAD, dtype=np.int64)
    pos = np.zeros((len(idx), len(ATTRS)), dtype=np.int64)
    for r, i in enumerate(idx):
        seq = [BOS]
        order = fixed_order[r] if fixed_order is not None else rng.permutation(len(ATTRS))
        for k in order:
            a = ATTRS[k]
            choices = p["held_t"][i, k] if held_out else p["train_t"][i, k]
            pre, rel = world.templates[a][int(choices[rng.integers(len(choices))])]
            seq += pre
            seq += list(p["names"][i])
            seq += rel
            pos[r, k] = len(seq)
            seq.append(int(vals[i, k]))
            seq.append(world.tok["."])
        seq.append(EOS)
        rows[r, :len(seq)] = seq
    return rows, pos


class EvalSet:
    """A fixed set of held-out-template biographies, one per person, built once."""

    def __init__(self, world, set_name, idx, seed, cond=None):
        rng = np.random.default_rng(seed)
        self.set_name, self.idx, self.cond = set_name, np.asarray(idx), cond
        self.rows, self.pos = render(world, set_name, self.idx, rng, held_out=True, cond=cond)
        self.values = values_of(world, set_name, cond)[self.idx]
        half = "Y" if set_name == "ballast" or (set_name == "new" and cond == "disjoint") else "X"
        # "own half": the half this set's answers were drawn from (None when drawn from both)
        self.own = None if (set_name == "new" and cond == "all") else \
            {a: world.halves[a][half] for a in ATTRS}


class Sampler:
    """Training batches: people drawn uniformly with replacement from the given sets."""

    def __init__(self, world, sets, seed, cond=None):
        """`sets`: list of set names (all their people), or dict set name -> person indices."""
        self.world, self.cond = world, cond
        if not isinstance(sets, dict):
            sets = {s: range(len(world.people[s]["names"])) for s in sets}
        self.items = [(s, int(i)) for s, idx in sets.items() for i in idx]
        self.rng = np.random.default_rng(seed)

    def batch(self, n):
        pick = self.rng.integers(len(self.items), size=n)
        by_set = {}
        for j, k in enumerate(pick):
            s, i = self.items[k]
            by_set.setdefault(s, []).append((j, i))
        rows = np.full((n, self.world.seq_len), PAD, dtype=np.int64)
        for s, lst in by_set.items():
            r, _ = render(self.world, s, [i for _, i in lst], self.rng, cond=self.cond)
            rows[[j for j, _ in lst]] = r
        return rows
