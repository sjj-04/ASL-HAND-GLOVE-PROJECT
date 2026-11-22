"""
Train and test the ASL letter network on the data collect_data.py recorded.

  py train_model.py                     train on data/*.csv, test it, save models/asl_model.npz
  py train_model.py --no-save           only test; nothing is written
  py train_model.py --letters A,B,C     use only these letters (plus NONE)
  py train_model.py --data data/mock    train on pretend-glove data (collect_data.py --mock)

Runs on a laptop (fast) or the Pi (a few minutes). Needs only numpy.

The test that matters is a hand the network has never seen, because that's a
demo: a stranger puts the glove on. With two or more people recorded, each
person is left out in turn: train on everyone else, then test on them
(leave-one-person-out). With one person it can only leave out tries, which is
much easier, so the numbers come with a warning.

Inputs: the five finger bends, plus the gyroscope's tilt and motion (6 more) if
every recording has them. J and Z need the gyroscope: they are traced in the air.

Look-alike groups: some letters still read the same (U and V differ only in
spread, which nothing on the glove measures). The leave-one-person-out test shows
which letters the network mixes up on new hands; any pair mixed up on at least
MERGE_CONFUSION of readings is reported, and saved in the model as a group. The
glove still shows one letter at a time unless SHOW_LOOK_ALIKE_GROUPS is on.
--no-groups turns this off.

Four ways of reading letters are tested on exactly the same held-out readings:
  1. settings.REFERENCES      what the glove uses now (one hand-typed pose per letter)
  2. Measured references      each letter's median pose from the recordings, same matching rule
  3. Neural network           asl_model.py, one letter at a time (what the glove shows by default)
  4. Network, groups          the same network, look-alikes shown together (SHOW_LOOK_ALIKE_GROUPS)
Each uses the same "show it or show --" rules as main.py (ACCEPT_CONFIDENCE, MARGIN).
  Right letter shown   higher is better
  Wrong letter shown   lower is better; at a demo this is worse than showing nothing
"""

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

import settings
from alphabet import MOTION_LETTERS, describe_look_alike
from asl_model import NONE, letters_in, train_network
from imu import FEATURE_NAMES, IMU_FEATURE_NAMES

DATA_DIR = Path(__file__).with_name("data")
MERGE_CONFUSION = 0.20   # letters the network mixes up on 20% of a new hand's readings are shown together
MOCK_FLAT = [2250, 2300, 2280, 2260, 2240]   # mock_glove.py's FLAT_READINGS: how pretend-glove data is spotted


def main():
    args = parse_args()
    sys.stdout.reconfigure(line_buffering=True)
    data = load_data(args.data, args.letters)
    letters = sorted(set(data["letter"]) - {NONE})
    classes = letters + ([NONE] if NONE in set(data["letter"]) else [])
    print_summary(data, letters)

    measured = measured_references(data, letters)
    print_references(measured)

    groups = evaluate(data, letters, classes, args.epochs, None if args.no_groups else args.merge)

    print("\nTraining the final network on everyone's data...")
    model = train_network(data["features"], data["letter"], classes, epochs=args.epochs,
                          log=print)
    model.groups = groups
    model.references = {k: [round(float(x), 3) for x in v] for k, v in measured.items()}
    model.info = {"people": sorted(set(data["person"])), "letters": letters, "inputs": data["inputs"],
                  "frames": int(len(data["letter"])), "epochs": args.epochs, "groups": groups}
    if args.no_save:
        print("--no-save: model not written.")
        return
    model.save(args.out)
    print(f"\nSaved {args.out} ({Path(args.out).stat().st_size / 1024:.0f} KB).")
    print("To use it: set USE_MODEL = True in settings.py, then run main.py as usual.")
    if len(data["inputs"]) > len(settings.FINGERS):
        print("It was trained with the gyroscope, so the glove needs a working gyroscope to use it.")


# ============================================================================
# Loading
# ============================================================================

def load_data(folder, only_letters=None):
    """Every data/*.csv as arrays: person, session, letter, rep, try_id, bends (n x 5) and
    features (what the network reads: n x 11 with the gyroscope, or the same as bends)."""
    files = sorted(Path(folder).glob("*.csv"))
    if not files:
        raise SystemExit(f"No data in {folder}/. Record some first:  python3 collect_data.py --person yourname")
    keep = None
    if only_letters:
        keep = {x.strip().upper() for x in only_letters.split(",") if x.strip()} | {NONE}
    bend_columns = [f"bend_{f.lower()}" for f in settings.FINGERS]
    rows = []
    for path in files:
        with open(path, newline="", encoding="utf-8") as file:
            file_rows = list(csv.DictReader(file))
        if file_rows and looks_like_mock(file_rows[0]) and "mock" not in str(folder).lower():
            print(f"WARNING: {path.name} looks like pretend-glove data (its calibration matches "
                  f"mock_glove.py exactly).\n         Move it to data/mock/ unless it really came from the glove.\n")
        rows.extend(r for r in file_rows if keep is None or r["letter"] in keep)
    if not rows:
        raise SystemExit("No rows left after --letters filter.")
    data = {key: np.array([r[key] for r in rows]) for key in ("person", "session", "letter", "rep")}
    data["bends"] = np.array([[float(r[c]) for c in bend_columns] for r in rows])
    # The gyroscope's tilt and motion are used only if every row has them; one file without
    # them (recorded before the gyroscope, or with it unplugged) means bends only for everyone.
    with_imu = all(r.get(c) not in (None, "") for r in rows for c in IMU_FEATURE_NAMES)
    if with_imu:
        data["inputs"] = FEATURE_NAMES
        data["features"] = np.array([[float(r[c]) for c in FEATURE_NAMES] for r in rows])
        print("Inputs: 5 finger bends + gyroscope tilt and motion (11 numbers).")
    else:
        data["inputs"] = bend_columns
        data["features"] = data["bends"]
        some = any(r.get(IMU_FEATURE_NAMES[0]) not in (None, "") for r in rows)
        print("Inputs: the 5 finger bends only."
              + (" Some recordings have no gyroscope readings, so it isn't used. Record everyone again\n"
                 "        with the gyroscope working (or move the old CSVs out of the folder)." if some else ""))
        motion = sorted(set(r["letter"] for r in rows) & set(MOTION_LETTERS))
        if motion:
            print(f"        {', '.join(motion)} can't be told apart without the gyroscope's motion reading.")
    # One id per recorded try, so a try's 20 frames always stay together.
    data["try_id"] = np.array([f"{r['person']}|{r['session']}|{r['letter']}|{r['rep']}" for r in rows])
    return data


def looks_like_mock(row):
    """True if this recording's FLAT calibration is mock_glove.py's, to within 3 ADC counts."""
    try:
        flat = [float(row[f"flat_{f.lower()}"]) for f in settings.FINGERS]
    except (KeyError, ValueError):
        return False
    return all(abs(a - b) <= 3 for a, b in zip(flat, MOCK_FLAT))


def print_summary(data, letters):
    print(f"Loaded {len(data['letter'])} frames.\n")
    recorded = set(data["person"])
    missing = [name for name in getattr(settings, "TEAM", []) if name not in recorded]
    if missing:
        print(f"Not recorded yet: {', '.join(missing)}. Every extra person makes the network better on a "
              "stranger:\n" + "".join(f"  python3 collect_data.py --person {name}\n" for name in missing))
    tries = defaultdict(Counter)
    for try_id in set(data["try_id"]):
        person, _, letter, _ = try_id.split("|")
        tries[person][letter] += 1
    header = "person".ljust(12) + "".join(l.rjust(5) for l in letters) + "  NONE frames"
    print("Tries recorded per letter:")
    print(header)
    for person in sorted(tries):
        none_frames = int(np.sum((data["person"] == person) & (data["letter"] == NONE)))
        print(person.ljust(12) + "".join(str(tries[person][l]).rjust(5) for l in letters)
              + str(none_frames).rjust(13))
    print()


# ============================================================================
# Measured references (an improvement even without the network)
# ============================================================================

def measured_references(data, letters, mask=None):
    """Each letter's typical pose: the median over tries of each try's median."""
    mask = np.ones(len(data["letter"]), bool) if mask is None else mask
    references = {}
    for letter in letters:
        tries = sorted(set(data["try_id"][mask & (data["letter"] == letter)]))
        if tries:
            medians = [np.median(data["bends"][data["try_id"] == t], axis=0) for t in tries]
            references[letter] = np.median(medians, axis=0)
    return references


def print_references(references):
    print("Measured REFERENCES (median of every try). The model file saves these for lessons.")
    print("To use them without the network, paste only the letters you demo into settings.py:")
    print("REFERENCES = {")
    for letter, pose in references.items():
        print(f'    "{letter}": [{", ".join(f"{x:.2f}" for x in pose)}],')
    print("}\n")


# ============================================================================
# Testing
# ============================================================================

def gate(signs, confidence, margin):
    """main.py's rule: show a letter only if confident enough and clearly ahead. None = '--'."""
    active = settings.ACTIVE_LETTERS
    return [s if (s is not None and c >= settings.ACCEPT_CONFIDENCE and m >= settings.MARGIN
                  and (not active or s in active)) else None
            for s, c, m in zip(signs, confidence, margin)]


def nearest_reference(bends, references):
    """sign_feedback.recognize_with_margin()'s rule, for many readings at once."""
    if not references:
        return [None] * len(bends), np.zeros(len(bends)), np.zeros(len(bends))
    names = list(references)
    matrix = np.array([references[n] for n in names])
    confidence = 1.0 - np.mean(np.abs(bends[:, None, :] - matrix[None]), axis=2)
    order = np.argsort(-confidence, axis=1)
    rows = np.arange(len(bends))
    best = confidence[rows, order[:, 0]]
    second = confidence[rows, order[:, 1]] if len(names) > 1 else np.zeros(len(bends))
    return [names[i] for i in order[:, 0]], best, best - second


def network_predictions(model, features, use_groups=True):
    results = [model.recognize_with_margin(f, settings.ACTIVE_LETTERS, settings.MODEL_MAX_DISTANCE,
                                           use_groups=use_groups)
               for f in features]
    signs, confidence, margin = zip(*results)
    return list(signs), np.array(confidence), np.array(margin)


def evaluate(data, letters, classes, epochs, merge=MERGE_CONFUSION):
    """Run the held-out test, print the results, and return the look-alike groups to save."""
    people = sorted(set(data["person"]))
    if len(people) >= 2:
        print(f"Test: leave-one-person-out ({len(people)} people). Each person is tested on a")
        print("network trained only on the others, like a stranger at a demo.\n")
        folds = [(p, data["person"] == p) for p in people]
    else:
        reps = sorted(set(data["rep"]) - {"0"}, key=int)
        print("WARNING: only one person recorded, so this can only test on tries the network")
        print("didn't train on, from the SAME hand. Real accuracy on a new person will be lower.")
        print("Record at least one more person for an honest number.\n")
        folds = [(f"try {r}", (data["rep"] == r) | ((data["letter"] == NONE) & (np.arange(len(data["rep"])) % len(reps) == i)))
                 for i, r in enumerate(reps)]

    methods = ("settings.REFERENCES", "Measured references", "Neural network", "Network, groups")
    shown_all = {m: [] for m in methods}
    truth_all, top_all, distances, tested = [], [], [], []
    for name, test in folds:
        train = ~test
        if not np.any(train) or not np.any(test):
            continue
        model = train_network(data["features"][train], data["letter"][train], classes, epochs=epochs)
        bends, features, truth = data["bends"][test], data["features"][test], data["letter"][test]
        # The reference methods only see the bends, so they can't use J and Z (traced letters).
        typed = {k: np.array(v) for k, v in settings.REFERENCES.items() if k not in MOTION_LETTERS}
        measured = {k: v for k, v in measured_references(data, letters, train).items() if k not in MOTION_LETTERS}
        shown = {
            methods[0]: gate(*nearest_reference(bends, typed)),
            methods[1]: gate(*nearest_reference(bends, measured)),
            methods[2]: gate(*network_predictions(model, features, use_groups=False)),
        }
        for method in methods[:3]:
            shown_all[method].extend(shown[method])
        truth_all.extend(truth)
        # The network's first choice with no "--" rules: this is what shows the look-alikes.
        top_all.extend(model.classes[i] for i in np.argmax(model.probabilities(features), axis=1))
        distances.extend(model.nearest_distance(f, t) for f, t in zip(features, truth) if t != NONE)
        tested.append((model, features))
        right = score(truth, shown[methods[2]])[0]
        print(f"  {name:<12} network: right letter {right:.0%}")

    truth_all = np.array(truth_all)
    groups = find_groups(truth_all, np.array(top_all), letters, merge) if merge else []
    print_groups(groups, merge)
    for model, bends in tested:                     # the same held-out readings, look-alikes shown together
        model.groups = groups
        shown_all[methods[3]].extend(gate(*network_predictions(model, bends)))

    print("\nResults on held-out readings (each reading = one 1/20 s frame):")
    print(f"  {'method':<22}{'right letter shown':>20}{'wrong letter shown':>20}{'letter during NONE':>20}")
    for method in methods:
        right, wrong, false_none = score(truth_all, shown_all[method])
        none_text = "n/a" if np.isnan(false_none) else f"{false_none:.0%}"
        print(f"  {method:<22}{right:>20.0%}{wrong:>20.0%}{none_text:>20}")
    print("  (For 'Network, groups', showing U/V for a V counts as right.)")
    missing = sorted(set(letters) - set(settings.REFERENCES))
    if missing:
        print(f"  (settings.REFERENCES has no entry for {', '.join(missing)}, so it can never show them.)")

    shown_by_glove = methods[3] if settings.SHOW_LOOK_ALIKE_GROUPS else methods[2]
    print_per_letter(truth_all, shown_all[shown_by_glove], letters, shown_by_glove)

    distances = np.array(distances)
    if len(distances):
        rejected = np.mean(distances > settings.MODEL_MAX_DISTANCE)
        print(f"\nDistance check: MODEL_MAX_DISTANCE = {settings.MODEL_MAX_DISTANCE} turns away "
              f"{rejected:.0%} of real letters from new hands")
        print(f"  (95% of them are within {np.percentile(distances, 95):.2f}). If that's more than ~10%,"
              f" raise it in settings.py or record more people.")
    return groups


def find_groups(truth, top, letters, merge=MERGE_CONFUSION):
    """Letters the network mixes up on new hands, joined into groups.

    A letter joins another letter's group if at least `merge` of its readings
    came out as any letter of that group. This repeats until nothing changes,
    so mix-ups spread over a group count together: if S is read as N 10% of the
    time and as T 11%, and N/T is already a group, S joins it (21%).
    """
    group = {letter: frozenset([letter]) for letter in letters}
    changed = True
    while changed:
        changed = False
        for a in letters:
            index = truth == a
            if not np.any(index):
                continue
            landed = Counter(group[t] for t in top[index] if t in group and group[t] != group[a])
            for other, count in landed.items():
                if count / np.sum(index) >= merge:
                    joined = group[a] | other
                    for letter in joined:
                        group[letter] = joined
                    changed = True
                    break
    return sorted(sorted(str(x) for x in g) for g in set(group.values()) if len(g) > 1)


def print_groups(groups, merge):
    print()
    if not merge:
        print("Look-alike groups: off (--no-groups). Every letter is shown on its own.")
    elif not groups:
        print("Look-alike groups: none. The network tells every letter apart on new hands.")
    else:
        together = "shown together" if settings.SHOW_LOOK_ALIKE_GROUPS else \
            "the glove still shows one letter, the more likely one"
        print(f"Look-alike groups (mixed up on {merge:.0%}+ of a new hand's readings; {together}):")
        for group in groups:
            print(f"  {'/'.join(group):<12} {describe_look_alike(group)}")
        print("  To split a group, add a sensor that measures the difference (see TRAINING.md),")
        print("  or keep only one letter of it in ACTIVE_LETTERS.")


def print_per_letter(truth, shown, letters, method="Neural network"):
    """One row per letter: how often the glove showed it, something wrong, or nothing."""
    shown = np.array([s if s is not None else "--" for s in shown])
    print(f"\nPer letter ({method}, what the glove shows), on people it was not trained on:")
    print(f"  {'sign':<6}{'right':>7}{'wrong':>7}{'--':>7}   most common mistake")
    for letter in letters + [NONE]:
        index = truth == letter
        if not np.any(index):
            continue
        hit = np.array([letter in letters_in(s) for s in shown[index]], dtype=bool)
        blank = shown[index] == "--"
        wrong = ~hit & ~blank
        mistake = ""
        if np.any(wrong):
            label, count = Counter(shown[index][wrong]).most_common(1)[0]
            mistake = f"{label} ({count / np.sum(index):.0%})"
        if letter == NONE:
            print(f"  {letter:<6}{'':>7}{np.mean(~blank):>7.0%}{np.mean(blank):>7.0%}   {mistake}"
                  "   (wrong = a letter shown while not signing)")
        else:
            print(f"  {letter:<6}{np.mean(hit):>7.0%}{np.mean(wrong):>7.0%}{np.mean(blank):>7.0%}   {mistake}")


def score(truth, shown):
    """(right letter shown, wrong letter shown on a letter, any letter shown during NONE)."""
    truth = np.asarray(truth)
    shown = np.array([s if s is not None else "--" for s in shown])
    letter = truth != NONE
    hit = np.array([t in letters_in(s) for t, s in zip(truth, shown)], dtype=bool)
    right = np.mean(hit[letter]) if np.any(letter) else np.nan
    wrong = np.mean(~hit[letter] & (shown[letter] != "--")) if np.any(letter) else np.nan
    false_none = np.mean(shown[~letter] != "--") if np.any(~letter) else np.nan
    return right, wrong, false_none


def parse_args():
    parser = argparse.ArgumentParser(description="Train and test the ASL letter network")
    parser.add_argument("--data", default=str(DATA_DIR), help="folder of CSVs from collect_data.py")
    parser.add_argument("--out", default=str(settings.MODEL_FILE), help="where to save the model")
    parser.add_argument("--letters", help="only these letters, e.g. A,B,C (NONE is always kept)")
    parser.add_argument("--epochs", type=int, default=150, help="training passes over the data")
    parser.add_argument("--no-save", action="store_true", help="test only; don't write the model")
    parser.add_argument("--merge", type=float, default=MERGE_CONFUSION,
                        help=f"group letters mixed up this often on new hands (default {MERGE_CONFUSION})")
    parser.add_argument("--no-groups", action="store_true", help="never group look-alike letters")
    return parser.parse_args()


if __name__ == "__main__":
    main()
