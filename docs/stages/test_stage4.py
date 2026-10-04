"""Standalone test of Step 4's credit-assignment + action-picking logic,
extracted from the generated agent so we can verify it without needing the
full Agent base class / arcengine / live gateway."""
import random


class Core:
    def __init__(self):
        self.action_stats = {}
        self.region_stats = {}
        self.graph = {}
        self.tried_from = {}
        self.path_value = {}
        self.episode_path = []
        self.prev_levels_completed = 0
        self.last_hash = None
        self.pending_id = None

    def record_outcome(self, current_hash, levels_completed):
        if self.pending_id is None or self.last_hash is None:
            return
        changed = current_hash != self.last_hash
        stats = self.action_stats.setdefault(self.pending_id, [0, 0])
        stats[0] += 1
        stats[1] += int(changed)
        self.graph.setdefault(self.last_hash, {})[self.pending_id] = current_hash
        self.tried_from.setdefault(self.last_hash, set()).add(self.pending_id)
        self.episode_path.append((self.last_hash, self.pending_id))

        if levels_completed > self.prev_levels_completed:
            for pair in self.episode_path:
                self.path_value[pair] = self.path_value.get(pair, 0.0) + 1.0
            self.episode_path = []
        self.prev_levels_completed = levels_completed

    def pick_action(self, node_hash, legal_ids):
        valued = [(a, self.path_value[(node_hash, a)]) for a in legal_ids
                  if self.path_value.get((node_hash, a), 0.0) > 0]
        if valued:
            best_val = max(v for _, v in valued)
            best_actions = [a for a, v in valued if v == best_val]
            return random.choice(best_actions)
        tried_here = self.tried_from.get(node_hash, set())
        untried = [a for a in legal_ids if a not in tried_here]
        if untried:
            return random.choice(untried)
        return random.choice(legal_ids)


# --- Test 1: credit assignment on level-up ---
c = Core()
# Simulate: state A --act1--> B --act2--> C, then level completes at C
c.last_hash, c.pending_id = None, None
# turn 1: no prior action, just establishes last_hash="A"
c.record_outcome("A", levels_completed=0)
c.last_hash = "A"
c.pending_id = 1  # took action 1 from A
c.record_outcome("B", levels_completed=0)  # arrived at B, no level-up yet
assert c.path_value == {}, f"expected no credit yet, got {c.path_value}"
assert c.episode_path == [("A", 1)], f"episode_path wrong: {c.episode_path}"

c.last_hash = "B"
c.pending_id = 2  # took action 2 from B
c.record_outcome("C", levels_completed=1)  # arrived at C, level completed!
assert c.path_value == {("A", 1): 1.0, ("B", 2): 1.0}, f"credit wrong: {c.path_value}"
assert c.episode_path == [], f"episode_path should clear after credit: {c.episode_path}"
print("Test 1 (credit assignment) PASSED:", c.path_value)

# --- Test 2: path_value persists across a simulated RESET, and is preferred ---
# Simulate agent resets (episode_path cleared, but NOT path_value) and lands
# back at state "A" again. It should now strongly prefer action 1 from A,
# since (A,1) has a proven value of 1.0 and nothing else does.
c.episode_path = []  # what a RESET does in choose_action
c.last_hash = None
c.pending_id = None
legal = [1, 3, 4]  # action 1 is the proven-good one; 3,4 are untried from A
chosen = c.pick_action("A", legal)
assert chosen == 1, f"expected learned action 1 to be preferred, got {chosen}"
print("Test 2 (learned value overrides frontier) PASSED: chose action", chosen)

# --- Test 3: with no learned value, frontier (untried) wins over tried ---
c2 = Core()
c2.tried_from["X"] = {5}  # action 5 already tried from X
legal2 = [5, 6, 7]
chosen2 = c2.pick_action("X", legal2)
assert chosen2 in (6, 7), f"expected an untried action, got {chosen2}"
print("Test 3 (frontier preference with no learned value) PASSED: chose action", chosen2)

# --- Test 4: decrease in levels_completed (e.g. counter resets) doesn't over-credit ---
c3 = Core()
c3.last_hash = "P"
c3.pending_id = 9
c3.record_outcome("Q", levels_completed=0)  # drop from whatever prev was (0) to 0, no credit
assert c3.path_value == {}, f"should not credit on non-increase: {c3.path_value}"
print("Test 4 (no false credit on non-increase) PASSED")

print("\\nALL TESTS PASSED")
