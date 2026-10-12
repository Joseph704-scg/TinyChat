"""
tinychat.py - a chatbot brain built from scratch.

This one file holds a complete, very small language model of the same kind that
powers the big chatbots (a "transformer"). It uses NumPy for arithmetic on
grids of numbers and nothing else: no AI library.

    python3 tinychat.py train    learn from chat_data.txt (about 20 minutes)
    python3 tinychat.py chat     talk to the trained model

The brain cannot reach the internet or read a clock. For questions that need
those it has learned to answer with a command such as [weather], and tools.py
does the fetching. webchat.py puts the whole thing on a web page.

How it works, in the order the code appears:

    1. SETTINGS    the size of the brain and how long it practices
    2. TENSOR      numbers that remember how they were computed, so that after
                   a wrong guess we can work out which way to nudge each one
    3. THE MODEL   the transformer: it reads text one character at a time and
                   scores every possible next character
    4. TRAINING    show it real text, measure how wrong its guesses are,
                   nudge every number to be less wrong, repeat
    5. CHATTING    feed it "you: hello / bot: " and let it keep guessing
                   the next character until it finishes the line, then run
                   a tool if the line it wrote was a command
"""
import sys
import time

import numpy as np

import tools

# ----------------------------------------------------------------------------
# 1. SETTINGS
# ----------------------------------------------------------------------------
DATA_FILE = "chat_data.txt"
WEIGHTS_FILE = "tinychat_weights.npz"

BLOCK = 128           # how many characters back the model can see
EMBED = 96            # how many numbers describe each character
HEADS = 4             # how many things attention can look for at once
LAYERS = 2            # how many times the model refines its understanding
BATCH = 32            # how many text snippets it studies per step
STEPS = 7000          # how many practice steps
LEARNING_RATE = 3e-3  # how big each nudge is
TEMPERATURE = 0.5     # chat randomness: lower is safer, higher is wilder
DTYPE = np.float32


# ----------------------------------------------------------------------------
# 2. TENSOR
#
# A Tensor is a grid of numbers plus a note about which Tensors it was computed
# from. Calling backward() on the final "how wrong were we" number walks those
# notes in reverse and fills in .grad on every Tensor along the way: how much
# the wrongness would change if that number changed. This is called
# backpropagation, and it is the entire mechanism by which the model learns.
# ----------------------------------------------------------------------------
class Tensor:
    def __init__(self, data, parents=(), backward_fn=None):
        self.data = np.asarray(data, dtype=DTYPE)
        self.grad = None
        self.parents = parents
        self.backward_fn = backward_fn

    def add_grad(self, g):
        self.grad = g if self.grad is None else self.grad + g

    def backward(self):
        # List every Tensor that led to this one, inputs first.
        order, seen = [], set()

        def visit(t):
            if id(t) not in seen:
                seen.add(id(t))
                for parent in t.parents:
                    visit(parent)
                order.append(t)

        visit(self)
        self.grad = np.ones_like(self.data)
        # Walk from the result back to the inputs, passing blame along.
        for t in reversed(order):
            if t.backward_fn is not None:
                t.backward_fn(t.grad)

    # Each operation computes its result and records how to pass blame back.
    def __add__(self, other):
        if not isinstance(other, Tensor):      # adding a plain constant
            return Tensor(self.data + other, (self,), self.add_grad)

        def backward_fn(g):
            self.add_grad(unbroadcast(g, self.data.shape))
            other.add_grad(unbroadcast(g, other.data.shape))

        return Tensor(self.data + other.data, (self, other), backward_fn)

    def __mul__(self, other):
        if not isinstance(other, Tensor):      # scaling by a plain constant
            return Tensor(self.data * other, (self,), lambda g: self.add_grad(g * other))

        def backward_fn(g):
            self.add_grad(unbroadcast(g * other.data, self.data.shape))
            other.add_grad(unbroadcast(g * self.data, other.data.shape))

        return Tensor(self.data * other.data, (self, other), backward_fn)

    def __matmul__(self, other):               # matrix multiplication
        def backward_fn(g):
            self.add_grad(unbroadcast(g @ other.data.swapaxes(-1, -2), self.data.shape))
            other.add_grad(unbroadcast(self.data.swapaxes(-1, -2) @ g, other.data.shape))

        return Tensor(self.data @ other.data, (self, other), backward_fn)

    def reshape(self, *shape):
        return Tensor(self.data.reshape(shape), (self,),
                      lambda g: self.add_grad(g.reshape(self.data.shape)))

    def transpose(self, *axes):
        return Tensor(self.data.transpose(axes), (self,),
                      lambda g: self.add_grad(g.transpose(np.argsort(axes))))


def unbroadcast(g, shape):
    """NumPy stretches small arrays to fit big ones; undo that for the blame."""
    while g.ndim > len(shape):
        g = g.sum(axis=0)
    for axis, size in enumerate(shape):
        if size == 1 and g.shape[axis] != 1:
            g = g.sum(axis=axis, keepdims=True)
    return g


def lookup(table, ids):
    """Pick rows out of a table: turns character numbers into their vectors."""
    def backward_fn(g):
        grad = np.zeros_like(table.data)
        np.add.at(grad, ids, g)
        table.add_grad(grad)

    return Tensor(table.data[ids], (table,), backward_fn)


def relu(x):
    """Keep positive numbers, zero out negative ones."""
    return Tensor(np.maximum(x.data, 0), (x,), lambda g: x.add_grad(g * (x.data > 0)))


def softmax(x):
    """Turn a row of scores into positive shares that add up to 1."""
    shifted = np.exp(x.data - x.data.max(axis=-1, keepdims=True))
    out = shifted / shifted.sum(axis=-1, keepdims=True)

    def backward_fn(g):
        x.add_grad(out * (g - (g * out).sum(axis=-1, keepdims=True)))

    return Tensor(out, (x,), backward_fn)


def normalize(x, eps=1e-5):
    """Recentre and rescale each vector so the numbers stay a sensible size."""
    centred = x.data - x.data.mean(axis=-1, keepdims=True)
    spread = np.sqrt((centred ** 2).mean(axis=-1, keepdims=True) + eps)
    out = centred / spread

    def backward_fn(g):
        x.add_grad((g - g.mean(axis=-1, keepdims=True)
                    - out * (g * out).mean(axis=-1, keepdims=True)) / spread)

    return Tensor(out, (x,), backward_fn)


def wrongness(scores, targets):
    """The loss: how surprised the model was by the real next characters.

    0 means it predicted every one perfectly; higher means worse guesses.
    """
    flat = scores.data.reshape(-1, scores.data.shape[-1])
    flat = flat - flat.max(axis=-1, keepdims=True)
    probs = np.exp(flat)
    probs /= probs.sum(axis=-1, keepdims=True)
    answers = targets.reshape(-1)
    rows = np.arange(len(answers))
    loss = -np.log(probs[rows, answers] + 1e-12).mean()

    def backward_fn(g):
        grad = probs.copy()
        grad[rows, answers] -= 1          # (what it guessed) - (the truth)
        scores.add_grad(g * (grad / len(answers)).reshape(scores.data.shape))

    return Tensor(loss, (scores,), backward_fn)


# ----------------------------------------------------------------------------
# 3. THE MODEL
#
# Everything the model "knows" lives in the weights: about a quarter of a
# million numbers that start out random. forward() is the fixed recipe that combines
# those numbers with the text to score each possible next character.
# ----------------------------------------------------------------------------
def new_weights(vocab_size, rng):
    def random(*shape):
        return Tensor(rng.normal(0, 0.02, shape))

    def zeros(*shape):
        return Tensor(np.zeros(shape))

    def ones(*shape):
        return Tensor(np.ones(shape))

    w = {
        "char": random(vocab_size, EMBED),     # what each character means
        "position": random(BLOCK, EMBED),      # what each position means
    }
    for i in range(LAYERS):
        w[f"layer{i}.norm1.scale"], w[f"layer{i}.norm1.shift"] = ones(EMBED), zeros(EMBED)
        w[f"layer{i}.query"] = random(EMBED, EMBED)
        w[f"layer{i}.key"] = random(EMBED, EMBED)
        w[f"layer{i}.value"] = random(EMBED, EMBED)
        w[f"layer{i}.attn_out"], w[f"layer{i}.attn_out_bias"] = random(EMBED, EMBED), zeros(EMBED)
        w[f"layer{i}.norm2.scale"], w[f"layer{i}.norm2.shift"] = ones(EMBED), zeros(EMBED)
        w[f"layer{i}.think_in"], w[f"layer{i}.think_in_bias"] = random(EMBED, 4 * EMBED), zeros(4 * EMBED)
        w[f"layer{i}.think_out"], w[f"layer{i}.think_out_bias"] = random(4 * EMBED, EMBED), zeros(EMBED)
    w["final_norm.scale"], w["final_norm.shift"] = ones(EMBED), zeros(EMBED)
    w["output"], w["output_bias"] = random(EMBED, vocab_size), zeros(vocab_size)
    return w


def forward(w, ids):
    """ids: a grid of character numbers, shape (snippets, length).

    Returns a score for every possible next character at every position.
    """
    snippets, length = ids.shape
    head_size = EMBED // HEADS

    # Start: each character becomes a vector, plus a vector for where it sits.
    x = lookup(w["char"], ids) + lookup(w["position"], np.arange(length))

    # A character may look at earlier characters but never at later ones,
    # otherwise it could cheat by peeking at the answer.
    no_peeking = np.triu(np.full((length, length), -1e9, dtype=DTYPE), k=1)

    for i in range(LAYERS):
        name = f"layer{i}."

        # --- Attention: every character gathers information from earlier ones.
        h = normalize(x) * w[name + "norm1.scale"] + w[name + "norm1.shift"]

        def split_heads(t):
            return t.reshape(snippets, length, HEADS, head_size).transpose(0, 2, 1, 3)

        query = split_heads(h @ w[name + "query"])    # what am I looking for?
        key = split_heads(h @ w[name + "key"])        # what do I contain?
        value = split_heads(h @ w[name + "value"])    # what do I pass along?

        match = (query * head_size ** -0.5) @ key.transpose(0, 1, 3, 2) + no_peeking
        attention = softmax(match)                    # how much to listen to each
        gathered = (attention @ value).transpose(0, 2, 1, 3).reshape(snippets, length, EMBED)
        x = x + gathered @ w[name + "attn_out"] + w[name + "attn_out_bias"]

        # --- Thinking: each character processes what it just gathered.
        h = normalize(x) * w[name + "norm2.scale"] + w[name + "norm2.shift"]
        h = relu(h @ w[name + "think_in"] + w[name + "think_in_bias"])
        x = x + h @ w[name + "think_out"] + w[name + "think_out_bias"]

    # Finish: turn each vector into one score per character in the vocabulary.
    x = normalize(x) * w["final_norm.scale"] + w["final_norm.shift"]
    return x @ w["output"] + w["output_bias"]


# ----------------------------------------------------------------------------
# 4. TRAINING
# ----------------------------------------------------------------------------
def load_text():
    with open(DATA_FILE) as f:
        text = f.read()
    chars = sorted(set(text))            # the vocabulary: every character used
    return text, chars


class Adam:
    """Nudges each weight against its blame, with momentum so progress is
    smooth and a per-weight step size so no weight moves too fast or slow."""

    def __init__(self, weights):
        self.weights = weights
        self.m = {k: np.zeros_like(t.data) for k, t in weights.items()}
        self.v = {k: np.zeros_like(t.data) for k, t in weights.items()}
        self.t = 0

    def step(self, rate):
        self.t += 1
        for k, tensor in self.weights.items():
            g = tensor.grad
            self.m[k] = 0.9 * self.m[k] + 0.1 * g
            self.v[k] = 0.999 * self.v[k] + 0.001 * g * g
            m_hat = self.m[k] / (1 - 0.9 ** self.t)
            v_hat = self.v[k] / (1 - 0.999 ** self.t)
            tensor.data -= rate * m_hat / (np.sqrt(v_hat) + 1e-8)
            tensor.grad = None


def train():
    rng = np.random.default_rng(0)
    text, chars = load_text()
    to_id = {c: i for i, c in enumerate(chars)}
    data = np.array([to_id[c] for c in text])

    weights = new_weights(len(chars), rng)
    count = sum(t.data.size for t in weights.values())
    print(f"{len(text):,} characters of text, {len(chars)} different characters")
    print(f"the model has {count:,} numbers to tune\n")

    optimizer = Adam(weights)
    recent, start = [], time.time()
    for step in range(1, STEPS + 1):
        # Grab random snippets. The target is the same text shifted one
        # character along: at every position, "what comes next?"
        starts = rng.integers(0, len(data) - BLOCK - 1, BATCH)
        inputs = np.stack([data[s:s + BLOCK] for s in starts])
        targets = np.stack([data[s + 1:s + BLOCK + 1] for s in starts])

        loss = wrongness(forward(weights, inputs), targets)   # guess, and score it
        loss.backward()                                       # assign blame

        # Nudge: gentle at first, strongest early on, fading towards the end.
        warmup = min(1.0, step / 100)
        fade = 0.5 * (1 + np.cos(np.pi * step / STEPS))
        optimizer.step(LEARNING_RATE * warmup * (0.1 + 0.9 * fade))

        recent.append(float(loss.data))
        if step % 100 == 0 or step == 1:
            average = sum(recent) / len(recent)
            recent = []
            print(f"step {step:5d}/{STEPS}   wrongness {average:.3f}   {time.time() - start:5.0f}s")
        if step % 500 == 0:
            print(f'      you: hello\n      bot: {reply(weights, chars, "hello", rng)}')

    np.savez(WEIGHTS_FILE, vocabulary=np.array("".join(chars)),
             **{k: t.data for k, t in weights.items()})
    print(f"\nsaved the trained model to {WEIGHTS_FILE}")
    print("now run:  python3 tinychat.py chat")


# ----------------------------------------------------------------------------
# 5. CHATTING
# ----------------------------------------------------------------------------
def reply(weights, chars, message, rng, max_chars=120):
    """Write the conversation so far, then let the model continue it."""
    to_id = {c: i for i, c in enumerate(chars)}
    message = "".join(c for c in message.lower() if c in to_id and c != "\n")
    ids = [to_id[c] for c in f"you: {message}\nbot: "]
    answer = ""
    for _ in range(max_chars):
        scores = forward(weights, np.array([ids[-BLOCK:]])).data[0, -1]
        scores = scores.astype(np.float64) / TEMPERATURE
        chances = np.exp(scores - scores.max())
        chances /= chances.sum()
        next_id = int(rng.choice(len(chars), p=chances))   # roll the dice
        if chars[next_id] == "\n":                         # end of its line
            break
        ids.append(next_id)
        answer += chars[next_id]
    return answer


class Bot:
    """The finished chatbot: the trained brain plus its tools.

    This is the piece to build on. From any other Python program:

        from tinychat import Bot
        bot = Bot()
        print(bot.ask("what is the weather in tokyo"))
    """

    def __init__(self):
        try:
            saved = np.load(WEIGHTS_FILE)
        except FileNotFoundError:
            sys.exit("No trained model found. Run this first:  python3 tinychat.py train")
        self.chars = list(str(saved["vocabulary"]))
        self.weights = {k: Tensor(saved[k]) for k in saved.files if k != "vocabulary"}
        self.size = sum(t.data.size for t in self.weights.values())
        self.rng = np.random.default_rng()

    def think(self, message):
        """What the brain alone says: a sentence, or a command like [news]."""
        return reply(self.weights, self.chars, message, self.rng)

    def respond(self, message):
        """Returns the answer and the name of the tool used (None if none)."""
        return tools.use_tools(message, self.think(message))

    def ask(self, message):
        return self.respond(message)[0]


def chat():
    bot = Bot()
    print("Chat with your model. Press Ctrl+C or type 'quit' to stop.\n")
    while True:
        try:
            message = input("you: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if message.lower() in ("quit", "exit"):
            break
        if message:
            answer, tool = bot.respond(message)
            if tool:
                print(f"     (the brain asked for its {tool} tool)")
            print("bot:", answer, "\n")


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else ""
    if command == "train":
        train()
    elif command == "chat":
        chat()
    else:
        print("usage:  python3 tinychat.py train   (then)   python3 tinychat.py chat")
