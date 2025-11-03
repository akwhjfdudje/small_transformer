import os, random

OUT_DIR = "data"
OUT_FILE = os.path.join(OUT_DIR, "program_corpus.txt")
NUM_LINES = 3000
SEED = 12345

random.seed(SEED)

# Templates for prose lines
prose_templates = [
    "The function {name} returns the {ret} for a given {arg}.",
    "Use constant-time algorithms when handling cryptographic keys.",
    "Always validate input before dereferencing a pointer in C++.",
    "This module implements a thread-safe queue with lock-free push/pop.",
    "Prefer immutable data structures when writing concurrent code.",
    "The algorithm uses a priority queue to achieve O(n log n) complexity.",
    "Document the public API with examples and expected edge cases.",
    "Profiling revealed the bottleneck is I/O-bound, not CPU-bound.",
    "Use vectorized operations to speed up numeric computations in Python.",
    "When compiling with nvcc set optimization flags appropriate for release.",
    "A guard clause simplifies the early-exit logic and improves readability.",
    "We cache DNS lookups to reduce latency on repeated network calls.",
    "This patch fixes a memory leak caused by an unfreed buffer.",
    "The build system uses CMake with separate targets for tests and libs.",
    "Prefer explicit error handling over exceptions for low-level code.",
    "The data structure supports amortized O(1) append operations.",
    "Use a sliding window to compute running statistics efficiently.",
    "An exponential backoff reduces contention during retries.",
    "Avoid global state when designing libraries to enable reentrancy.",
    "The test suite uses fixtures to set up and tear down temporary state.",
    "We rely on semantic versioning to indicate breaking API changes.",
    "Instrument code with lightweight logging to diagnose production issues.",
    "A bloom filter quickly tests approximate membership with small memory.",
    "Use integer types of fixed width when serializing cross-platform data.",
    "Explicitly close file descriptors in error paths to prevent leaks.",
    "The following heuristic balances load across available workers.",
    "Implement a cache eviction policy based on LRU with TTL support.",
    "Prefer zero-copy techniques for high-throughput network proxies.",
    "This function is annotated as noexcept to avoid unwinding overhead.",
]

# Short code line templates (single-line snippets)
python_snippets = [
    "def {name}({arg}): return {expr}",
    "for i in range({n}): print(i)",
    "with open('{file}','r') as f: data = f.read()",
    "result = [x*2 for x in seq]",
    "nums = sorted(nums, key=lambda x: x.value)",
    "import math; x = math.sqrt({n})",
    "if __name__ == '__main__': main()",
    "class {name}:\n    pass",
    "data = dict(zip(keys, values))",
    "s = ' '.join(map(str, items))",
]

cpp_snippets = [
    "std::vector<int> v; v.reserve({n});",
    "for (int i = 0; i < n; ++i) std::cout << i << '\\n';",
    "std::unique_ptr<Foo> p = std::make_unique<Foo>();",
    "if (ptr) {{ ptr->do_work(); }}",
    "std::sort(v.begin(), v.end());",
    "template<typename T> T add(T a, T b) {{ return a + b; }}",
    "constexpr int kMax = {n};",
    "std::atomic<int> counter(0); counter.fetch_add(1);",
]

bash_snippets = [
    "gcc -O2 -o app main.c utils.c",
    "tar -czf archive.tgz src/",
    "docker run -it --rm myimage:latest /bin/bash",
    "ssh -i id_rsa user@host.example.com",
    "git clone https://github.com/example/repo.git",
]

cuda_snippets = [
    "__global__ void kernel(float *a) {{ int i = blockIdx.x*blockDim.x + threadIdx.x; a[i] += 1.0f; }}",
    "cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);",
    "dim3 grid((N+255)/256); kernel<<<grid,256>>>(dptr);",
]

sql_snippets = [
    "SELECT id, name FROM users WHERE active = 1 ORDER BY id DESC;",
    "CREATE INDEX idx_user_email ON users(email);",
    "INSERT INTO logs (ts, level, msg) VALUES (NOW(), 'info', 'started');",
]

# Algorithmic descriptions / one-liners
alg_templates = [
    "Use Dijkstra's algorithm to compute shortest paths in weighted graphs.",
    "Apply quickselect to find the k-th smallest element in average O(n) time.",
    "Use reservoir sampling to select uniformly from a stream of unknown size.",
    "Use prefix sums to answer range sum queries in O(1) after O(n) preprocessing.",
    "Use a Fenwick tree for dynamic prefix sums with O(log n) updates.",
    "Apply the two-pointer technique for subarray problems with linear time.",
    "Use Tarjan's algorithm to find strongly connected components in O(V+E).",
    "Use a union-find (disjoint set) to maintain connectivity under merges.",
]

# Doc-like one-liners and config examples
doc_templates = [
    "Parameters: count (int) - number of retries before giving up.",
    "Returns: True if the resource was updated, False otherwise.",
    "Example: client.connect(host='127.0.0.1', port=8080)",
    "Config: max_connections=100; timeout_seconds=30",
    "Note: this API is experimental and subject to change.",
    "Warning: enabling debug mode may leak sensitive information to logs.",
]

# utility to generate random names and numbers
def rand_name():
    return random.choice(["process", "handler", "compute", "merge", "encode", "decode", "serialize", "parse", "update", "filter"]) + str(random.randint(1,999))

def rand_file():
    return random.choice(["main.py", "utils.py", "kernel.cu", "server.cpp", "Makefile", "Dockerfile", "README.md"])

def rand_num(a=2, b=1024):
    return random.randint(a,b)

# Build pool of line generators
lines = []

# Strategy: create roughly balanced mixture
num_prose = int(NUM_LINES * 0.40)
num_code = int(NUM_LINES * 0.40)
num_algo = int(NUM_LINES * 0.12)
num_doc = NUM_LINES - (num_prose + num_code + num_algo)

# Generate prose lines
for _ in range(num_prose):
    tmpl = random.choice(prose_templates)
    line = tmpl.format(name=rand_name(), arg="input", ret="result", n=rand_num(), file=rand_file())
    lines.append(line)

# Generate code lines (mix languages)
code_templates = python_snippets + cpp_snippets + bash_snippets + cuda_snippets + sql_snippets
for _ in range(num_code):
    tmpl = random.choice(code_templates)
    # some templates contain {name}, {arg}, {n}, {file}
    try:
        line = tmpl.format(name=rand_name(), arg="x", expr="x*2", n=rand_num(), file=rand_file())
    except Exception:
        line = tmpl
    # single-line normalization: replace newlines inside templates if any
    line = " ".join(line.splitlines())
    lines.append(line)

# Algorithmic and doc lines
for _ in range(num_algo):
    lines.append(random.choice(alg_templates))
for _ in range(num_doc):
    lines.append(random.choice(doc_templates))

# If any shortages or overshoot, adjust
if len(lines) < NUM_LINES:
    # add more prose until fill
    while len(lines) < NUM_LINES:
        tmpl = random.choice(prose_templates)
        lines.append(tmpl.format(name=rand_name(), arg="input", ret="result", n=rand_num(), file=rand_file()))
elif len(lines) > NUM_LINES:
    lines = lines[:NUM_LINES]

# Shuffle to mix code and prose
random.shuffle(lines)

# ensure directory
os.makedirs(OUT_DIR, exist_ok=True)

# write file
with open(OUT_FILE, "w", encoding="utf-8") as f:
    for ln in lines:
        # ensure line is stripped and single-line
        ln = ln.strip()
        if not ln:
            ln = "noop"
        f.write(ln + "\n")

print(f"Wrote {NUM_LINES} lines to {OUT_FILE}")
