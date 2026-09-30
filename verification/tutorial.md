# Two ways to verify programs with Lean (and why the distinction matters)

## One-sentence summary

Lean checks a *mathematical proof about a precisely stated program and property*. You can make the program a Lean function, or represent an existing language/machine *inside* Lean and prove facts about that representation. In either case, you still have to establish that the thing you proved corresponds to the program you intend to run.

## The big idea

Imagine a recipe and a food-safety claim. You can rewrite the recipe in a language the inspector understands, then inspect that new recipe. Or you can give the inspector a precise model of the original recipe's instructions and inspect the original recipe through that model. In the first case, you must justify that the rewrite preserves the recipe; in the second, you must justify that the model captures what the kitchen really does.

Here “verify” means prove a statement for **all executions covered by a stated model**, not test a few inputs. A *specification* (“spec”) states the intended property. The spec can be supplied by a human, but a human sentence must first become a precise mathematical statement. Lean checks the proof **relative to that statement**; it cannot decide whether the human asked for the right thing.

I assume your starting point is a program written in a language other than Lean and that your goal is a claim about the actual executable, not just an interesting Lean model. When the deliverable really *is* a Lean program, there is no cross-language translation problem.

## Route 1: write the algorithm as a Lean program

For a simple example, a Lean implementation and a property can look like this:

```lean
def increment (n : Nat) : Nat := n + 1

theorem increment_spec (n : Nat) : increment n > n := by
  simp [increment]
```

`Nat` means nonnegative mathematical integers. Lean can inspect the definition of `increment`, and its kernel checks the proof of `increment_spec`. This example says nothing about overflow of a fixed-width machine integer: that would require a different type and spec. For loops, mutation, exceptions, concurrency, and I/O, you need an appropriate account of those effects too.

If the original was a Python, Rust, or C function, a Lean reimplementation by itself only proves the **Lean version**. To claim that the original executable satisfies the property, you additionally need a credible correspondence between their behaviors: perhaps a verified compiler, a proved equivalence/simulation theorem, or a narrowly justified translation process. “These two snippets look alike” is not such a theorem. Sometimes the intended workflow is instead to make Lean the *source of truth* and generate executable code from it; then the generated-code correctness becomes the question.

This route suits new algorithms you can implement in Lean, pure functions, and mathematical components with a small, controlled execution boundary. It is not automatically suited to proving what an independently compiled C kernel does.

**Vero** illustrates this world, with a qualification. Its benchmark supplies curated, multi-module Lean 4 repositories with frozen interfaces/specifications. In *code-and-proof* mode, an agent fills in Lean implementations **and** proofs; in *proof-only* mode it proves specifications against given reference implementations. The site describes 43 instances, 743 scored APIs, and 2,705 specifications. Source projects (including Dafny, Verus, Coq, and Python) are **translated and reviewed** to make Lean benchmark instances. Success means the submitted Lean project builds and its obligations are checked under the benchmark's rules; it does **not** by itself prove that an unrelated upstream binary is equivalent to the translated Lean program. The site's example of mutually incompatible specs is also a warning: specification authoring and review are real verification work. [2]

## Route 2: embed the existing program's execution in Lean

Here Lean contains a *description of another language or machine*. A deep embedding might define syntax such as “load,” “add,” “store,” and “jump,” plus a relation saying how one machine state becomes the next. The actual program can then be represented as a syntax tree, compiler IR, bytecode, or even the bytes of a machine-code image. You prove a theorem about what the represented program does under those execution rules. For example, the informal goal might be:

> For every allowed starting state and every execution of these binary bytes, whenever the program produces a result, the result has the required value and it never enters a forbidden state.

This is a sketch, **not** a theorem verified by Lean. A real theorem must define “allowed,” “execution,” “result,” and “forbidden.” It must also decide whether it promises eventual completion (liveness) or merely that no bad state occurs (safety).

“DSL” and “IR” name different things. A **DSL** is a language tailored to a domain, often designed for easy reasoning. An **IR** is a program representation used between source and machine code, such as a compiler's control-flow graph. Either can be embedded in Lean; a verifier can also go straight to machine code. A *deep embedding* explicitly represents program syntax and its execution rules as data and relations. A *shallow embedding* instead maps constructs directly to existing Lean functions/relations. The two routes are therefore useful pictures, **not** an exhaustive or mutually exclusive taxonomy: translating a source program into a Lean function is itself a kind of shallow embedding, and projects often mix layers.

**xv6iris/MachCSL** is the striking counterexample to “verification means translating the C source to Lean.” The [paper's main development][3] was checked in **Rocq** (formerly Coq), not Lean. It reasons about the actual compiled xv6 kernel and user-program images, using a RISC-V hardware model derived from Sail plus models of memory, devices, interrupts, and power cuts. Iris-style concurrent separation logic lets the authors reason compositionally about resources owned by different cores and devices. The published paper reports an application-level property about console output after `echo hello world`, ten xv6 bugs found, and one bug found in the Sail model. It explicitly does **not** claim liveness or information-flow noninterference. [3]

There is now also a **Lean 4 port**, on the repository's `lean` branch; the repository says the original Rocq/Iris development is on `main`. The Lean port describes checked-in kernel and user ELF bytes as Lean data, the Sail RISC-V semantics compiled to Lean, an Iris-based machine-level logic, per-function contracts, and top-level theorems about machine executions and traces. It pins particular image revisions: the theorem is about those images under the specified hardware/device model, not every future build of arbitrary xv6 sources. Reading the paper as if its original 93-day Rocq project had instead been carried out in Lean would be misleading. [1], [3]

The same high-level challenge occurs in an IR-based pipeline, but **Creusot is not a Lean verifier**. The linked 2022 paper translates Rust's compiler **MIR** into Why3's WhyML/MLCFG representation. Why3 generates proof obligations and sends them to automated solvers such as Z3. Programmers write preconditions, postconditions, and (when needed) loop invariants in Creusot's Pearlite spec language. Its treatment of Rust mutable borrows relies on Rust's ownership/borrow checking and uses a logical “future value” (*prophecy*) for a borrow. The paper explicitly notes a remaining gap between an earlier mechanized soundness result for a simplified language and Creusot's implementation. This example clarifies what “translate into an IR, then prove” can mean, while showing that the destination need not be Lean. [4] (§§2–3, 6)

| Example | What is proved about | How the source reaches the prover | Important boundary |
| --- | --- | --- | --- |
| Lean `increment` above | A Lean function | Written directly in Lean | Another-language implementation is not thereby proved |
| Vero [2] | Curated Lean implementations/specs | Human-reviewed translation of source projects into Lean benchmark instances | Benchmark success is a Lean-project result |
| xv6iris [1], [3] | Pinned machine-code images, under a hardware model | Binary bytes represented as proof-assistant data; Sail hardware semantics | Model, device assumptions, image identity, and specified properties matter |
| Creusot [4] | A supported Rust program under its logical translation | Rust MIR → Why3 representation → proof obligations → SMT solvers | Not Lean; relies on translation, borrow checker, contracts, and trusted assumptions |

## How a human spec becomes a trustworthy claim

Suppose you ask: “Sorting preserves all items and returns them in ascending order.” First decide which inputs are allowed and what “same items” means (a permutation, including duplicates). Then write a formal condition relating input and output. The proof needs to establish **both** sortedness and permutation; proving only sortedness would allow a function that always returns `[]`. For a mutable API, add what may change; for an OS, say which observable events and failure cases count. A spec that says `True`, or has an impossible precondition, can be easy to prove and useless.

Whichever route you choose, ask these questions:

1. **Which artifact?** Lean definition, original source, compiler IR, a particular executable, or *all* future versions? The closer the artifact is to deployed machine code, the less compiler correspondence you must assume, but the larger the proof usually becomes.
2. **Which behaviors?** Mathematical integers versus fixed-width overflow; normal returns versus crashes; one thread versus every scheduling interleaving; modeled hardware versus a physical board.
3. **Which property?** Safety (“nothing bad happens”), functional correctness (“any result has this value”), termination/progress (“something good eventually happens”), or information-flow security? They are different theorems.
4. **Which trusted links?** Check the spec, semantics, translation/compiler or loaded bytes, imported axioms, hardware assumptions, and proof checker. A proof checker only checks the theorem it was actually given. Automated tools or AI can *find* proofs, but acceptance by the kernel does not repair an incorrect spec or model.

## What remains uncertain

I do not know which source language, program, runtime, hardware, or exact property you want to verify. That choice determines whether a direct Lean implementation, an embedded interpreter/IR, a binary-level proof, or a different tool such as Creusot makes sense. I also cannot infer from these sources that the Lean port of xv6iris and the paper's Rocq proof have identical statements, trusted bases, or pinned binaries in every detail; the repository calls the Rocq development the design authority, so compare the specific theorem and revision before treating them as interchangeable. “Verified” should always be read with the theorem statement and assumptions beside it.

## If you remember three things

1. Proving a Lean rewrite correct is **not yet** proving that an independent non-Lean program behaves the same way.
2. Embedding syntax, IR, or machine code lets you reason about existing programs, but the model and any translation still need justification.
3. The human's spec is part of what you trust: Lean proves what it says, not necessarily what was intended.

## Sources

- [1] — Repository README (`lean` branch), with the original Rocq development on `main`.
- [2] — Vero project and benchmark description.
- [3] — Kaashoek and Zeldovich, *Extending concurrent separation logic to the hardware level to verify the xv6 OS kernel on RISC-V with AI agents* (v2, 2026).
- [4] — Denis, Jourdan, and Marché, *Creusot: a Foundry for the Deductive Verification of Rust Programs* (2022).

[1]: https://github.com/mit-pdos/xv6iris "xv6iris repository (README on the lean branch and original main branch)"
[2]: https://vero.verina.io/ "Vero project site and benchmark description"
[3]: https://arxiv.org/abs/2609.04043 "Kaashoek and Zeldovich, Extending concurrent separation logic to the hardware level to verify the xv6 OS kernel on RISC-V with AI agents (v2, 2026)"
[4]: https://inria.hal.science/hal-03737878v1/document "Denis, Jourdan, and Marché, Creusot: a Foundry for the Deductive Verification of Rust Programs (2022)"
