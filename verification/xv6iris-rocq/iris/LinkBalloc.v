(* LinkBalloc.v -- the only file where balloc's proof meets its callees'.
   All four (bread, log_write, brelse, memset -- the last two of which are
   what the INLINED bzero at +0x4c runs) are PROVEN, so nothing here is
   assumed: [ProofBalloc.v] replaced the [Axiom] this file used to carry,
   and it is the single file the bmap cone's one assumption lived in.

   balloc's two dead arms -- the [beqz a5,+0xf6] at +0x12 (refuted from
   [0 < size]) and the second outer iteration at +0x98 (refuted from
   [size <= BPB]) -- are refuted inside the proof, so no panic contract is
   instantiated here either.

   *)
Require Import LinkBread LinkLogWrite LinkBrelse LinkMemsetArray ProofBalloc.
Require Import LinkPrintk.

(* the whole-function memset spec [MEMSET] is [MemsetArray] (LinkMemsetArray),
   not the [MEMSET_PARTS] module [Memset] *)
Module Balloc := BallocProof Bread LogWrite Brelse MemsetArray PrintkGen.
