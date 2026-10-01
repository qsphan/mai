(* LinkUartinitone.v -- instantiates the Uartinitone proof against its
   callees: the PORT-INDEXED UART device leaves (Uart) and initlock.  Sealed,
   so this is the only place the three ever meet.

   BEWARE: FORGETTING THE SECOND ARGUMENT HERE COMPILES.  Rocq accepts PARTIAL
   functor application, so `UartinitoneProof Uart` against the two-argument
   functor silently defines [Uartinitone] as a FUNCTOR rather than a module --
   this file goes green and the mistake surfaces only downstream, where
   [LinkUartinit]'s `UartinitProof Uartinitone` rejects it. *)
Require Import LinkUart LinkInitlock ProofUartinitone.

Module Uartinitone := UartinitoneProof Uart Initlock.
