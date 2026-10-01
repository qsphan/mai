(* LinkUartinit.v -- instantiates the Uartinit proof against its ONE callee.

   At XV6_REV 163d39b uartinit is a two-call wrapper around [uartinitone], so
   this file links exactly one thing: the UART device leaves and [initlock]
   are reached through [Uartinitone]'s own seal ([LinkUartinitone]) and never
   appear here. *)
Require Import LinkUartinitone ProofUartinit.

Module Uartinit := UartinitProof Uartinitone.
