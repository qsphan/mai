(* ====================================================================== *)
(* EchoFsPure.v -- THE ECHO APPLICATION'S PURE FILE-SYSTEM CLAIM.          *)
(*                                                                        *)
(* [echo_fs_pure] alone: the conjunction of the three era-0 pins, as a     *)
(* [Prop].  It owns nothing -- that is the whole point of its being pure   *)
(* (see [AppEcho.v]'s note) -- so it needs only the three pin files that   *)
(* state its conjuncts.                                                   *)
(*                                                                        *)
(* IT IS HERE BECAUSE [UInitSh.v] NAMES IT IN A STATEMENT.  The record     *)
(* /init hands /sh ([init_sh_slot_core]) carries the WHOLE pins law and    *)
(* each consumer projects its own conjunct (lane E4), so /sh's proof must  *)
(* name this [Prop] -- but it wants nothing else of the echo application:  *)
(* not the ghosts, not the invariant, not the movers.  Parked in           *)
(* [AppEcho.v] it put that whole application -- and, through it,           *)
(* [FsConsPin] -- in front of /sh.                                        *)
(*                                                                        *)
(* [AppEcho.v] RE-EXPORTS this file, so every existing importer is         *)
(* unchanged.                                                             *)
(* ====================================================================== *)
Require Import FsInitPinBoot.      (* [era0_pins]       *)
Require Import FsShPin.            (* [era0_sh_pins]    *)
Require Import FsEchoPin.          (* [era0_echo_pins]  *)
Require Import FsAbsDefs.          (* [aview]           *)

(* /init, /sh and /echo are the image's, path and content, on the abstract
   state's VIEW.  Per-inum rather than "the map is the image's" on purpose:
   the durable snapshot pins a state per inum and no whole-map equality
   exists (fs-syscall-specs.md lane D, gap (3)).  This half of the claim is
   PURE -- it owns nothing -- so it is a [Prop] and the predicate embeds
   it. *)
Definition echo_fs_pure (av : aview) : Prop :=
  era0_pins av /\ era0_sh_pins av /\ era0_echo_pins av.
