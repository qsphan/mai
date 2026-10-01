(* ====================================================================== *)
(* LogEntryDefs.v -- THE CONSOLE INPUT LOG'S ENTRY TYPE.                   *)
(*                                                                        *)
(* [log_entry] and its three projections, over nothing but [mobs] and a    *)
(* byte.  The TSO points-to layer ([RiscvPtsto.v]) names this type to      *)
(* state the console log's resource and wants nothing else from the log's  *)
(* theory; parked in [ConsLog.v] it put that whole theory -- the echo      *)
(* shapes, the erase arms, the history lemmas -- in front of it.           *)
(*                                                                        *)
(* [ConsLog.v] RE-EXPORTS this file, so every existing importer sees these *)
(* four names exactly where it saw them before.                           *)
(* ====================================================================== *)
From Stdlib Require Import ZArith List.
From stdpp Require Import list bitvector.definitions.
Require Import RiscvLang.        (* [mobs] *)

Definition log_entry : Type := (list mobs * bv 8 * list (bv 8))%type.
Definition le_hist (e : log_entry) : list mobs := e.1.1.
Definition le_byte (e : log_entry) : bv 8 := e.1.2.
Definition le_echo (e : log_entry) : list (bv 8) := e.2.

(* THE CONSOLEINTR ARM IN PROGRESS (redesign R2): the byte [le_byte] was
   accepted at [le_hist], the echo [le_echo] was chosen for it, and [ca_sent]
   of that echo's bytes have already reached the wire.  [None] between arms.
   It lives HERE, and not in [ConsLog.v], for the reason [log_entry] does:
   the ghost camera ([Xv6Cameras.uartGhostG]) and the port's own ghost
   ([WpUart.uart_arm]) name the type and want nothing of the log's theory. *)
Definition cons_arm : Type := (log_entry * nat)%type.
Definition ca_hist (a : cons_arm) : list mobs := le_hist a.1.
Definition ca_byte (a : cons_arm) : bv 8 := le_byte a.1.
Definition ca_echo (a : cons_arm) : list (bv 8) := le_echo a.1.
Definition ca_sent (a : cons_arm) : nat := a.2.

(* THE CONSOLE HISTORY (redesign R2): everything the console port's one
   resource is about -- the bytes the UART has accepted, the accepted-input
   log, what has been delivered to processes, and the arm in progress.
   Here, and not in [ConsLog.v], because [RiscvPtsto.v] names the TYPE for
   the fixed record's field and wants none of the log's theory.  The events
   that move it, and their well-formedness, stay in [ConsLog.v]. *)
Record cons_hist := MkCH {
  ch_acc : list (bv 8);                    (* every byte the console UART accepted *)
  ch_log : list log_entry;                 (* every accepted input, with what was echoed *)
  ch_dl  : list (list mobs * bv 8);        (* the inputs delivered to processes *)
  ch_arm : option cons_arm                 (* the arm in progress *)
}.
