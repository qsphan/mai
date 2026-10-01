(* ====================================================================== *)
(* HartTranslateM.v -- THE BARE-MODE TRANSLATION WALK, SHARED.             *)
(*                                                                        *)
(* [hfrun_translateAddr_M]: M-mode's identity translation, GENERIC IN THE  *)
(* ACCESS, with the two access-dependent facts left as premises -- so the  *)
(* fetch path and the store path share one walk instead of owning a copy   *)
(* each.  The two reduction tactics its proof is written in come with it.  *)
(*                                                                        *)
(* IT IS HERE AND NOT IN [HartMFetch.v] BECAUSE THE STORE PATH NEEDS IT.   *)
(* [HartMStore.v] names this one lemma out of that file's 44 and nothing   *)
(* else; taking it from the fetch file put the whole fetch tower on the    *)
(* store's critical path.  See claude-notes/design/code-organization.md --  *)
(* “when you need a fact from another function's file, that is the signal  *)
(* to move the fact down, never to add the import”.                       *)
(*                                                                        *)
(* [tr_cbn] / [tr_read] ARE PLAIN [Ltac], NOT [Local]: they were local to  *)
(* the fetch file, which still uses them for its own walks and now reads   *)
(* them back through its [Require Export] of this file.                   *)
(* ====================================================================== *)
From Stdlib Require Import ZArith Lia.
From stdpp Require Import gmap bitvector.definitions.
Require Import SailStdpp.Operators_mwords.
Require Import Riscv.rv64d_types Riscv.rv64d.
Require Import RiscvExtras.        (* [fetch_pa_id] *)
Require Import HartSpan.           (* [hfrun], [hfrun_read], [hfrun_ret] *)

Ltac tr_cbn :=
  cbn beta iota zeta delta
    [Defs.bind Defs.bind0 Interface.iMon_bind Defs.liftR Defs.try_catch
     Defs.catch_early_return Defs.returnm returnM returnR Defs.returnR
     Defs.read_reg Defs.early_return Defs.throw Defs.and_boolM Defs.or_boolM
     andb orb negb not].

Ltac tr_read :=
  rewrite hfrun_read;
  match goal with
  | |- context [ bool_decide ?P ] =>
      rewrite (bool_decide_eq_true_2 P ltac:(assumption))
  end.

(* the identity translation's address, at the spelling [translateAddr]'s
   Bare arm produces *)
Local Lemma zext_pc_id (x : SailStdpp.Values.mword 64) :
  zero_extend' 64 (bits_of_virtaddr (Virtaddr x)) = x.
Proof. exact (fetch_pa_id x). Qed.

(* GENERIC IN THE ACCESS.  The Bare-mode translation is access-agnostic;
   the access enters in exactly two places, and each is a one-line premise
   the caller discharges -- so the fetch and the store share this walk
   rather than owning a copy each. *)
Lemma hfrun_translateAddr_M (D Drw : gset register) (rs : regstate)
    (pc : SailStdpp.Values.mword 64)
    (acc : MemoryAccessType mem_payload) :
  (mstatus : register) ∈ D ->
  (cur_privilege : register) ∈ D ->
  register_lookup cur_privilege rs = Machine ->
  effectivePrivilege acc (register_lookup mstatus rs) Machine
    = returnM Machine ->
  is_shadow_stack_access acc = returnM false ->
  hfrun 8 D Drw rs (translateAddr (Virtaddr pc) acc)
  = Some (Values.Ok (Physaddr pc, PBMT_PMA, init_ext_ptw), rs).
Proof.
  intros HD1 HD2 Hpriv Hep Hss.
  unfold translateAddr. tr_cbn.
  tr_read. tr_cbn.
  tr_read. rewrite Hpriv. tr_cbn.
  rewrite Hep. tr_cbn.
  unfold translationMode.
  change (Instances.generic_eq Machine Machine) with true. tr_cbn.
  rewrite Hss. tr_cbn.
  change (Instances.generic_eq Bare Bare) with true. tr_cbn.
  rewrite zext_pc_id. apply hfrun_ret.
Qed.
