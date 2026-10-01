(* ===================================================================== *)
(* UserPermDenied.v -- THE NEGATIVE HALF OF THE LEAF-BIT TRANSFER: a page  *)
(* the key's projection maps WITHOUT W is a page the model DENIES a store  *)
(* on.                                                                     *)
(*                                                                         *)
(* [UserPerm] §3/§4 carries the three POSITIVE transfers -- an X page is a *)
(* fetch-ok leaf, a W page is a store-ok leaf, a mapped page is a load-ok  *)
(* leaf ([UserPerm.uleaf_fetch_of_bits] / [uleaf_store_of_bits] /          *)
(* [uleaf_load_of_bits], and [perm_of_X] / [perm_of_W] / [perm_of_R] over  *)
(* them).  Every consumer so far REFUTES a fault, so the negative          *)
(* direction was never needed: [UkStore.wp_uk_store_later] discharges its  *)
(* fault witness by [UserPtTree.uleaf_ok_denied_excl], i.e. by showing the *)
(* leaf is store-ok and therefore not denied.                              *)
(*                                                                         *)
(* A program that faults ON PURPOSE needs the other direction.  sh's       *)
(* forked child stores through a NULL [malloc] result; VA 0 is sh's TEXT   *)
(* page, mapped R+X and NOT W, so the store traps with cause 15 and the    *)
(* kernel kills the process (app-echo.md, lane SELF-KILL).  To WALK that   *)
(* store the leaf must PRODUCE [UserPtTree.u_fault_flavor]'s denied arm    *)
(* out of the one thing the process can read off its own key: the page's   *)
(* permission bits.  That is what this file is.                            *)
(*                                                                         *)
(* WHY IT IS A LEAF FILE AND NOT A SECTION OF [UserPerm]: [UserPerm] sits  *)
(* near the bottom of the tree (the engine, every walk and every program   *)
(* file are over it), and an additive lemma there costs the whole cone.    *)
(* Nothing yet depends on this file, so it costs its own compile.  Fold it *)
(* into [UserPerm] on the next occasion that file is edited for another    *)
(* reason.                                                                 *)
(*                                                                         *)
(* THE REFUTATION IS [UserPerm]'s, MIRRORED.  With the six permission bits *)
(* fixed, each verdict is a closed computation; the enumeration over       *)
(* [UserPerm.flags6] is the same 64-way split, and the case that has to    *)
(* die is the OK one rather than the denial.  The verdict is compared as a *)
(* BOOLEAN of the result ([is_success_result], [UserPerm.is_noperm_result]'s *)
(* twin) for that lemma's reason: the refutation's proof term must not     *)
(* carry the normal form of the machine state.                             *)
(* ===================================================================== *)
From Stdlib Require Import ZArith Bool Lia List.
From stdpp Require Import gmap sets bitvector.definitions.
Require Import SailStdpp.ConcurrencyInterface SailStdpp.ConcurrencyInterfaceBuiltins SailStdpp.ConcurrencyInterfaceTypes SailStdpp.Operators_mwords.
Require Import SailStdpp.Base SailStdpp.TypeCasts SailStdpp.Values SailStdpp.MachineWord.
Require Import RiscvLang RiscvPtsto RiscvFetchExec.
Require Import WpDecodeBridge.  (* [dstate] -- the concrete state to refute at *)
Require Import PtAdBits PtTree.
Require Import UserPtTree.
Require Import ProcPtOwn.
Require Import UserPerm.
Require Import UmodeMem.  (* [uva_canon] -- the fault wrapper's first arm *)
Require Import Riscv.rv64d_types Riscv.rv64d.
Local Open Scope Z_scope.
Import Defs.

(* ===================================================================== *)
(* §1 The verdict as a boolean, and the case refutation.                  *)
(* ===================================================================== *)

(* [UserPerm.is_noperm_result]'s twin, for refuting an OK verdict. *)
Definition is_success_result (r : option (PTE_Check * mstate)) : bool :=
  match r with
  | Some (PTE_Check_Success _, _) => true
  | _ => false
  end.

(* [UserPerm]'s [perm_case_refute], with the OK side killed instead of the
   denial.  [Hok] is a [pte_check_ok] at the case's flag byte. *)
Local Ltac perm_case_refute_ok Hlt Hv Hok :=
  exfalso;
  specialize (Hok (mword_of_int 0) (mword_of_int 0) false false
                 (dstate MENVCFG_S User));
  unfold pte_check_ok, Mk_PTE_Flags in Hok;
  rewrite (pte_flags_byte_of_flags6 _ _ _ Hlt Hv) in Hok;
  rewrite pte_set_ad_ext in Hok;
  match goal with
  | Hx : ext_bits_of_PTE _ = _ |- _ => rewrite Hx in Hok
  end;
  match goal with
  | Hg : flags6 _ = _ |- _ => rewrite Hg in Hok
  end;
  apply (f_equal is_success_result) in Hok;
  vm_compute in Hok; discriminate Hok.

(* ===================================================================== *)
(* §2 THE BIT TRANSFER: a user leaf without W denies a store.             *)
(* ===================================================================== *)

(* [UserPerm.uleaf_store_of_bits]'s mirror.  [upt_acc_wf]'s disjunction is
   what makes it a case analysis rather than a model evaluation: the entry
   is store-ok or store-denied, and with U set and W clear the OK side is
   refutable case by case. *)
Lemma uleaf_store_denied_of_bits (w : mword 64) :
  uleaf_wf w -> pte_bit w 4 = true -> pte_bit w 2 = false ->
  uleaf_denied (Store Data) w.
Proof.
  intros Hwf Hu Hw.
  pose proof (uleaf_wf_ext w Hwf) as Hext.
  destruct (uleaf_wf_lt w Hwf) as [Hlt Hv].
  destruct Hwf as (_ & Hacc).
  destruct (Hacc (Store Data) ltac:(right; right; left; reflexivity)) as [Hok | Hd];
    [ | exact Hd ].
  rewrite <- (flags6_bit w 4 ltac:(lia)) in Hu.
  rewrite <- (flags6_bit w 2 ltac:(lia)) in Hw.
  destruct (UserPerm.z64_cases (flags6 w) (flags6_range w)) as
    [Hg|[Hg|[Hg|[Hg|[Hg|[Hg|[Hg|[Hg|[Hg|[Hg|[Hg|[Hg|[Hg|[Hg|[Hg|[Hg|[Hg|[Hg|[Hg|[Hg|[Hg|[Hg|[Hg|[Hg|[Hg|[Hg|[Hg|[Hg|[Hg|[Hg|[Hg|[Hg|[Hg|[Hg|[Hg|[Hg|[Hg|[Hg|[Hg|[Hg|[Hg|[Hg|[Hg|[Hg|[Hg|[Hg|[Hg|[Hg|[Hg|[Hg|[Hg|[Hg|[Hg|[Hg|[Hg|[Hg|[Hg|[Hg|[Hg|[Hg|[Hg|[Hg|[Hg|Hg]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]].
  all: rewrite Hg in Hu, Hw.
  all: try discriminate Hu.
  all: try discriminate Hw.
  (* the surviving cases are the sixteen with U set and W clear; each is a
     closed verdict.  Separate [all:] passes, not one chained tactic --
     [UserPerm.uleaf_fetch_of_bits]'s note. *)
  all: perm_case_refute_ok Hlt Hv Hok.
Qed.

(* ===================================================================== *)
(* §3 THE PROJECTION TRANSFER: a key page without W is a mapped, denied    *)
(* leaf of the table.                                                      *)
(* ===================================================================== *)

(* [UserPerm.perm_of_W]'s mirror, and it does NOT take the entry as an
   argument the way that one does: a page without W CANNOT be a filled
   lazy page ([UserPerm.perm_fill] hands out [uperm_rw], whose W is set),
   so the projection's own entry already proves the table maps it.  That
   is what makes this usable by a leaf whose only premise is the key. *)
Lemma perm_of_notW_denied (pt : uptd) (sz : Z) (p : mword 27) (q : uperm) :
  proc_pt_wf pt ->
  perm_of (ud_um pt) sz !! p = Some q -> up_W q = false ->
  exists w : mword 64, ud_um pt !! p = Some w /\ uleaf_denied (Store Data) w.
Proof.
  intros Hwf Hl Hnw.
  destruct (perm_of_lookup_Some _ _ _ _ Hl) as [(w & Hw & Hu & _ & ->) | [_ ->]].
  - exists w. split; [ exact Hw | ].
    apply (uleaf_store_denied_of_bits w (proc_pt_wf_uleaf_wf pt p w Hwf Hw) Hu).
    exact Hnw.
  - discriminate Hnw.
Qed.

(* ...at an ADDRESS, which is the form a store leaf reads off its key
   ([UserPerm.uperm_at] is the projection at [svpn_of va]). *)
Lemma uperm_at_notW_denied (pt : uptd) (sz : Z) (va : mword 64) (q : uperm) :
  proc_pt_wf pt ->
  uperm_at (perm_of (ud_um pt) sz) va = Some q -> up_W q = false ->
  exists w : mword 64,
    ud_um pt !! svpn_of va = Some w /\ uleaf_denied (Store Data) w.
Proof.
  intros Hwf Hl Hnw. unfold uperm_at in Hl.
  exact (perm_of_notW_denied pt sz (svpn_of va) q Hwf Hl Hnw).
Qed.

(* ===================================================================== *)
(* §4 THE FAULT WITNESS -- what a deliberate store hands the engine.       *)
(* ===================================================================== *)

(* [UserPtTree.u_fault_flavor]'s DENIED arm, produced (not refuted) out of
   the key.  This is the premise [UkStore.uk_store_fault_post_fetch] takes,
   and the whole point of the file: a store to a page the process's own
   permission map shows without W faults, and the process can say so
   before it issues the store. *)
Lemma u_fault_flavor_store_notW (pt : uptd) (sz : Z) (va : mword 64) (q : uperm) :
  proc_pt_wf pt -> uva_canon va ->
  uperm_at (perm_of (ud_um pt) sz) va = Some q -> up_W q = false ->
  u_fault_flavor (Store Data) (ud_tfp pt) (ud_um pt) va.
Proof.
  intros Hwf Hcanon Hl Hnw.
  destruct (uperm_at_notW_denied pt sz va q Hwf Hl Hnw) as (w & Hw & Hden).
  right. right. split; [ exact Hcanon | ].
  exists w. split; [ | exact Hden ].
  right. right. exact Hw.
Qed.

(* ...and the shape the ENGINE's premise list is written at, where the key's
   projection is already named [π] and the table equation is the walk's own
   ([UkStore.wp_uk_store_later]'s [Hpm]).  Stated so that a caller supplies
   the permission fact and nothing about the table. *)
Lemma u_fault_flavor_store_key (pt : uptd) (sz : Z)
    (pi : gmap (mword 27) uperm) (va : mword 64) :
  proc_pt_wf pt -> perm_of (ud_um pt) sz = pi -> uva_canon va ->
  (exists q : uperm, uperm_at pi va = Some q /\ up_W q = false) ->
  u_fault_flavor (Store Data) (ud_tfp pt) (ud_um pt) va.
Proof.
  intros Hwf Hpm Hcanon (q & Hl & Hnw). subst pi.
  exact (u_fault_flavor_store_notW pt sz va q Hwf Hcanon Hl Hnw).
Qed.
