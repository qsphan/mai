(* PipesPair.v -- THE MINIMAL PURE OUTCOMES OF A PIPE'S TWO ENDS, and how
   they pair (design pipes-general SS3.2, cut C4's local stand-in for cut
   C2's [PipesDisc]).

   A node of the command tree reads one pipe off its two children's exit
   payloads ([PipeProto.node_reading]).  What the WRITER end did and what
   the READER end saw are recorded here as data, so that the reading's
   conclusion is a pure relation between them, [pipe_pair], and not a
   list of resource arms.  Cut C2 owns the full per-stage outcome model
   (its [st_out]/[stage_out] carry the console stream beside these); it
   is expected to take these two types and [pipe_pair] over, or to prove
   its own versions equivalent.

   THE LOOSE CORNER (design SS2.4, ruled (B)) is NOT here.  It is about
   the CONSOLE's prefix at the moment a middle cat's diagnostic starts,
   i.e. about a state in the middle of a round.  At the END of a round
   the pairing below is exact: a halted writer and an end-of-file reader
   on the same pipe are refuted by the protocol's two enders
   ([PipeProto.pipe_body_P6U]), so [WrHalt] beside [RdEof] is [False].
   A model that admits the corner may weaken this to [True]; it follows.

   Pure; nothing but stdpp. *)
From stdpp Require Import list bitvector.definitions.

(* WHAT THE READER END SAW: an end of file after exactly the bytes [D]
   (the frozen contents), or nothing it can vouch for -- its reader exec
   failed, halted on its own write error, or was killed. *)
Inductive rd_out : Type :=
  | RdEof (D : list (bv 8))
  | RdGone.

(* WHAT THE WRITER END DID: wrote all of [D] (its whole input, for a
   middle cat; the whole line, for echo), stopped after [D] because the
   read end was shut, or never wrote (its exec failed, or it read an
   empty input and so had nothing to copy). *)
Inductive wr_out : Type :=
  | WrAll (D : list (bv 8))
  | WrHalt (D : list (bv 8))
  | WrNone.

(* THE PAIRING: an end of file sees exactly what the writer wrote; a gone
   reader constrains nothing. *)
Definition pipe_pair (w : wr_out) (r : rd_out) : Prop :=
  match w, r with
  | WrAll D, RdEof D' => D' = D
  | WrNone, RdEof D' => D' = []
  | WrHalt _, RdEof _ => False
  | _, RdGone => True
  end.

(* EVERY BYTE A PIPE CARRIES IS A BYTE OF THE LINE [L]: the protocol's
   (P1), read at the two outcomes. *)
Definition wr_in (L : list (bv 8)) (w : wr_out) : Prop :=
  match w with
  | WrAll D | WrHalt D => D `prefix_of` L
  | WrNone => True
  end.

Definition rd_in (L : list (bv 8)) (r : rd_out) : Prop :=
  match r with
  | RdEof D => D `prefix_of` L
  | RdGone => True
  end.

(* ---- the readings a consumer does on a pair ---- *)

Lemma pipe_pair_eof (w : wr_out) (D : list (bv 8)) :
  pipe_pair w (RdEof D) -> w = WrAll D \/ (w = WrNone /\ D = []).
Proof using .
  destruct w as [D' | D' |]; cbn [pipe_pair]; intros H.
  - subst. left. reflexivity.
  - destruct H.
  - right. split; [ reflexivity | exact H ].
Qed.

Lemma pipe_pair_gone (w : wr_out) : pipe_pair w RdGone.
Proof using . destruct w; exact I. Qed.

Lemma pipe_pair_halt (D : list (bv 8)) (r : rd_out) :
  pipe_pair (WrHalt D) r -> r = RdGone.
Proof using . destruct r; cbn [pipe_pair]; [ intros [] | reflexivity ]. Qed.

(* the one-pipe corner the landed round lives at: echo wrote the whole
   line and cat saw it, so what cat saw IS the line *)
Lemma pipe_pair_all_line (L D : list (bv 8)) :
  pipe_pair (WrAll L) (RdEof D) -> D = L.
Proof using . cbn [pipe_pair]. exact id. Qed.
