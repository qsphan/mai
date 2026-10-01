(* StringBytes.v -- a Rocq [string] as a list of bytes: [string_bytes] is its
   characters, [cstring_bytes] those characters followed by the terminating
   NUL.  Every resident string in the tree is measured against that shape,
   which mentions neither a machine nor separation logic, so it does not
   belong beside [RiscvPtsto.string_pointsto], which merely resides it.

   RiscvPtsto.v re-exports this file, so whatever it requires is loaded across
   most of the tree: keep it to stdlib and stdpp.  That is why
   [PStringBytes.v] (the same reading for a hex PrimString) stays separate, and
   why the inverse direction is [CstringInv.v], which needs PrintkFmt. *)

From Stdlib Require Import ZArith List Ascii String.
From stdpp Require Import list bitvector.definitions.

Local Open Scope Z_scope.

(* the characters of [s] as bytes (no terminator) *)
Fixpoint string_bytes (s : string) : list (bv 8) :=
  match s with
  | String.EmptyString => []
  | String.String c s' => Z_to_bv 8 (Z.of_N (Ascii.N_of_ascii c)) :: string_bytes s'
  end.

(* the C representation of [s]: its characters followed by the NUL byte *)
Definition cstring_bytes (s : string) : list (bv 8) :=
  string_bytes s ++ [Z_to_bv 8 0].

(* the terminating NUL is the last byte, so a C string resides |s|+1 bytes --
   which is what every resident-string resource counts. *)
Lemma cstring_bytes_length s :
  length (cstring_bytes s) = S (String.length s).
Proof using .
  unfold cstring_bytes. rewrite length_app. simpl.
  induction s as [|c s IH]; simpl; [reflexivity | rewrite IH; reflexivity].
Qed.

(* [cstring_bytes] on a cons: the NUL rides at the END, so prefixing a
   character prefixes its byte.  Definitional, and it is what keeps an
   induction over a C string from having to reassociate anything. *)
Lemma cstring_bytes_cons (c : ascii) (s : string) :
  cstring_bytes (String c s)
  = Z_to_bv 8 (Z.of_N (Ascii.N_of_ascii c)) :: cstring_bytes s.
Proof using . reflexivity. Qed.
