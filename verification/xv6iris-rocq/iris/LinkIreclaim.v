(* LinkIreclaim.v -- the only file where ireclaim's proof meets its callees'.
   All eight (bread, brelse, iget, begin_op, ilock, iunlock, iput and end_op)
   are PROVEN, so nothing here is assumed:

   - bread / brelse arrive through their own Link files, the same instances
     ialloc, iupdate and balloc use;
   - iget arrives through LinkIget.v (acquire / release), i.e. the same
     instance dirlookup's tail call and ialloc's claim arm use;
   - begin_op / end_op arrive through LinkBeginOp.v / LinkEndOp.v, the pair
     kexit's cwd release already drives;
   - ilock / iunlock / iput arrive through their own Link files.

   ireclaim's TWO dead arms are refuted inside the proof, so no panic
   contract is instantiated here:

   - the [bgeu a5,a4] at +0x0a -- the empty-region exit, which would return
     through the SECOND [c.jr ra] at +0xc6 with the frame NEVER pushed -- is
     refuted from the contract's [1 < ninodes], exactly as ialloc's +0x12 arm
     and balloc's [beqz a5] at the same offset are;
   - the [beq s3,zero] at +0x50 -- the C source's [if(ip)] -- is refuted from
     iget's POSTCONDITION ([a0 = ientry k] with [k < NINODE], hence
     [IcacheRefDefs.ientry_ne_zero]) and NOT from any premise of this contract.
     That is the one refutation in this cone that a caller cannot see.

   The [kernel_data] / [panic_env] the contract takes are threaded to all
   eight callees, whose own panic arms are discharged against [Panic].

   *)
Require Import LinkBread LinkBrelse LinkIget LinkBeginOp
                LinkIlock LinkIunlock LinkIput LinkEndOp
                ProofIreclaim.
Require Import LinkPrintk.

Module Ireclaim := IreclaimProof Bread Brelse Iget BeginOp
                                 Ilock Iunlock Iput EndOp PrintkGen.
