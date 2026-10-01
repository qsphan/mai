/-
Link `growproc`: the proof instance clients import.  It calls `myproc`,
`uvmalloc` and `uvmdealloc`; the interfaces stay parameters here.
-/
import Xv6.ProofGrowproc

namespace Xv6

/-- The proved `growproc` interface, given `myproc`, `uvmalloc` and `uvmdealloc`. -/
theorem Growproc (MP : MYPROC) (UA : UVMALLOC) (UD : UVMDEALLOC) : GROWPROC :=
  growproc_proof MP UA UD

end Xv6
