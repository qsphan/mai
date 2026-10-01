/-
Link `freeproc`: the sealed proof instance clients import.
-/
import Xv6.ProofFreeproc

namespace Xv6

/-- The proved `freeproc` interface, given `kfree`, `proc_freepagetable`,
`acquire` and `release`. -/
theorem Freeproc (KF : KFREE) (PFP : PROC_FREEPAGETABLE) (AC : ACQUIRE) (RE : RELEASE) :
    FREEPROC := freeproc_proof KF PFP AC RE

end Xv6
