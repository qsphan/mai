/-
Link `kalloc`: the proof instance clients import.  `kalloc` calls
`acquire`, `release` and `memset`; the interfaces stay parameters here, so
a client may close them with the linked ones (`LinkAcquire`,
`LinkRelease`, `LinkMemset`) or with its own.
-/
import Xv6.ProofKalloc

namespace Xv6

/-- The proved `kalloc` interface, given `acquire`, `release` and `memset`. -/
theorem Kalloc (AC : ACQUIRE) (RE : RELEASE) (MS : MEMSET) : KALLOC := kalloc_proof AC RE MS

end Xv6
