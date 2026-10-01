/-
Link `safestrcpy`: the sealed proof instance clients import.  `safestrcpy`
calls nothing, so the interface closes with no callee premises.
-/
import Xv6.ProofSafestrcpy

namespace Xv6

/-- The proved `safestrcpy` interface (the `n = 16` case). -/
theorem Safestrcpy : SAFESTRCPY := safestrcpy_proof

end Xv6
