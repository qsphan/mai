/-
Link `swtch`: the sealed proof instance clients import.
-/
import Xv6.ProofSwtch

namespace Xv6

/-- The proved `swtch` interface. -/
theorem Swtch : SWTCH := swtch_proof

end Xv6
