/-
Link `initlock`: the sealed proof instance clients import.
-/
import Xv6.ProofInitlock

namespace Xv6

/-- The proved `initlock` interface. -/
theorem Initlock : INITLOCK := initlock_proof

end Xv6
