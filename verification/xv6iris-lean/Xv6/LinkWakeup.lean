/-
Link `wakeup`: the sealed proof instance clients import.
-/
import Xv6.ProofWakeup
import Xv6.LinkAcquire
import Xv6.LinkRelease

namespace Xv6

/-- The proved `wakeup` interface. -/
theorem Wakeup : WAKEUP := wakeup_proof Acquire Release

end Xv6
