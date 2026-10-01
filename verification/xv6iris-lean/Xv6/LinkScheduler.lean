/-
Link `scheduler`: the sealed proof instance clients import.
-/
import Xv6.ProofScheduler
import Xv6.LinkSwtch
import Xv6.LinkAcquire
import Xv6.LinkRelease

namespace Xv6

/-- The proved `scheduler` interface. -/
theorem Scheduler : SCHEDULER := scheduler_proof Swtch Acquire Release

end Xv6
